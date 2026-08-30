"""Tests for the JEPA inference path: run loaders, the curve adapter and the curve predictor.

Covers component wiring, the adapter that decodes an embedding map into a denormalised profile
cube, the loaders that rebuild a JepaModule from the persisted autoencoder and backbone configs
and validate the head width, checkpoint loading with either a numpy or a tensor z axis,
and JepaCurvePredictor stitching predicted and ground-truth curve cubes, sorting and masking the
ground-truth parameters, and feeding the embedding evaluator from a single forward pass."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from pipelines.backbone.dataset.spatial                  import GridInfo
from pipelines.backbone.inference.loader                 import RunLoader
from pipelines.backbone.inference.model_wrapper          import ModelWrapper
from pipelines.backbone.inference.predictor              import Predictor
from configuration.architectures                         import Conv2dImageAutoencoderConfig, MlpAutoencoderConfig, UNetConfig
from pipelines.jepa.inference.loader                     import JepaInferenceModel, JepaParamRunLoader, JepaRunLoader
from pipelines.shared.config.config_persistence          import BackboneModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO
from pipelines.jepa.inference.pipeline                   import JEPA_INFERENCE_COMPONENTS, JEPA_PARAM_INFERENCE_COMPONENTS
from pipelines.jepa.inference.predictor                  import JepaCurvePredictor
from pipelines.jepa.training.trainer                     import JepaModule
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.conftest       import SilentLogger
from tests.jepa.conftest  import DEPTH, EMBEDDING_DIM, HIDDEN_DIM, N_GAUSSIANS, PROFILE_LENGTH, SPATIAL, make_autoencoder
from tools.data.gaussians import GaussianReconstructor


def build_inference_module():
    """Builds a JepaModule pairing a one-by-one convolution backbone with an L2-normalised profile autoencoder."""
    autoencoder = make_autoencoder("l2")
    backbone    = nn.Conv2d(2, EMBEDDING_DIM, kernel_size=1)
    return JepaModule(backbone, profile_autoencoder=autoencoder, image_autoencoder=None)


def test_inference_components_wiring():
    """The curve component set uses the JEPA loader and curve predictor in curve space, while the parameter set stays in parameter space."""
    assert JEPA_INFERENCE_COMPONENTS.loader_cls        is JepaRunLoader
    assert JEPA_INFERENCE_COMPONENTS.predictor_cls     is JepaCurvePredictor
    assert JEPA_INFERENCE_COMPONENTS.param_space       is False
    assert JEPA_PARAM_INFERENCE_COMPONENTS.param_space is True


def test_jepa_inference_model_outputs_denormalized_curve():
    """The adapter turns images into a profile cube of shape (batch, profile_length, height, width) with finite values."""
    module     = build_inference_module()
    normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    adapter    = JepaInferenceModel(module, normalizer).eval()
    images     = torch.randn(2, 2, SPATIAL, SPATIAL)

    with torch.no_grad():
        out = adapter(images)

    assert out.shape == (2, PROFILE_LENGTH, SPATIAL, SPATIAL)
    assert torch.isfinite(out).all()


def test_jepa_inference_model_curve_is_nonnegative():
    """The decoded profile cube carries no negative power."""
    module     = build_inference_module()
    normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    adapter    = JepaInferenceModel(module, normalizer).eval()
    images     = torch.randn(2, 2, SPATIAL, SPATIAL)

    with torch.no_grad():
        out = adapter(images)

    assert (out >= 0.0).all()


def test_jepa_param_run_loader_builds_image_frontend_module(tmp_path):
    """The parameter loader builds a JepaModule with an image-autoencoder front end and no profile branch, emitting the parameter channel count."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()

    image_cfg = Conv2dImageAutoencoderConfig(in_channels=2, embedding_dim=4, base_channels=4, depth=1, downsample_factor=2)
    ImageAutoencoderConfigIO.save(image_cfg, "conv2d_ae", meta_dir)

    backbone_cfg = UNetConfig(in_channels=4, out_channels=6, features=(8, 16))
    BackboneModelConfigIO.save(backbone_cfg, "unet", meta_dir)

    loader = JepaParamRunLoader(tmp_path, logger=SilentLogger())
    module = loader._build_model("unet", in_channels=2, out_channels=6, patch_size=(8, 8))
    module.eval()

    assert isinstance(module, JepaModule)
    assert module.profile_autoencoder is None
    assert module.image_autoencoder   is not None

    with torch.no_grad():
        out = module(torch.randn(2, 2, 8, 8))

    assert out.shape == (2, 6, 8, 8)


def _profile_ae_meta(tmp_path, embedding_dim: int):
    """Persists an MLP profile-autoencoder config of the given embedding width into meta/ and returns that directory."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()

    ae_cfg = MlpAutoencoderConfig(profile_length=PROFILE_LENGTH, embedding_dim=embedding_dim, hidden_dim=HIDDEN_DIM, depth=DEPTH)
    ProfileAutoencoderConfigIO.save(ae_cfg, "mlp_ae", meta_dir)

    return meta_dir


def test_jepa_run_loader_reads_persisted_n_gaussians(tmp_path):
    """The Gaussian count is read from the run summary and the embedding-width head passes validation."""
    _profile_ae_meta(tmp_path, embedding_dim=EMBEDDING_DIM)

    loader = JepaRunLoader(tmp_path, logger=SilentLogger())

    assert loader._persisted_n_gaussians({"n_gaussians": N_GAUSSIANS}) == N_GAUSSIANS

    loader._validate_out_channels(EMBEDDING_DIM, N_GAUSSIANS)


def test_jepa_run_loader_rejects_summary_without_n_gaussians(tmp_path):
    """A run summary lacking n_gaussians raises KeyError rather than guessing."""
    _profile_ae_meta(tmp_path, embedding_dim=EMBEDDING_DIM)

    loader = JepaRunLoader(tmp_path, logger=SilentLogger())

    with pytest.raises(KeyError, match="n_gaussians"):
        loader._persisted_n_gaussians({"out_channels": EMBEDDING_DIM})


def test_jepa_run_loader_rejects_out_channels_that_are_not_the_embedding_width(tmp_path):
    """A parameter-width head is rejected on a curve-coupled run."""
    _profile_ae_meta(tmp_path, embedding_dim=EMBEDDING_DIM)

    loader = JepaRunLoader(tmp_path, logger=SilentLogger())

    with pytest.raises(ValueError, match="profile autoencoder embeds"):
        loader._validate_out_channels(3 * N_GAUSSIANS, N_GAUSSIANS)


def test_param_run_loader_requires_a_parameter_head_width(tmp_path):
    """The parameter loader accepts a 3K-wide head and rejects an embedding-wide one."""
    loader = JepaParamRunLoader(tmp_path, logger=SilentLogger())

    loader._validate_out_channels(3 * N_GAUSSIANS, N_GAUSSIANS)

    with pytest.raises(ValueError, match="parameter head"):
        loader._validate_out_channels(EMBEDDING_DIM, N_GAUSSIANS)


def test_wrap_model_returns_model_wrapper_with_curve_output():
    """The wrapped model is a ModelWrapper turning numpy images into a float32 profile cube."""
    module     = build_inference_module()
    normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))

    loader                    = JepaRunLoader.__new__(JepaRunLoader)
    loader.profile_normalizer = normalizer

    wrapper = loader._wrap_model(module, device="cpu", norm_stats=None, z_axis=None, amp_max=None)

    assert isinstance(wrapper, ModelWrapper)

    images = np.random.randn(2, 2, SPATIAL, SPATIAL).astype(np.float32)
    out    = wrapper(images)

    assert out.shape == (2, PROFILE_LENGTH, SPATIAL, SPATIAL)
    assert out.dtype == np.float32


def test_load_checkpoint_round_trip_numpy_x_axis(tmp_path):
    """A checkpoint holding a numpy z axis restores the axis as float32 along with the epoch metadata."""
    module = build_inference_module()
    z_axis = np.linspace(-4.0, 4.0, PROFILE_LENGTH).astype(np.float32)

    ckpt = {
        "epoch"         : 7,
        "best_val_loss" : 0.123,
        "best_epoch"    : 5,
        "z_axis"        : z_axis,
        "params"        : module.state_dict(),
    }
    path = tmp_path / "best_model.pt"
    torch.save(ckpt, path)

    loader             = JepaRunLoader.__new__(JepaRunLoader)
    loaded, axis, meta = RunLoader._load_checkpoint(loader, path, "cpu")

    assert np.allclose(axis, z_axis)
    assert axis.dtype            == np.float32
    assert meta["epoch"]         == 7
    assert meta["best_epoch"]    == 5
    assert meta["best_val_loss"] == pytest.approx(0.123)


def test_load_checkpoint_accepts_tensor_x_axis(tmp_path):
    """A checkpoint holding a torch z axis restores it as an equal float32 array."""
    module = build_inference_module()
    z_axis = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)

    ckpt = {
        "epoch"         : 1,
        "best_val_loss" : 1.0,
        "best_epoch"    : 1,
        "z_axis"        : z_axis,
        "params"        : module.state_dict(),
    }
    path = tmp_path / "best_model.pt"
    torch.save(ckpt, path)

    loader     = JepaRunLoader.__new__(JepaRunLoader)
    _, axis, _ = RunLoader._load_checkpoint(loader, path, "cpu")

    assert axis.dtype == np.float32
    assert np.allclose(axis, z_axis.numpy())


def build_fake_run(model_fn, n_gaussians, n_z, spatial):
    """Builds a fake inference run over a single unpadded patch.

    Args:
        model_fn    : Callable mapping an image batch to a predicted curve cube of shape (batch, n_z, spatial, spatial).
        n_gaussians : Number of Gaussian slots in the ground-truth parameter cube.
        n_z      : Number of z bins in the profile.
        spatial     : Side length in pixels of the single square patch.

    Returns:
        Namespace carrying the loader, model, dataset, grid, split region and z axis.
    """
    grid = GridInfo(
        n_v          = 1,
        n_h          = 1,
        pad_top      = 0,
        pad_bot      = 0,
        pad_left     = 0,
        pad_right    = 0,
        patch_size   = (spatial, spatial),
        stride       = (spatial, spatial),
        spatial_size = (spatial, spatial),
    )

    gt     = torch.rand(1, n_gaussians * 3, spatial, spatial)
    loader = [(torch.rand(1, 2, spatial, spatial), gt)]
    z_axis = np.linspace(-4.0, 4.0, n_z).astype(np.float32)

    class FakeNormalizer:
        """Normalizer stub that returns ground-truth parameters unchanged."""
        def denormalize_output(self, tensor):
            """Returns the parameter tensor unchanged."""
            return tensor

    dataset = types.SimpleNamespace(normalizer=FakeNormalizer())
    region  = types.SimpleNamespace(row_start=0, col_start=0)

    return types.SimpleNamespace(
        n_gaussians   = n_gaussians,
        z_axis        = z_axis,
        x_axis_length = n_z,
        loader        = loader,
        model         = model_fn,
        dataset       = dataset,
        grid          = grid,
        split_region  = region,
    )


def test_jepa_curve_predictor_produces_curve_cubes(tmp_path):
    """Inference yields predicted and ground-truth curve cubes plus a finite per-pixel MSE map."""
    n_gaussians = 2
    n_z         = 6
    spatial     = 4

    def model_fn(images):
        """Returns a random curve cube for the given image batch."""
        batch = images.shape[0]
        return np.random.rand(batch, n_z, spatial, spatial).astype(np.float32)

    run  = build_fake_run(model_fn, n_gaussians, n_z, spatial)
    meta = types.SimpleNamespace(cube_dir=tmp_path)

    predictor = JepaCurvePredictor(
        run,
        SilentLogger(),
        window_kind = "uniform",
        cube_dtype  = "float32",
        save_cubes  = False,
        meta        = meta,
    )

    result = predictor.run_inference()

    assert result.pred_curves.shape == (n_z, spatial, spatial)
    assert result.gt_curves.shape   == (n_z, spatial, spatial)
    assert result.pixel_mse.shape   == (spatial, spatial)
    assert np.isfinite(result.pixel_mse).all()


def test_jepa_curve_predictor_zero_error_on_identical_output(tmp_path):
    """Predicting exactly the ground-truth curves drives the per-pixel MSE to zero."""
    n_gaussians = 2
    n_z         = 6
    spatial     = 4

    x = np.linspace(-4.0, 4.0, n_z).astype(np.float32).reshape(1, 1, -1, 1, 1)

    captured = {}

    def model_fn(images):
        """Returns the curves reconstructed from the ground-truth Gaussian parameters."""
        gt_params = run["gt"][:, : n_gaussians * 3].cpu().numpy().astype(np.float32)
        batch     = gt_params.shape[0]
        gt_gauss  = gt_params.reshape(batch, n_gaussians, 3, spatial, spatial)
        curves    = GaussianReconstructor.reconstruct_batch(gt_gauss, x)
        captured["curves"] = curves
        return curves.astype(np.float32)

    gt     = torch.rand(1, n_gaussians * 3, spatial, spatial)
    loader = [(torch.rand(1, 2, spatial, spatial), gt)]
    run    = {"gt": gt}

    grid = GridInfo(
        n_v=1, n_h=1, pad_top=0, pad_bot=0, pad_left=0, pad_right=0,
        patch_size=(spatial, spatial), stride=(spatial, spatial), spatial_size=(spatial, spatial),
    )

    class FakeNormalizer:
        """Normalizer stub that returns ground-truth parameters unchanged."""
        def denormalize_output(self, tensor):
            """Returns the parameter tensor unchanged."""
            return tensor

    fake_run = types.SimpleNamespace(
        n_gaussians   = n_gaussians,
        z_axis        = np.linspace(-4.0, 4.0, n_z).astype(np.float32),
        x_axis_length = n_z,
        loader        = loader,
        model         = model_fn,
        dataset       = types.SimpleNamespace(normalizer=FakeNormalizer()),
        grid          = grid,
        split_region  = types.SimpleNamespace(row_start=0, col_start=0),
    )

    predictor = JepaCurvePredictor(
        fake_run,
        SilentLogger(),
        window_kind = "uniform",
        cube_dtype  = "float32",
        save_cubes  = False,
        meta        = types.SimpleNamespace(cube_dir=tmp_path),
    )

    result = predictor.run_inference()

    assert result.pixel_mse.max() == pytest.approx(0.0, abs=1e-8)


def test_jepa_curve_predictor_stitches_mu_sorted_gt_parameters(tmp_path):
    """The stitched ground-truth parameters are mu-sorted and no predicted parameters are produced."""
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4

    def model_fn(images):
        """Returns a random curve cube for the given image batch."""
        return np.random.rand(images.shape[0], n_elev, spatial, spatial).astype(np.float32)

    run  = build_fake_run(model_fn, n_gaussians, n_elev, spatial)
    meta = types.SimpleNamespace(cube_dir=tmp_path)

    predictor = JepaCurvePredictor(run, SilentLogger(), window_kind="uniform", cube_dtype="float32", save_cubes=False, meta=meta)
    result    = predictor.run_inference()

    assert result.params_gt.shape == (n_gaussians * 3, spatial, spatial)
    assert result.params_pred is None

    raw      = run.loader[0][1].numpy().reshape(1, n_gaussians, 3, spatial, spatial)
    expected = Predictor._sort_by_mu(raw).reshape(1, n_gaussians * 3, spatial, spatial)[0]

    np.testing.assert_allclose(result.params_gt[0], expected[0], rtol=1e-5)
    np.testing.assert_allclose(result.params_gt[3], expected[3], rtol=1e-5)


def test_jepa_curve_predictor_masks_inactive_gt_slots(tmp_path):
    """A zero-amplitude ground-truth slot has its mu and sigma channels written as NaN."""
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4

    def model_fn(images):
        """Returns an all-zero curve cube for the given image batch."""
        return np.zeros((images.shape[0], n_elev, spatial, spatial), dtype=np.float32)

    run = build_fake_run(model_fn, n_gaussians, n_elev, spatial)
    run.loader[0][1][:, 3] = 0.0

    predictor = JepaCurvePredictor(run, SilentLogger(), window_kind="uniform", cube_dtype="float32", save_cubes=False, meta=types.SimpleNamespace(cube_dir=tmp_path))
    result    = predictor.run_inference()

    assert np.all(np.isnan(result.params_gt[4]))
    assert np.all(np.isnan(result.params_gt[5]))
    assert not np.any(np.isnan(result.params_gt[1]))


def test_jepa_curve_predictor_masks_at_the_configured_threshold(tmp_path):
    """A slot whose amplitude sits below a raised active_amp_thr is masked, while the default keeps it."""
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4

    def model_fn(images):
        """Returns an all-zero curve cube for the given image batch."""
        return np.zeros((images.shape[0], n_elev, spatial, spatial), dtype=np.float32)

    def faint_run():
        """Returns a run whose second GT slot is faint (5e-4) and sorted after the strong first slot."""
        run = build_fake_run(model_fn, n_gaussians, n_elev, spatial)
        run.loader[0][1][:, 0] = 1.0
        run.loader[0][1][:, 1] = 0.1
        run.loader[0][1][:, 3] = 5e-4
        run.loader[0][1][:, 4] = 0.9
        return run

    kept   = JepaCurvePredictor(faint_run(), SilentLogger(), window_kind="uniform", cube_dtype="float32", save_cubes=False, meta=types.SimpleNamespace(cube_dir=tmp_path)).run_inference()
    masked = JepaCurvePredictor(faint_run(), SilentLogger(), window_kind="uniform", cube_dtype="float32", save_cubes=False, meta=types.SimpleNamespace(cube_dir=tmp_path), active_amp_thr=1e-3).run_inference()

    assert not np.any(np.isnan(kept.params_gt[4]))
    assert np.all(np.isnan(masked.params_gt[4]))
    assert np.all(np.isnan(masked.params_gt[5]))
    assert not np.any(np.isnan(masked.params_gt[1]))


def test_jepa_curve_predictor_stitches_each_batch_before_the_next_forward(tmp_path):
    """Each batch is stitched into both cubes before the next forward pass runs."""
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4
    events      = []

    def model_fn(images):
        """Records a forward event and returns a random curve cube."""
        events.append("forward")
        return np.random.rand(images.shape[0], n_elev, spatial, spatial).astype(np.float32)

    class RecordingPredictor(JepaCurvePredictor):
        """Curve predictor whose stitchers log every patch they receive into the shared event list."""
        def _create_stitcher(self, n_channels, name):
            """Returns the usual stitcher with its add_patch wrapped to record an event."""
            stitcher = super()._create_stitcher(n_channels, name)
            add      = stitcher.add_patch

            def record(idx, patch):
                """Logs the stitcher name and forwards the patch to the real add_patch."""
                events.append(f"add:{name}")
                add(idx, patch)

            stitcher.add_patch = record
            return stitcher

    run      = build_fake_run(model_fn, n_gaussians, n_elev, spatial)
    run.grid = GridInfo(
        n_v          = 1,
        n_h          = 2,
        pad_top      = 0,
        pad_bot      = 0,
        pad_left     = 0,
        pad_right    = 0,
        patch_size   = (spatial, spatial),
        stride       = (spatial, spatial),
        spatial_size = (spatial, 2 * spatial),
    )
    run.loader = [
        (torch.rand(1, 2, spatial, spatial), torch.rand(1, n_gaussians * 3, spatial, spatial)),
        (torch.rand(1, 2, spatial, spatial), torch.rand(1, n_gaussians * 3, spatial, spatial)),
    ]

    predictor = RecordingPredictor(run, SilentLogger(), window_kind="uniform", cube_dtype="float32", save_cubes=False, meta=types.SimpleNamespace(cube_dir=tmp_path))
    result    = predictor.run_inference()

    assert result.pred_curves.shape == (n_elev, spatial, 2 * spatial)
    assert events == ["forward", "add:pred_curves", "add:gt_curves", "forward", "add:pred_curves", "add:gt_curves"]


def test_jepa_curve_predictor_feeds_the_embedding_evaluator_in_the_same_pass(tmp_path):
    """The embedding evaluator is fed from the single forward pass, without a second loader iteration."""
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4

    class SpyEvaluator:
        """Embedding evaluator stand-in recording the shapes of every batch it is handed.

        Attributes:
            batches : One entry per accumulated batch, holding the two batch sizes and the ground-truth curve shape.
        """
        def __init__(self):
            """Starts with an empty batch record."""
            self.batches = []

        def accumulate_batch(self, z_hat_n, curve_n_hat, gt_curves):
            """Records the embedding and curve batch sizes and the ground-truth curve shape."""
            self.batches.append((z_hat_n.shape[0], curve_n_hat.shape[0], gt_curves.shape))

    class CountingLoader(list):
        """Batch list that counts how many times it has been iterated.

        Attributes:
            iterations : Number of times __iter__ has been called.
        """
        def __init__(self, batches):
            """Stores the batches and zeroes the iteration counter."""
            super().__init__(batches)
            self.iterations = 0

        def __iter__(self):
            """Counts the iteration and returns the underlying list iterator."""
            self.iterations += 1
            return super().__iter__()

    class SingleForwardModule:
        """Module stand-in returning curves and embeddings from one call, counting its forwards.

        Attributes:
            forwards : Number of times forward_with_embedding has been called.
        """
        def __init__(self):
            """Zeroes the forward counter."""
            self.forwards = 0

        def forward_with_embedding(self, images):
            """Returns random curves, a random embedding map and the same curves as the decoded output."""
            self.forwards += 1
            curves        = torch.rand(images.shape[0], n_elev, spatial, spatial)
            embedding     = torch.rand(images.shape[0], 4, spatial, spatial)

            return curves, embedding, curves

    class FakeWrapper:
        """Model wrapper whose call path fails, proving the embedding branch reuses the single forward.

        Attributes:
            module : The wrapped module exposing forward_with_embedding.
            device : Device string reported to the predictor.
        """
        def __init__(self, module):
            """Stores the wrapped module and reports the CPU device."""
            self.module = module
            self.device = "cpu"

        def __call__(self, images):
            """Fails, because a second wrapper call would mean a second forward pass."""
            raise AssertionError("the embedding path must reuse the single forward, not call the wrapper again")

    module     = SingleForwardModule()
    run        = build_fake_run(FakeWrapper(module), n_gaussians, n_elev, spatial)
    run.loader = CountingLoader([(torch.rand(1, 2, spatial, spatial), torch.rand(1, n_gaussians * 3, spatial, spatial))])
    spy        = SpyEvaluator()

    predictor = JepaCurvePredictor(
        run,
        SilentLogger(),
        window_kind         = "uniform",
        cube_dtype          = "float32",
        save_cubes          = False,
        meta                = types.SimpleNamespace(cube_dir=tmp_path),
        embedding_evaluator = spy,
    )
    predictor.run_inference()

    assert run.loader.iterations == 1
    assert module.forwards       == 1
    assert spy.batches           == [(1, 1, (1, n_elev, spatial, spatial))]
