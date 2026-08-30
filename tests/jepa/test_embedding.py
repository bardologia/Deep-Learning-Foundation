"""Tests for JepaEmbeddingEvaluator, which separates predictor error from decoder error.

Covers the embedding MSE, RMSE and cosine scalars, the decoder reconstruction floor, the chained
curve error, the predictor excess above the floor, stitching of the per-pixel error maps onto the
scene grid, correlation against an externally supplied curve-error map, and the guards on empty
loaders and mismatched map grids."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.loader import GridInfo
from pipelines.jepa.inference.embedding   import JepaEmbeddingEvaluator
from pipelines.jepa.inference.pipeline    import JEPA_INFERENCE_COMPONENTS
from tools.data.gaussians                 import GaussianReconstructor


from tests.conftest      import SilentLogger
from tests.jepa.conftest import PROFILE_LENGTH, make_autoencoder


class _FakeProfileAe:
    """Profile autoencoder stand-in whose encoder truncates the curve and whose decoder zero-pads it.

    Attributes:
        embedding_dim  : Number of leading curve samples kept as the embedding.
        profile_length : Length of the elevation axis the decoder pads back out to.
    """
    def __init__(self, embedding_dim: int):
        """Stores the embedding width used to truncate curves."""
        self.embedding_dim = embedding_dim

    def normalize_embedding(self, z):
        """Returns the embedding unchanged."""
        return z

    def encode(self, curve):
        """Returns the leading embedding_dim elevation samples of the curve."""
        return curve[:, : self.embedding_dim]

    def decode(self, z):
        """Zero-pads the embedding back to the full profile length along the channel axis."""
        pad = torch.zeros(z.shape[0], self.profile_length - z.shape[1], *z.shape[2:], dtype=z.dtype)
        return torch.cat([z, pad], dim=1)


class _FakeJepa:
    """JEPA stand-in that ignores its input and returns a fixed embedding map.

    Attributes:
        profile_autoencoder : The autoencoder whose encode and decode the evaluator calls.
        z_out               : Embedding map of shape (batch, embedding_dim, height, width) returned on every call.
    """
    def __init__(self, autoencoder, z_out):
        """Stores the paired autoencoder and the canned embedding output."""
        self.profile_autoencoder = autoencoder
        self.z_out               = z_out

    def parameters(self):
        """Yields a single dummy parameter so device probing succeeds."""
        return iter([torch.zeros(1)])

    def __call__(self, images):
        """Returns the canned embedding map regardless of the input images."""
        return self.z_out


def _grid(spatial: int, n_patches: int) -> GridInfo:
    """Returns a GridInfo of unpadded square patches laid out in one row of `n_patches`."""
    return GridInfo(
        n_v          = 1,
        n_h          = n_patches,
        pad_top      = 0,
        pad_bot      = 0,
        pad_left     = 0,
        pad_right    = 0,
        patch_size   = (spatial, spatial),
        stride       = (spatial, spatial),
        spatial_size = (spatial, spatial * n_patches),
    )


class _IdentityNormalizer:
    """Normalizer stub that leaves curves and parameters untouched."""
    def normalize(self, curve):
        """Returns the curve unchanged."""
        return curve

    def denormalize_output(self, params):
        """Returns the parameters unchanged."""
        return params


def _build_case(z_offset: float = 0.0, flip: bool = False):
    """Builds a fake JEPA run whose prediction is the encoded ground truth, optionally perturbed.

    Args:
        z_offset : Constant added to every embedding component before it is returned.
        flip     : When true the prediction is the negated target embedding instead.

    Returns:
        Tuple of the run namespace, the ground-truth curves of shape
        (batch, n_elev, height, width), and the fake profile autoencoder.
    """
    torch.manual_seed(0)

    n_gaussians, n_z, spatial, emb_dim = 2, 8, 3, 4
    z_axis = np.linspace(-4.0, 4.0, n_z).astype(np.float32)
    x      = z_axis.reshape(1, 1, -1, 1, 1)

    gt_params           = torch.rand(2, n_gaussians * 3, spatial, spatial) + 0.1
    gt_params[:, 1::3]  = torch.rand(2, n_gaussians, spatial, spatial) * 6.0 - 3.0
    gt_params[:, 2::3] += 0.5

    gt_np     = gt_params.numpy().astype(np.float32)
    gt_gauss  = gt_np.reshape(2, n_gaussians, 3, spatial, spatial)
    gt_curves = torch.from_numpy(GaussianReconstructor.reconstruct_batch(gt_gauss, x).astype(np.float32))

    autoencoder                = _FakeProfileAe(emb_dim)
    autoencoder.profile_length = n_z

    z_star = autoencoder.encode(gt_curves)
    z_out  = -z_star if flip else z_star + z_offset

    adapter = types.SimpleNamespace(
        jepa               = _FakeJepa(autoencoder, z_out),
        profile_normalizer = _IdentityNormalizer(),
    )

    run = types.SimpleNamespace(
        model       = types.SimpleNamespace(module=adapter, device="cpu"),
        loader      = [(torch.rand(2, 2, spatial, spatial), gt_params)],
        dataset     = types.SimpleNamespace(normalizer=_IdentityNormalizer()),
        n_gaussians = n_gaussians,
        z_axis      = z_axis,
        grid        = _grid(spatial, n_patches=2),
    )

    return run, gt_curves, autoencoder


def _build_layernorm_case():
    """Builds a fake run whose target embeddings come from a real LayerNorm-normalised autoencoder."""
    torch.manual_seed(0)

    n_gaussians, spatial = 2, 3
    n_z    = PROFILE_LENGTH
    z_axis = np.linspace(-4.0, 4.0, n_z).astype(np.float32)
    x      = z_axis.reshape(1, 1, -1, 1, 1)

    gt_params           = torch.rand(2, n_gaussians * 3, spatial, spatial) + 0.1
    gt_params[:, 1::3]  = torch.rand(2, n_gaussians, spatial, spatial) * 6.0 - 3.0
    gt_params[:, 2::3] += 0.5

    gt_gauss  = gt_params.numpy().astype(np.float32).reshape(2, n_gaussians, 3, spatial, spatial)
    gt_curves = torch.from_numpy(GaussianReconstructor.reconstruct_batch(gt_gauss, x).astype(np.float32))

    autoencoder = make_autoencoder("layernorm")
    autoencoder.eval()
    with torch.no_grad():
        autoencoder.embedding_layernorm.norm.weight.mul_(2.0).add_(0.3)
        autoencoder.embedding_layernorm.norm.bias.add_(0.5)
        z_out = autoencoder.encoder(gt_curves)

    adapter = types.SimpleNamespace(
        jepa               = _FakeJepa(autoencoder, z_out),
        profile_normalizer = _IdentityNormalizer(),
    )

    run = types.SimpleNamespace(
        model       = types.SimpleNamespace(module=adapter, device="cpu"),
        loader      = [(torch.rand(2, 2, spatial, spatial), gt_params)],
        dataset     = types.SimpleNamespace(normalizer=_IdentityNormalizer()),
        n_gaussians = n_gaussians,
        z_axis      = z_axis,
        grid        = _grid(spatial, n_patches=2),
    )

    return run


def _evaluate(run, pixel_mse=None):
    """Runs the evaluator over the fake run's loader and returns its metrics and maps.

    Args:
        run       : Fake run namespace carrying the model, loader, dataset, grid and elevation axis.
        pixel_mse : Optional per-pixel curve error map of shape (height, width) to correlate against.

    Returns:
        Tuple of the metric dict and the stitched embedding map container.
    """
    evaluator = JepaEmbeddingEvaluator(run, SilentLogger())
    n_K       = run.n_gaussians
    x         = np.asarray(run.x_axis, dtype=np.float32).reshape(1, 1, -1, 1, 1)

    jepa        = run.model.module.jepa
    autoencoder = jepa.profile_autoencoder

    for images, gt_params in run.loader:
        gt_phys    = run.dataset.normalizer.denormalize_output(gt_params[:, : n_K * 3]).cpu().numpy().astype(np.float32)
        B, _, H, W = gt_phys.shape
        gt_curves  = GaussianReconstructor.reconstruct_batch(gt_phys.reshape(B, n_K, 3, H, W), x)

        with torch.no_grad():
            z_hat_n     = autoencoder.normalize_embedding(jepa(images))
            curve_n_hat = autoencoder.decode(z_hat_n)

        evaluator.accumulate_batch(z_hat_n, curve_n_hat, gt_curves)

    return evaluator.run(pixel_mse=pixel_mse)


def test_components_carry_embedding_evaluator():
    """The JEPA curve component set wires in the embedding evaluator."""
    assert JEPA_INFERENCE_COMPONENTS.embedding_evaluator_cls is JepaEmbeddingEvaluator


def test_perfect_prediction_scores_zero_embedding_error():
    """An exact embedding prediction scores zero MSE, unit cosine and a chain error equal to the decoder floor."""
    run, gt_curves, autoencoder = _build_case()

    metrics, maps = _evaluate(run)

    assert metrics["jepa_embedding_mse"]    == pytest.approx(0.0, abs=1e-10)
    assert metrics["jepa_embedding_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["jepa_chain_mse_norm"]   == pytest.approx(metrics["jepa_decode_mse_norm"], abs=1e-10)


def test_decoder_floor_matches_reconstruction_tail():
    """The decoder floor equals the energy of the curve tail the truncating encoder discards."""
    run, gt_curves, autoencoder = _build_case()

    metrics, maps = _evaluate(run)

    tail     = gt_curves[:, autoencoder.embedding_dim:]
    expected = float((tail ** 2).sum()) / gt_curves.numel()

    assert metrics["jepa_decode_mse_norm"] == pytest.approx(expected, rel=1e-5)


def test_flipped_prediction_scores_negative_cosine():
    """A sign-flipped embedding scores cosine -1 and pushes the chain error above the decoder floor."""
    run, _, _ = _build_case(flip=True)

    metrics, maps = _evaluate(run)

    assert metrics["jepa_embedding_cosine"] == pytest.approx(-1.0, abs=1e-6)
    assert metrics["jepa_embedding_mse"] > 0.0
    assert metrics["jepa_chain_mse_norm"] > metrics["jepa_decode_mse_norm"]


def test_offset_prediction_separates_chain_from_decoder_error():
    """A constant embedding offset of 0.5 gives MSE 0.25, RMSE 0.5 and a chain error above the floor."""
    run, _, _ = _build_case(z_offset=0.5)

    metrics, maps = _evaluate(run)

    assert metrics["jepa_embedding_mse"] == pytest.approx(0.25, rel=1e-5)
    assert metrics["jepa_embedding_rmse"] == pytest.approx(0.5, rel=1e-5)
    assert metrics["jepa_chain_mse_norm"] > metrics["jepa_decode_mse_norm"]


def test_layernorm_target_is_normalized_exactly_once():
    """A LayerNorm-normalised target is not normalised a second time, so a matching prediction scores zero."""
    run = _build_layernorm_case()

    metrics, maps = _evaluate(run)

    assert metrics["jepa_embedding_mse"]    == pytest.approx(0.0, abs=1e-10)
    assert metrics["jepa_embedding_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_empty_loader_raises():
    """A loader yielding no batches raises rather than reporting empty metrics."""
    run, _, _ = _build_case()
    run.loader = []

    with pytest.raises(ValueError, match="no samples"):
        _evaluate(run)


def test_embedding_maps_are_stitched_to_the_scene_grid():
    """Every error map is stitched to the scene size, finite, and averages the injected offset error."""
    run, _, _ = _build_case(z_offset=0.5)

    _metrics, maps = _evaluate(run)

    assert maps.embedding_error.shape  == run.grid.spatial_size
    assert maps.embedding_cosine.shape == run.grid.spatial_size
    assert maps.decode_mse.shape       == run.grid.spatial_size
    assert maps.chain_mse.shape        == run.grid.spatial_size

    assert np.isfinite(maps.embedding_error).all()
    assert maps.embedding_error.mean() == pytest.approx(0.25, rel=1e-4)


def test_predictor_excess_isolates_the_decoder_floor():
    """Predictor excess equals chain error minus decoder floor, with the floor a proper fraction of the total."""
    run, _, _ = _build_case(z_offset=0.5)

    metrics, _maps = _evaluate(run)

    assert metrics["jepa_predictor_excess_mse"] == pytest.approx(metrics["jepa_chain_mse_norm"] - metrics["jepa_decode_mse_norm"], rel=1e-9)
    assert 0.0 < metrics["jepa_decoder_floor_frac"] < 1.0
    assert metrics["jepa_decode_amplification"] > 0.0


def test_perfect_prediction_leaves_no_predictor_excess():
    """An exact prediction leaves zero predictor excess and attributes the whole error to the decoder floor."""
    run, _, _ = _build_case()

    metrics, _maps = _evaluate(run)

    assert metrics["jepa_predictor_excess_mse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["jepa_decoder_floor_frac"]   == pytest.approx(1.0, rel=1e-6)


def test_embedding_error_correlates_with_a_supplied_curve_error():
    """Supplying a curve-error map yields a finite embedding-error correlation scalar."""
    run, _, _ = _build_case(z_offset=0.5)

    pixel_mse      = np.linspace(0.0, 1.0, int(np.prod(run.grid.spatial_size)), dtype=np.float32).reshape(run.grid.spatial_size)
    metrics, _maps = _evaluate(run, pixel_mse=pixel_mse)

    assert "jepa_embedding_error_corr" in metrics
    assert np.isfinite(metrics["jepa_embedding_error_corr"])


def test_mismatched_curve_error_map_raises():
    """A curve-error map stitched on a different grid is rejected."""
    run, _, _ = _build_case()

    with pytest.raises(ValueError, match="stitched on different grids"):
        _evaluate(run, pixel_mse=np.zeros((2, 2), dtype=np.float32))
