"""Tests covering patch windows, cube stitching and the inference predictor loop.

The predictor cases build a tiny UNet run over a synthetic patch grid and check the
stitched curve and parameter cubes, the CPU render worker and the result-window budget.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from models.backbone                                 import get_backbone
from pipelines.backbone.dataset.spatial              import Patcher
from pipelines.backbone.inference.model_wrapper      import ModelWrapper
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Result
from pipelines.backbone.inference.predictor          import CubeStitcher, Predictor, SelectStitcher
from tools.data.regions                              import CropRegion
from configuration.inference                         import InferenceConfig

from tests.conftest import RecordingLogger, SilentLogger


N_GAUSSIANS = 5
OUT_CH      = N_GAUSSIANS * 3
N_Z         = 16
PATCH       = 16


class _IdentityNormalizer:
    """Normalizer stub that leaves denormalized outputs untouched."""
    @staticmethod
    def denormalize_output(t):
        """Returns the tensor unchanged."""
        return t


def _build_run(n_rows: int, n_cols: int, in_channels: int = 4):
    """Builds a synthetic inference run over a single-batch patch grid.

    Args:
        n_rows        : Row size of the crop region, in pixels.
        n_cols        : Range size of the crop region, in pixels.
        in_channels : Number of input channels fed to the backbone.

    Returns:
        Namespace holding the wrapped model, the patch grid, a one-batch loader of
        images of shape (n_patches, in_channels, PATCH, PATCH) paired with
        ground-truth parameter maps of shape (n_patches, OUT_CH, PATCH, PATCH), and
        the crop region the patches cover.
    """
    region = CropRegion(row_start=0, row_end=n_rows, col_start=0, col_end=n_cols)
    patch  = Patcher.build(spatial_size=(n_rows, n_cols), patch_size=(PATCH, PATCH), stride=(PATCH, PATCH), use_symmetric_padding=True)
    grid   = patch.grid

    torch.manual_seed(0)
    model, _ = get_backbone("unet", in_channels=in_channels, out_channels=OUT_CH, features=[8, 16], bottleneck_factor=1)
    model.eval()

    z_axis  = np.linspace(-20.0, 80.0, N_Z).astype(np.float32)
    wrapper = ModelWrapper(model, "cpu", z_axis=torch.from_numpy(z_axis), amp_max=80.0, normalizer=None)

    n_patches = grid.number_of_patches
    rng       = np.random.default_rng(0)

    batch_imgs = torch.from_numpy(rng.standard_normal((n_patches, in_channels, PATCH, PATCH)).astype(np.float32))
    gt_params  = torch.zeros((n_patches, OUT_CH, PATCH, PATCH), dtype=torch.float32)
    gt_params[:, 0] = 1.0
    gt_params[:, 1] = 20.0
    gt_params[:, 2] = 5.0

    loader = [(batch_imgs, gt_params)]

    dataset = types.SimpleNamespace(normalizer=_IdentityNormalizer())

    run = types.SimpleNamespace(
        model              = wrapper,
        n_gaussians        = N_GAUSSIANS,
        activity_threshold = 1e-4,
        out_channels       = OUT_CH,
        z_axis             = z_axis,
        x_axis_length      = N_Z,
        grid               = grid,
        loader             = loader,
        dataset            = dataset,
        split_region       = region,
        full_curves        = None,
    )
    return run


def _make_predictor(tmp_path, run) -> Predictor:
    """Returns a predictor writing into a fresh inference metadata tree under tmp_path."""
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="pred", device="cpu")
    meta = InferenceMetadata(cfg)
    meta.create_dirs()

    return Predictor(
        run         = run,
        logger      = SilentLogger(),
        window_kind = "hann",
        cube_dtype  = "float32",
        save_cubes  = False,
        meta        = meta,
        cpu_workers = 1,
    )


def test_make_patch_window_uniform():
    """Uniform windows are all-ones over the patch shape."""
    w = CubeStitcher.make_patch_window((4, 4), kind="uniform")
    assert w.shape == (4, 4)
    assert np.allclose(w, 1.0)


def test_make_patch_window_hann_is_positive_and_bell():
    """Hann windows stay strictly positive and peak at the patch centre."""
    w = CubeStitcher.make_patch_window((8, 8), kind="hann")
    assert np.all(w >= 1e-3)
    assert w[4, 4] > w[0, 0]


def test_make_patch_window_unknown_raises():
    """An unknown window kind is rejected."""
    with pytest.raises(ValueError, match="Unknown window"):
        CubeStitcher.make_patch_window((4, 4), kind="bogus")


def test_cube_stitcher_single_patch_reconstructs():
    """A single uniform-weighted patch stitches back to the original cube."""
    patch = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=(8, 8), use_symmetric_padding=False)
    grid  = patch.grid

    stitcher = CubeStitcher(grid, n_channels=3, window_kind="uniform")
    data     = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    stitcher.add_patch(0, data)

    cube = stitcher.finalize_cube()

    assert cube.shape == (3, 8, 8)
    assert np.allclose(cube, data)


def test_select_stitcher_takes_nearest_patch_centre():
    """Overlapping patches are resolved by the nearest patch centre."""
    patcher = Patcher.build(spatial_size=(8, 12), patch_size=(8, 8), stride=(4, 4), use_symmetric_padding=False)
    grid    = patcher.grid

    stitcher = SelectStitcher(grid, n_channels=1)
    stitcher.add_patch(0, np.full((1, 8, 8), 1.0, dtype=np.float32))
    stitcher.add_patch(1, np.full((1, 8, 8), 2.0, dtype=np.float32))

    cube = stitcher.finalize_cube()

    assert cube.shape == (1, 8, 12)
    assert np.allclose(cube[0, :, :4], 1.0)
    assert np.allclose(cube[0, :, 8:], 2.0)


def test_select_stitcher_single_patch_is_exact():
    """Nearest-centre selection is exact for a lone covering patch."""
    patcher = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=(8, 8), use_symmetric_padding=False)
    grid    = patcher.grid

    stitcher = SelectStitcher(grid, n_channels=3)
    data     = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    stitcher.add_patch(0, data)

    cube = stitcher.finalize_cube()

    assert np.allclose(cube, data)


def test_cube_stitcher_rectangular_patches_reconstruct():
    """Hann-weighted overlapping rectangular patches reconstruct the source cube."""
    patcher = Patcher.build(spatial_size=(8, 24), patch_size=(8, 12), stride=(8, 6), use_symmetric_padding=False)
    grid    = patcher.grid

    data     = np.arange(2 * 8 * 24, dtype=np.float32).reshape(2, 8, 24)
    stitcher = CubeStitcher(grid, n_channels=2, window_kind="hann")

    for idx in range(grid.number_of_patches):
        stitcher.add_patch(idx, patcher.extract(data, idx))

    cube = stitcher.finalize_cube()

    assert cube.shape == (2, 8, 24)
    assert np.allclose(cube, data, atol=1e-4)


def test_select_stitcher_rectangular_patches_reconstruct():
    """Nearest-centre selection reconstructs the source cube from rectangular patches."""
    patcher = Patcher.build(spatial_size=(12, 8), patch_size=(6, 8), stride=(3, 8), use_symmetric_padding=False)
    grid    = patcher.grid

    data     = np.arange(1 * 12 * 8, dtype=np.float32).reshape(1, 12, 8)
    stitcher = SelectStitcher(grid, n_channels=1)

    for idx in range(grid.number_of_patches):
        stitcher.add_patch(idx, patcher.extract(data, idx))

    cube = stitcher.finalize_cube()

    assert cube.shape == (1, 12, 8)
    assert np.allclose(cube, data)


def test_select_stitcher_raises_on_uncovered_pixels():
    """Finalizing a cube with pixels no patch covered raises."""
    patcher = Patcher.build(spatial_size=(8, 12), patch_size=(8, 8), stride=(4, 4), use_symmetric_padding=False)
    grid    = patcher.grid

    stitcher = SelectStitcher(grid, n_channels=1)
    stitcher.add_patch(0, np.ones((1, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="uncovered"):
        stitcher.finalize_cube()


def test_cpu_worker_render_floor_drops_weak_components():
    """Components below the amplitude floor vanish from the rendered curves while staying in the parameter cube."""
    z_axis = np.linspace(0.0, 40.0, 32).astype(np.float32)

    pred       = np.zeros((1, 6, 4, 4), dtype=np.float32)
    pred[:, 0] = 1.0
    pred[:, 1] = 5.0
    pred[:, 2] = 2.0
    pred[:, 3] = 5e-4
    pred[:, 4] = 20.0
    pred[:, 5] = 3.0

    gt = pred.copy()

    unmasked = Predictor._cpu_worker((pred, gt, z_axis, 2, 0.0,  1e-4))
    masked   = Predictor._cpu_worker((pred, gt, z_axis, 2, 1e-3, 1e-4))

    idx20 = int(np.argmin(np.abs(z_axis - 20.0)))

    assert unmasked[0][0, idx20, 0, 0] > 1e-4
    assert masked[0][0, idx20, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert masked[1][0, idx20, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert masked[2][0, 3, 0, 0] == pytest.approx(5e-4)


def test_result_window_is_bounded_by_the_byte_budget(tmp_path):
    """The result window follows the byte budget, clamped to at least one task and at most the patch count."""
    run       = _build_run(n_rows=16, n_cols=16)
    predictor = _make_predictor(tmp_path, run)
    predictor.cpu_workers = 8

    params = [np.zeros((4, OUT_CH, PATCH, PATCH), dtype=np.float32)]

    channels       = 2 * N_Z + 2 * (3 * N_GAUSSIANS)
    bytes_per_task = 4 * channels * PATCH * PATCH * 4

    predictor.RESULT_BUDGET_BYTES = 3 * bytes_per_task
    assert predictor._result_window(params) == 3

    predictor.RESULT_BUDGET_BYTES = bytes_per_task // 4
    assert predictor._result_window(params) == 1

    predictor.RESULT_BUDGET_BYTES = 1000 * bytes_per_task
    assert predictor._result_window(params) == 16


def test_predictor_run_inference_shapes(tmp_path):
    """Inference returns finite cubes shaped to the crop region and the z axis."""
    run       = _build_run(n_rows=16, n_cols=16)
    predictor = _make_predictor(tmp_path, run)

    result = predictor.run_inference()

    assert isinstance(result, Result)
    assert result.pred_curves.shape == (N_Z, 16, 16)
    assert result.gt_curves.shape   == (N_Z, 16, 16)
    assert result.params_pred.shape == (OUT_CH, 16, 16)
    assert result.params_gt.shape   == (N_GAUSSIANS * 3, 16, 16)
    assert result.pixel_mse.shape   == (16, 16)
    assert np.isfinite(result.pixel_mse).all()


def test_predictor_writes_pixel_maps_without_save_cubes(tmp_path):
    """The per-pixel error maps land on disk even when the heavy cubes are not saved."""
    run       = _build_run(n_rows=16, n_cols=16)
    predictor = _make_predictor(tmp_path, run)

    predictor.run_inference()

    for name in ("pixel_mse", "pixel_mae", "pixel_r2", "pixel_cos", "pixel_peak"):
        saved = np.load(predictor.cube_dir / f"{name}.npy")
        assert saved.shape == (16, 16)

    assert not (predictor.cube_dir / "pred_curves.npy").exists()


def test_predictor_gt_curves_match_gt_gaussian(tmp_path):
    """Ground-truth curves render the analytic Gaussian of the ground-truth parameters."""
    run       = _build_run(n_rows=16, n_cols=16)
    predictor = _make_predictor(tmp_path, run)

    result = predictor.run_inference()

    x        = run.z_axis
    expected = 1.0 * np.exp(-((x - 20.0) ** 2) / (2.0 * 5.0 * 5.0 + 1e-8))
    centre   = result.gt_curves[:, 8, 8]

    assert np.allclose(centre, expected.astype(np.float32), atol=1e-3)


def test_predictor_warns_when_a_gt_pixel_has_a_constant_profile(tmp_path):
    """A pixel whose ground-truth slots are all inactive has an undefined R2, which is counted and warned about instead of averaged."""
    run = _build_run(n_rows=16, n_cols=16)
    run.loader[0][1][:, 0, 0, 0] = 0.0

    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="pred", device="cpu")
    meta = InferenceMetadata(cfg)
    meta.create_dirs()

    logger    = RecordingLogger()
    predictor = Predictor(run=run, logger=logger, window_kind="hann", cube_dtype="float32", save_cubes=False, meta=meta, cpu_workers=1)

    result = predictor.run_inference()

    assert np.isnan(result.pixel_r2[0, 0])
    assert np.isfinite(np.delete(result.pixel_r2.reshape(-1), 0)).all()
    assert any(level == "warning" and "1 of 256 pixels have a constant GT profile" in message for level, message in logger.messages)


def test_predictor_offsets_from_region(tmp_path):
    """Result offsets are taken from the split-region start indices."""
    region = CropRegion(row_start=100, row_end=116, col_start=50, col_end=66)
    run    = _build_run(n_rows=16, n_cols=16)
    run.split_region = region

    predictor = _make_predictor(tmp_path, run)
    result    = predictor.run_inference()

    assert result.row_offset == 100
    assert result.col_offset   == 50


def test_backend_row_names_the_model_device_not_cuda_availability(tmp_path, monkeypatch):
    """A model on the CPU is logged as a CPU backend even when CUDA is available, with no GPU rows."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    rows = _make_predictor(tmp_path, _build_run(PATCH, PATCH))._device_rows()

    assert rows == {"Backend": "PyTorch / cpu"}


def test_backend_rows_name_the_physical_gpu_on_cuda(tmp_path, monkeypatch):
    """A model on a CUDA device is logged with the GPU name, the pinned CUDA_VISIBLE_DEVICES token and the inherited mask."""
    run       = _build_run(PATCH, PATCH)
    run.model = types.SimpleNamespace(device="cuda:0")

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("MLTEMPLATE_INHERITED_CUDA_MASK", "2,3")

    rows = _make_predictor(tmp_path, run)._device_rows()

    assert rows == {"Backend": "PyTorch / cuda:0", "GPU": "Test GPU", "CUDA_VISIBLE_DEVICES": "3", "Inherited CUDA mask": "2,3"}
