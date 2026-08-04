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

from tests.conftest import SilentLogger


N_GAUSSIANS = 5
OUT_CH      = N_GAUSSIANS * 3
N_ELEV      = 16
PATCH       = 16


class _IdentityNormalizer:
    @staticmethod
    def denormalize_output(t):
        return t


def _build_run(n_az: int, n_rg: int, in_channels: int = 4):
    region = CropRegion(azimuth_start=0, azimuth_end=n_az, range_start=0, range_end=n_rg)
    patch  = Patcher.build(spatial_size=(n_az, n_rg), patch_size=(PATCH, PATCH), stride=(PATCH, PATCH), use_symmetric_padding=True)
    grid   = patch.grid

    torch.manual_seed(0)
    model, _ = get_backbone("unet", in_channels=in_channels, out_channels=OUT_CH, features=[8, 16], bottleneck_factor=1)
    model.eval()

    x_axis  = np.linspace(-20.0, 80.0, N_ELEV).astype(np.float32)
    wrapper = ModelWrapper(model, "cpu", x_axis=torch.from_numpy(x_axis), amp_max=80.0, normalizer=None)

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
        model         = wrapper,
        n_gaussians   = N_GAUSSIANS,
        out_channels  = OUT_CH,
        x_axis        = x_axis,
        x_axis_length = N_ELEV,
        grid          = grid,
        loader        = loader,
        dataset       = dataset,
        split_region  = region,
        full_curves   = None,
    )
    return run


def _make_predictor(tmp_path, run) -> Predictor:
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
    w = CubeStitcher.make_patch_window((4, 4), kind="uniform")
    assert w.shape == (4, 4)
    assert np.allclose(w, 1.0)


def test_make_patch_window_hann_is_positive_and_bell():
    w = CubeStitcher.make_patch_window((8, 8), kind="hann")
    assert np.all(w >= 1e-3)
    assert w[4, 4] > w[0, 0]


def test_make_patch_window_unknown_raises():
    with pytest.raises(ValueError, match="Unknown window"):
        CubeStitcher.make_patch_window((4, 4), kind="bogus")


def test_cube_stitcher_single_patch_reconstructs():
    patch = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=(8, 8), use_symmetric_padding=False)
    grid  = patch.grid

    stitcher = CubeStitcher(grid, n_channels=3, window_kind="uniform")
    data     = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    stitcher.add_patch(0, data)

    cube = stitcher.finalize_cube()

    assert cube.shape == (3, 8, 8)
    assert np.allclose(cube, data)


def test_select_stitcher_takes_nearest_patch_centre():
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
    patcher = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=(8, 8), use_symmetric_padding=False)
    grid    = patcher.grid

    stitcher = SelectStitcher(grid, n_channels=3)
    data     = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    stitcher.add_patch(0, data)

    cube = stitcher.finalize_cube()

    assert np.allclose(cube, data)


def test_cube_stitcher_rectangular_patches_reconstruct():
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
    patcher = Patcher.build(spatial_size=(8, 12), patch_size=(8, 8), stride=(4, 4), use_symmetric_padding=False)
    grid    = patcher.grid

    stitcher = SelectStitcher(grid, n_channels=1)
    stitcher.add_patch(0, np.ones((1, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="uncovered"):
        stitcher.finalize_cube()


def test_cpu_worker_render_floor_drops_weak_components():
    x_axis = np.linspace(0.0, 40.0, 32).astype(np.float32)

    pred       = np.zeros((1, 6, 4, 4), dtype=np.float32)
    pred[:, 0] = 1.0
    pred[:, 1] = 5.0
    pred[:, 2] = 2.0
    pred[:, 3] = 5e-4
    pred[:, 4] = 20.0
    pred[:, 5] = 3.0

    gt = pred.copy()

    unmasked = Predictor._cpu_worker((pred, gt, x_axis, 2, 0.0))
    masked   = Predictor._cpu_worker((pred, gt, x_axis, 2, 1e-3))

    idx20 = int(np.argmin(np.abs(x_axis - 20.0)))

    assert unmasked[0][0, idx20, 0, 0] > 1e-4
    assert masked[0][0, idx20, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert masked[1][0, idx20, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert masked[2][0, 3, 0, 0] == pytest.approx(5e-4)


def test_predictor_run_inference_shapes(tmp_path):
    run       = _build_run(n_az=16, n_rg=16)
    predictor = _make_predictor(tmp_path, run)

    result = predictor.run_inference()

    assert isinstance(result, Result)
    assert result.pred_curves.shape == (N_ELEV, 16, 16)
    assert result.gt_curves.shape   == (N_ELEV, 16, 16)
    assert result.params_pred.shape == (OUT_CH, 16, 16)
    assert result.params_gt.shape   == (N_GAUSSIANS * 3, 16, 16)
    assert result.pixel_mse.shape   == (16, 16)
    assert np.isfinite(result.pixel_mse).all()


def test_predictor_gt_curves_match_gt_gaussian(tmp_path):
    run       = _build_run(n_az=16, n_rg=16)
    predictor = _make_predictor(tmp_path, run)

    result = predictor.run_inference()

    x        = run.x_axis
    expected = 1.0 * np.exp(-((x - 20.0) ** 2) / (2.0 * 5.0 * 5.0 + 1e-8))
    centre   = result.gt_curves[:, 8, 8]

    assert np.allclose(centre, expected.astype(np.float32), atol=1e-3)


def test_predictor_offsets_from_region(tmp_path):
    region = CropRegion(azimuth_start=100, azimuth_end=116, range_start=50, range_end=66)
    run    = _build_run(n_az=16, n_rg=16)
    run.split_region = region

    predictor = _make_predictor(tmp_path, run)
    result    = predictor.run_inference()

    assert result.azimuth_offset == 100
    assert result.range_offset   == 50
