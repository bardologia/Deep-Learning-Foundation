from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configuration.dataset                      import DatasetConfig, InputConfig, OutputConfig, PatchConfig, SplitRegions
from configuration.inference                    import InferenceConfig
from configuration.normalization                import ChannelStats, OutputClampConfig
from configuration.normalization.general        import ChannelStrategy, NormMethod
from models                                     import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.backbone.dataset.metadata_writer import MetadataWriter
from pipelines.backbone.dataset.stats           import Stats
from pipelines.backbone.inference.pipeline      import InferencePipeline
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from tools.data.io                              import FileIO
from tools.data.regions                         import CropRegion

from configuration.synthesis import SynthesisConfig

from tests.conftest import SilentLogger


N_GAUSSIANS      = SynthesisConfig().k_max
N_Z              = SynthesisConfig().profile_length
SECONDARY_LABELS = ("CH01", "CH02", "CH03", "CH04")


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    return DatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = SplitRegions(
            train = CropRegion(row_start=1000, row_end=1096, col_start=500, col_end=596),
            val   = CropRegion(row_start=1100, row_end=1196, col_start=500, col_end=596),
            test  = CropRegion(row_start=1200, row_end=1296, col_start=500, col_end=596),
        ),
        secondary_labels = SECONDARY_LABELS,
        patch            = PatchConfig(size=(64, 64), stride=(32, 32), use_symmetric_padding=True),
        input_config     = InputConfig(),
        output_config    = OutputConfig(),
        batch_size       = 4,
        num_workers      = 0,
    )


def _persist_stats(meta_dir, in_channels: int) -> None:
    zscore = ChannelStrategy(NormMethod.ZSCORE)

    stats = Stats(
        input_stats  = ChannelStats(
            loc        = [0.0] * in_channels,
            scale      = [1.0] * in_channels,
            names      = [f"in/{i}" for i in range(in_channels)],
            strategies = [zscore] * in_channels,
            clampable  = [False] * in_channels,
        ),
        output_stats = ChannelStats(
            loc        = [0.0] * (3 * N_GAUSSIANS),
            scale      = [1.0] * (3 * N_GAUSSIANS),
            names      = [f"out/{i}" for i in range(3 * N_GAUSSIANS)],
            strategies = [zscore] * (3 * N_GAUSSIANS),
            clampable  = [True, False, True] * N_GAUSSIANS,
        ),
        clamp        = OutputClampConfig(),
    )
    stats.save(meta_dir)


def _persist_model(run_dir, meta_dir, in_channels: int) -> None:
    config                   = BACKBONE_CONFIG_REGISTRY["unet"]()
    config.in_channels       = in_channels
    config.out_channels      = 3 * N_GAUSSIANS
    config.features          = [8, 16]
    config.bottleneck_factor = 1

    BackboneModelConfigIO.save(config, "unet", meta_dir)

    torch.manual_seed(0)
    model, _ = get_backbone("unet", config=config, in_channels=in_channels, out_channels=3 * N_GAUSSIANS)

    z_axis = np.linspace(-20.0, 80.0, N_Z).astype(np.float32)
    torch.save(
        {"params": model.state_dict(), "z_axis": z_axis, "epoch": 1, "best_val_loss": 0.1, "best_epoch": 1},
        run_dir / "best_model.pt",
    )


def _build_run_directory(tmp_path, test_data_dir, params_dir):
    run_dir  = tmp_path / "run_unet_e2e"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    dataset_config = _dataset_config(test_data_dir, params_dir)
    in_channels    = dataset_config.input_config.total_channels(0, len(SECONDARY_LABELS))

    MetadataWriter(run_dir, logger=SilentLogger()).save_dataset_configuration(dataset_config)
    FileIO.save_json({"model_name": "unet", "in_channels": in_channels, "out_channels": 3 * N_GAUSSIANS, "n_gaussians": N_GAUSSIANS}, meta_dir / "run_summary.json")
    FileIO.save_json({"geometry": {"z_axis_convention": "height"}}, run_dir / "docs" / "trainer_config.json")

    _persist_stats(meta_dir, in_channels)
    _persist_model(run_dir, meta_dir, in_channels)

    return run_dir


@pytest.mark.dataset
@pytest.mark.slow
def test_inference_pipeline_end_to_end(tmp_path, test_data_dir, params_dir):
    run_dir = _build_run_directory(tmp_path, test_data_dir, params_dir)

    config = InferenceConfig(
        run_directory            = run_dir,
        output_subdir            = "e2e",
        device                   = "cpu",
        split                    = "test",
        num_workers              = 0,
        cpu_workers              = 2,
        save_plots               = False,
        save_animations          = False,
        save_cubes               = True,
    )

    report_path = InferencePipeline(config).run()

    output_dir = run_dir / "inference" / "e2e"
    assert report_path == output_dir / "report.md"
    assert report_path.is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert metrics["split"] == "test"

    for key in ("curve_mse_gt", "overall_r2_gt", "pixel_mse_gt_mean", "ssim_norm_z_mean", "active_frac_gt", "matched_recall"):
        assert np.isfinite(metrics[key]), key

    cubes = output_dir / "cubes"
    assert np.load(cubes / "pred_curves.npy").shape == (N_Z, 96, 96)
    assert np.load(cubes / "gt_curves.npy").shape   == (N_Z, 96, 96)
    assert np.load(cubes / "params_pred.npy").shape == (3 * N_GAUSSIANS, 96, 96)
    assert np.load(cubes / "params_gt.npy").shape   == (3 * N_GAUSSIANS, 96, 96)

    report = report_path.read_text()
    assert "unet" in report


def test_equal_indices_are_unique_and_inside_the_axis():
    for n_total in (7, 10, 64, 150):
        for n_slices in (1, 3, 5, n_total):
            indices = InferencePipeline._equal_indices(n_total, n_slices)

            assert indices.size == np.unique(indices).size
            assert indices.min() >= 0
            assert indices.max() < n_total
