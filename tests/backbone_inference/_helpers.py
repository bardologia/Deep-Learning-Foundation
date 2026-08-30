"""Shared fixtures that assemble a miniature backbone run directory for inference tests.

The helpers persist the dataset configuration, channel statistics, model
configuration and a small U-Net checkpoint so that the loader and the end-to-end
inference pipeline can be exercised against the synthetic test scene.
"""

from __future__ import annotations

import numpy as np
import torch

from configuration.dataset                      import DatasetConfig, InputConfig, OutputConfig, PatchConfig, SplitRegions
from configuration.normalization                import ChannelStats, OutputClampConfig
from configuration.normalization.general        import ChannelStrategy, NormMethod
from configuration.synthesis                    import SynthesisConfig
from models                                     import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.backbone.dataset.metadata_writer import MetadataWriter
from pipelines.backbone.dataset.stats           import Stats
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from tools.data.io                              import FileIO
from tools.data.regions                         import CropRegion

from tests.conftest import SilentLogger


N_GAUSSIANS      = SynthesisConfig().k_max
N_Z              = SynthesisConfig().profile_length
SECONDARY_LABELS = ("CH01", "CH02", "CH03", "CH04")


def dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    """Builds a 96x96-per-split dataset configuration over the synthetic test scene.

    Args:
        test_data_dir : Preprocessing run directory holding the channel stack.
        params_dir    : Directory containing ``parameters.npy`` with the Gaussian labels.

    Returns:
        A dataset configuration with disjoint train/val/test crop regions, 64x64
        patches at stride 32 and the four secondary channels of ``SECONDARY_LABELS``.
    """
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
    """Writes identity z-score channel statistics for the given input width into the meta directory."""
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
    """Saves a two-level U-Net configuration and its ``best_model.pt`` checkpoint with the scene's z axis."""
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


def build_run_directory(tmp_path, test_data_dir, params_dir):
    """Assembles a complete backbone run directory that inference can load.

    Args:
        tmp_path      : Directory the run is created under.
        test_data_dir : Preprocessing run directory holding the channel stack.
        params_dir    : Directory containing ``parameters.npy`` with the Gaussian labels.

    Returns:
        Path of the run directory, populated with the dataset metadata, the run
        summary, the trainer configuration, the channel statistics and a trained
        U-Net checkpoint.
    """
    run_dir  = tmp_path / "run_unet_e2e"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    config      = dataset_config(test_data_dir, params_dir)
    in_channels = config.input_config.total_channels(0, len(SECONDARY_LABELS))

    MetadataWriter(run_dir, logger=SilentLogger()).save_dataset_configuration(config)
    FileIO.save_json({"model_name": "unet", "in_channels": in_channels, "out_channels": 3 * N_GAUSSIANS, "n_gaussians": N_GAUSSIANS}, meta_dir / "run_summary.json")
    FileIO.save_json({"geometry": {"z_axis_convention": "height"}}, run_dir / "docs" / "trainer_config.json")

    _persist_stats(meta_dir, in_channels)
    _persist_model(run_dir, meta_dir, in_channels)

    return run_dir
