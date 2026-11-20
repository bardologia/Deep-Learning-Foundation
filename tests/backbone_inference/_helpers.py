from __future__ import annotations

import contextlib

import torch

from configuration.dataset                      import DatasetConfig, PatchConfig
from configuration.normalization                import ChannelStats, NormMethod
from models                                     import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.backbone.dataset.metadata_writer import MetadataWriter
from pipelines.backbone.dataset.stats           import Stats
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.dataset.dataset_files     import DatasetFiles
from tools.data.io                              import FileIO

from tests.conftest import RecordingLogger, SilentLogger


def dataset_config(test_data_dir) -> DatasetConfig:
    return DatasetConfig(
        dataset_directory = test_data_dir,
        split_regions     = DatasetFiles.read(test_data_dir).splits,
        patch             = PatchConfig(size=(32, 32), stride=(16, 16), use_symmetric_padding=True),
        batch_size        = 4,
        num_workers       = 0,
    )


def _persist_stats(meta_dir, files: DatasetFiles) -> None:
    def identity(names):
        return ChannelStats(names=list(names), method=NormMethod.ZSCORE, loc=[0.0] * len(names), scale=[1.0] * len(names))

    Stats(input_stats=identity(files.input_channels), target_stats=identity(files.target_channels)).save(meta_dir)


def _persist_model(run_dir, meta_dir, in_channels: int, out_channels: int) -> None:
    config                   = BACKBONE_CONFIG_REGISTRY["unet"]()
    config.in_channels       = in_channels
    config.out_channels      = out_channels
    config.features          = [8, 16]
    config.bottleneck_factor = 1

    BackboneModelConfigIO.save(config, "unet", meta_dir)

    torch.manual_seed(0)
    model, _ = get_backbone("unet", config=config, in_channels=in_channels, out_channels=out_channels)

    torch.save({"params": model.state_dict(), "epoch": 1, "best_val_loss": 0.1, "best_epoch": 1}, run_dir / "best_model.pt")


def build_run_directory(tmp_path, test_data_dir):
    run_dir  = tmp_path / "run_unet_e2e"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    files = DatasetFiles.read(test_data_dir)

    MetadataWriter(run_dir, logger=SilentLogger()).save_dataset_configuration(dataset_config(test_data_dir))
    FileIO.save_json({"model_name": "unet", "in_channels": len(files.input_channels), "out_channels": len(files.target_channels), "input_channels": files.input_channels, "target_channels": files.target_channels}, meta_dir / "run_summary.json")

    _persist_stats(meta_dir, files)
    _persist_model(run_dir, meta_dir, len(files.input_channels), len(files.target_channels))

    return run_dir


class StageLogger(RecordingLogger):

    def __init__(self) -> None:
        super().__init__()
        self.timers   = []
        self.tables   = []
        self.totals   = []
        self.advances = 0

    def kv_table(self, data, title=None, **kwargs):
        self.tables.append(title)

    @contextlib.contextmanager
    def timer(self, label: str):
        self.timers.append(label)
        yield

    @contextlib.contextmanager
    def track(self, transient: bool = False):
        stage = self

        class _Progress:
            def add_task(self, description, total=None, **kwargs):
                stage.totals.append(total)
                return 0

            def advance(self, task, advance=1, **kwargs):
                stage.advances += advance

        yield _Progress()
