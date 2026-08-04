from __future__ import annotations

import os
from dataclasses import asdict
from pathlib     import Path

import torch

from tensorboard.summary.writer.event_file_writer import EventFileWriter as _  # noqa: F401
from torch.utils.tensorboard                      import SummaryWriter

from configuration.training                     import BackboneTrainerConfig
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from tools.data.io                              import FileIO
from tools.monitoring.logger                    import Logger
from tools.runtime.git_state                    import GitState
from tools.runtime.run_tag                      import RunTag


class TrainingRunMetadata:
    def __init__(self, trainer_config : BackboneTrainerConfig, model_name : str, base_logdir : Path, run_name : str | None = None, logger : Logger | None = None) -> None:
        self.trainer_config = trainer_config
        self.model_name     = model_name
        self.base_logdir    = Path(base_logdir)

        resolved_name = self.resolve_name(model_name, run_name)

        self.run_directory      = self.base_logdir / resolved_name
        self.tensorboard_dir    = self.run_directory / "tensorboard"
        self.docs_directory     = self.run_directory / "docs"
        self.logs_directory     = self.run_directory / "logs"
        self.metadata_directory = self.run_directory / "meta"

        FileIO.ensure_dirs(
            self.run_directory, self.tensorboard_dir, self.docs_directory,
            self.logs_directory, self.metadata_directory,
        )

        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        trainer_config.io.logdir = str(self.run_directory)
        trainer_config.io.writer = self.writer

        self.git_commit = GitState.commit()

        self._owns_logger = logger is None
        file_mode         = "a" if trainer_config.training.resume else "w"
        self.logger       = logger or Logger(log_dir = str(self.logs_directory), name = f"{model_name}_metadata", level = "INFO", file_mode = file_mode)

        self.logger.section("[Training RunMetadata Initialized]")
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset (all GPUs visible)")
        self.logger.kv_table({
            "Run Directory"   : self.run_directory,
            "Model"           : self.model_name,
            "Backend"         : "PyTorch",
            "Git commit"      : self.git_commit,
            "Visible devices" : f"{devices} -> {[torch.cuda.get_device_name(i) for i in range(devices)]} (after CUDA_VISIBLE_DEVICES={visible} masking; indices are relative to the mask)",
        })

    @staticmethod
    def resolve_name(model_name: str, run_name: str | None) -> str:
        return run_name or f"run_{model_name}_{RunTag.now()}"

    def __enter__(self) -> "TrainingRunMetadata":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def save_trainer_config(self) -> Path:
        out_path     = self.docs_directory / "trainer_config.json"
        serializable = asdict(self.trainer_config)
        serializable["io"]["writer"] = None

        FileIO.save_json(serializable, out_path)
        self.logger.info(f"Trainer config saved: {out_path}")

        return out_path

    def save_model_config(self, model_config, model_name: str, config_io=BackboneModelConfigIO) -> Path:
        out_path = config_io.save(model_config, model_name, self.metadata_directory)
        self.logger.info(f"Model config saved: {out_path}")

        return out_path

    def save_run_summary(self, model_name: str, in_channels: int, out_channels: int, x_axis_length: int, n_gaussians: int = 0, seed: int = 0) -> Path:
        out_path = self.metadata_directory / "run_summary.json"

        payload  = {
            "model_name"    : model_name,
            "in_channels"   : in_channels,
            "out_channels"  : out_channels,
            "x_axis_length" : x_axis_length,
            "run_directory" : str(self.run_directory),
            "framework"     : "pytorch",
            "n_devices"     : torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "n_gaussians"   : n_gaussians,
            "seed"          : seed,
            "git_commit"    : self.git_commit,
        }

        FileIO.save_json(payload, out_path)
        self.logger.info(f"Run summary saved: {out_path}")

        return out_path

    def save_test_metrics(self, metrics: dict) -> Path:
        out_path = self.metadata_directory / "test_metrics.json"

        FileIO.save_json(metrics, out_path)
        self.logger.info(f"Test metrics saved: {out_path}")

        return out_path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        if self._owns_logger:
            self.logger.close()
