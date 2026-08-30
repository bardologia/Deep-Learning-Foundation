"""Run directory layout, TensorBoard writer and metadata files of a training run.

Creates the run/tensorboard/docs/logs/meta tree, opens the summary writer and the
run logger, and writes the trainer config, architecture config, run summary and
test metrics that make a run self-describing.
"""

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
    """Owns the on-disk identity of one training run.

    Usable as a context manager; exiting flushes and closes the TensorBoard
    writer and, when it created one, the logger.

    Attributes:
        trainer_config     : Trainer configuration, whose io.logdir is repointed at the created run directory.
        model_name         : Architecture name the run trains.
        base_logdir        : Parent directory the run directory is created under.
        run_directory      : Directory holding every artifact of this run.
        tensorboard_dir    : TensorBoard event directory.
        docs_directory     : Human-readable documentation and serialized configs.
        logs_directory     : Text logs of the run.
        metadata_directory : Machine-readable metadata (meta/).
        writer             : Open TensorBoard summary writer.
        git_commit         : Commit the run was launched from.
        logger             : Logger used for the metadata section.
    """

    def __init__(self, trainer_config : BackboneTrainerConfig, model_name : str, base_logdir : Path, run_name : str | None = None, logger : Logger | None = None) -> None:
        """Creates the run directory tree and opens the writer and logger.

        Args:
            trainer_config : Trainer configuration to persist and repoint.
            model_name     : Architecture name the run trains.
            base_logdir    : Parent directory of the run directory.
            run_name       : Explicit run directory name; a timestamped name is derived from the model name when omitted.
            logger         : Existing logger to reuse; when omitted a run-owned logger is created under logs/, appending on resume and truncating otherwise.
        """
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
        """Returns the explicit run name, or a timestamped name built from the model name."""
        return run_name or f"run_{model_name}_{RunTag.now()}"

    def __enter__(self) -> "TrainingRunMetadata":
        """Returns this metadata object for use in a with-statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Closes the writer and the owned logger on leaving the context."""
        self.close()

    def save_trainer_config(self) -> Path:
        """Writes docs/trainer_config.json and returns its path."""
        out_path     = self.docs_directory / "trainer_config.json"
        serializable = asdict(self.trainer_config)

        FileIO.save_json(serializable, out_path)
        self.logger.info(f"Trainer config saved: {out_path}")

        return out_path

    def save_model_config(self, model_config, model_name: str, config_io=BackboneModelConfigIO) -> Path:
        """Persists the architecture config under meta/ and returns its path.

        Args:
            model_config : Architecture configuration dataclass instance.
            model_name   : Registry name of the architecture.
            config_io    : Family-specific writer selecting the filename and format.
        """
        out_path = config_io.save(model_config, model_name, self.metadata_directory)
        self.logger.info(f"Model config saved: {out_path}")

        return out_path

    def save_run_summary(self, model_name: str, in_channels: int, out_channels: int, x_axis_length: int, n_gaussians: int = 0, seed: int = 0) -> Path:
        """Writes meta/run_summary.json describing the model IO shape and returns its path.

        Args:
            model_name    : Architecture name the run trains.
            in_channels   : Number of input channels fed to the model.
            out_channels  : Number of channels the model predicts.
            x_axis_length : Number of z bins on the reconstruction axis.
            n_gaussians   : Number of Gaussian components predicted per pixel; zero for models with no parametric head.
            seed          : Random seed the run was trained with.
        """
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
        """Writes meta/test_metrics.json and returns its path.

        Args:
            metrics : Scalar test metrics keyed by metric name.
        """
        out_path = self.metadata_directory / "test_metrics.json"

        FileIO.save_json(metrics, out_path)
        self.logger.info(f"Test metrics saved: {out_path}")

        return out_path

    def close(self) -> None:
        """Flushes and closes the TensorBoard writer, and the logger when owned."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        if self._owns_logger:
            self.logger.close()
