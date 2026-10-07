"""Run directory layout, TensorBoard writer and metadata files of a training run.

Creates the run/tensorboard/docs/logs/meta tree, opens the run logger, clears
an unfinished earlier attempt and refuses a completed run, opens the summary
writer, and writes the trainer config, architecture config, run summary and
test metrics that make a run self-describing.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib     import Path

import torch

from rich.markup                                  import escape
from tensorboard.summary.writer.event_file_writer import EventFileWriter as _
from torch.utils.tensorboard                      import SummaryWriter

from configuration.training                     import BackboneTrainerConfig
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from tools.data.io                              import DatasetFingerprint, FileIO
from tools.monitoring.logger                    import Logger
from tools.runtime.completion                   import CompletionMarker
from tools.runtime.environment_capture          import EnvironmentCapture
from tools.runtime.git_state                    import GitState, RunProvenance
from tools.runtime.run_tag                      import RunTag
from tools.training.checkpoint                  import TrainerState


class RunDirectoryLeftovers:
    """Finds the artifacts an earlier training run left in a run directory.

    A run that does not resume must not land on top of them: a second
    TensorBoard event file would draw on the same tags, stale checkpoints and
    inference stamps would pass for the new model's, and an old completion
    marker would let a later resume reuse the directory without retraining.
    A relaunch therefore deletes what an unfinished attempt left, which is
    what a preempted, killed or crashed unit finds when its queue or a manual
    relaunch starts it again, and refuses a directory holding a completed
    run, which is never overwritten in place. The logs stay, so the earlier
    attempt's traceback survives the relaunch.

    Attributes:
        FILES       : Run-level files only training itself writes.
        DIRECTORIES : Run-level directories only training or its inference pass create.
        EVENTS_GLOB : Pattern of the TensorBoard event files below the run directory.
    """

    FILES       = (CompletionMarker.FILENAME, TrainerState.FILENAME, "best_model.pt", "final_model.pt")
    DIRECTORIES = ("checkpoints", "inference")
    EVENTS_GLOB = "tensorboard/events.out.tfevents.*"

    @classmethod
    def find(cls, run_directory: Path) -> list[str]:
        """Returns the earlier run's artifacts present in the directory, as paths relative to it.

        Args:
            run_directory : Run directory to inspect; an absent directory holds nothing.
        """
        run_directory = Path(run_directory)
        if not run_directory.is_dir():
            return []

        files       = [name for name in cls.FILES if (run_directory / name).is_file()]
        directories = [name for name in cls.DIRECTORIES if (run_directory / name).is_dir()]
        events      = sorted(str(path.relative_to(run_directory)) for path in run_directory.glob(cls.EVENTS_GLOB))

        return files + directories + events

    @classmethod
    def clear(cls, run_directory: Path, trainer_resume: bool, logger: Logger) -> None:
        """Deletes an unfinished earlier attempt's artifacts before a run that does not resume, and refuses a completed run.

        Args:
            run_directory  : Run directory the run is about to use.
            trainer_resume : Whether the trainer continues the run from its checkpoint, which keeps the artifacts by design.
            logger         : Logger the deletion warning is written to.

        Raises:
            ValueError : If `trainer_resume` is False and the directory holds a completed run; the message names the directory and the artifacts.
        """
        if trainer_resume:
            return

        run_directory = Path(run_directory)
        found         = cls.find(run_directory)
        if not found:
            return

        if CompletionMarker.is_complete(run_directory):
            raise ValueError(f"Run directory {run_directory} already holds a completed run ({', '.join(found)}) and training.resume is False, so a fresh run would overwrite it. Pick a new run_name, set resume=True to reuse the completed run, or delete the directory.")

        logger.warning(f"Run directory {run_directory} holds an unfinished earlier attempt; deleting its artifacts before the fresh run starts: {', '.join(found)}")

        for name in found:
            path = run_directory / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


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
        provenance         : Host, interpreter, environment and library versions of the launching process.
        git_diff           : Path of the saved diff when tracked files carried uncommitted changes, else None.
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

        Raises:
            ValueError : If the run does not resume and its directory already holds a completed run; an unfinished earlier attempt is deleted instead.
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

        self._owns_logger = logger is None
        file_mode         = "a" if trainer_config.training.resume else "w"
        self.logger       = logger or Logger(log_dir = str(self.logs_directory), name = f"{model_name}_metadata", level = "INFO", file_mode = file_mode)

        try:
            RunDirectoryLeftovers.clear(self.run_directory, trainer_config.training.resume, self.logger)
        except ValueError as error:
            if self._owns_logger:
                self.logger.close(error)
            raise

        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        trainer_config.io.logdir = str(self.run_directory)

        self.git_commit = GitState.commit()
        self.provenance = RunProvenance.collect()
        self.git_diff   = self._save_provenance()

        self.logger.section("[Training RunMetadata Initialized]")
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset (all GPUs visible)")
        self.logger.kv_table({
            "Run Directory"   : self.run_directory,
            "Model"           : self.model_name,
            "Backend"         : "PyTorch",
            "Git commit"      : self.git_commit,
            "Host"            : f"{self.provenance['host']} (python {self.provenance['python']}, conda env {self.provenance['conda_env']})",
            "Machine profile" : self.provenance["machine_profile"] or "none",
            "Libraries"       : f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version() if devices else 'not queried (no visible device)'}, numpy {self.provenance['numpy']}",
            "Visible devices" : f"{devices} -> {[torch.cuda.get_device_name(i) for i in range(devices)]} (after CUDA_VISIBLE_DEVICES={visible} masking; indices are relative to the mask)",
        })

        if self.git_diff is not None:
            self.logger.warning(f"Tracked files carry uncommitted changes; their diff is saved as {self.git_diff}, so the code this run executes is the recorded commit plus that patch.")

    @staticmethod
    def resolve_name(model_name: str, run_name: str | None) -> str:
        """Returns the explicit run name, or a timestamped name built from the model name."""
        return run_name or f"run_{model_name}_{RunTag.now()}"

    def _save_provenance(self) -> Path | None:
        """Writes the environment snapshot and the git working-tree state of the run.

        meta/environment.json records the interpreter, library and device state.
        docs/git_state.json records the commit, dirty flag and untracked paths,
        and docs/git_diff.patch holds the uncommitted tracked changes whenever
        the working tree is dirty, so a dirty run stays reconstructable from
        the recorded commit plus that patch.

        Returns:
            Path of the saved diff, or None when the tracked files carry no changes.
        """
        FileIO.save_json(EnvironmentCapture.snapshot(), self.metadata_directory / "environment.json")

        state = GitState.dirty_state()
        FileIO.save_json({"commit": self.git_commit, "dirty": state["dirty"], "untracked": state["untracked"]}, self.docs_directory / "git_state.json")

        if not state["dirty"]:
            return None

        path = self.docs_directory / "git_diff.patch"
        path.write_text(state["diff"], encoding="utf-8")

        return path

    def save_dataset_fingerprint(self, dataset_path, parameters_path, enabled: bool = True) -> Path | None:
        """Hashes the dataset artifacts into meta/dataset_fingerprint.json.

        Args:
            dataset_path    : Dataset directory whose data/ files are hashed.
            parameters_path : Parameter-label array hashed alongside, or None to hash the dataset files only.
            enabled         : Whether fingerprinting runs at all.

        Returns:
            The written path, or None when fingerprinting is disabled.
        """
        if not enabled:
            return None

        dataset_path = Path(dataset_path)
        files        = sorted(path for path in (dataset_path / "data").iterdir() if path.is_file())

        if parameters_path is not None:
            files.append(Path(parameters_path))

        payload  = DatasetFingerprint.compute(dataset_path, files)
        out_path = self.metadata_directory / "dataset_fingerprint.json"

        FileIO.save_json(payload, out_path)
        self.logger.info(f"Dataset fingerprint saved: {out_path}")

        return out_path

    def __enter__(self) -> "TrainingRunMetadata":
        """Returns this metadata object for use in a with-statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Closes the writer and the owned logger, recording the escaping exception when there is one."""
        self.close(exc_value if exc_type is not None else None)

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

    def save_run_summary(self, model_name: str, in_channels: int, out_channels: int, x_axis_length: int, n_gaussians: int = 0, seed: int = 0, patch_geometry: dict | None = None) -> Path:
        """Writes meta/run_summary.json describing the model IO shape and returns its path.

        Args:
            model_name     : Architecture name the run trains.
            in_channels    : Number of input channels fed to the model.
            out_channels   : Number of channels the model predicts.
            x_axis_length  : Number of z bins on the reconstruction axis.
            n_gaussians    : Number of Gaussian components predicted per pixel; zero for models with no parametric head.
            seed           : Random seed the run was trained with.
            patch_geometry : Configured and effective patch size and stride of a patch-trained model, or None for a run that records none.
        """
        out_path  = self.metadata_directory / "run_summary.json"
        n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

        payload  = {
            "model_name"     : model_name,
            "in_channels"    : in_channels,
            "out_channels"   : out_channels,
            "x_axis_length"  : x_axis_length,
            "run_directory"  : str(self.run_directory),
            "framework"      : "pytorch",
            "n_devices"      : n_devices,
            "n_gaussians"    : n_gaussians,
            "seed"           : seed,
            "patch_geometry" : patch_geometry,
            "git_commit"     : self.git_commit,
            "git_diff"       : str(self.git_diff) if self.git_diff is not None else None,
            "cudnn"          : torch.backends.cudnn.version() if n_devices else None,
            "provenance"     : self.provenance,
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

    def close(self, error: BaseException | None = None) -> None:
        """Flushes and closes the TensorBoard writer, and the logger when owned.

        Args:
            error : Exception that ended the run, written into the run log with its traceback before the log closes; None when the run completed.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        if self._owns_logger:
            self.logger.close(error)
        elif Logger.aborted(error):
            self.logger.error(f"[Training Aborted] {type(error).__name__}: {escape(str(error))}")
