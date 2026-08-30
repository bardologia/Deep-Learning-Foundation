"""Cluster scheduling of a cross-validation sweep.

Turns a cross-validation entry configuration into the staged prepare, per-unit
train and infer, and compare jobs the cluster scheduler submits.
"""

from __future__ import annotations

from pathlib import Path

from configuration.cross_validation      import CrossValidationConfig
from configuration.training              import CurriculumInheritance, default_curriculum
from pipelines.cross_validation.pipeline import CrossValidationPipeline
from pipelines.cross_validation.stages   import FoldUnitStage
from tools.runtime.config_cli            import ConfigCli
from tools.runtime.run_tag               import RunTag


class CrossValidationClusterScheduler(FoldUnitStage):
    """Describes the cross-validation sweep as cluster jobs.

    Attributes:
        config            : Cross-validation configuration for the sweep.
        entry_script      : Entry point re-invoked by every job.
        run_tag           : Tag identifying the sweep run directory.
        run_dir           : Sweep run directory.
        forward_overrides : CLI overrides passed on to the jobs, with the scheduler-local fields removed.
    """

    SCHEDULER_FIELDS = ("stage", "run_tag", "gpu", "gpus", "gpus_file", "poll_interval_s")

    def __init__(self, config: CrossValidationConfig, cli_overrides: dict, entry_script: Path) -> None:
        """Expands the fold x seed units and fixes the sweep run directory.

        Args:
            config        : Cross-validation configuration for the sweep.
            cli_overrides : Overrides typed on the command line, keyed by dotted config path.
            entry_script  : Entry point re-invoked by every job.
        """
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_tag      = config.run_tag or RunTag.now()
        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self._names = self._expand(config)

    @property
    def sweep_name(self) -> str:
        """Name identifying the sweep on the cluster."""
        return f"sweep_cross_validation_{self.config.training_type}"

    @property
    def entry_flags(self) -> tuple:
        """Extra flags every job's entry command carries; none for this sweep."""
        return ()

    @property
    def staged_inference(self) -> bool:
        """Whether the sweep schedules a separate inference stage."""
        return self.config.runs_inference()

    def setup_argv(self) -> list[str] | None:
        """Returns the argv of the prepare job that validates the fold plan."""
        return ConfigCli.to_argv({**self.forward_overrides, "run_tag": self.run_tag, "stage": "prepare"})

    def finalize_argv(self) -> list[str] | None:
        """Returns the argv of the compare job that writes the cross-fold report."""
        return ConfigCli.to_argv({**self.forward_overrides, "run_tag": self.run_tag, "stage": "compare"})

    def units(self) -> list[tuple[str, dict]]:
        """Returns the (run name, per-unit overrides) pairs of the sweep."""
        return [(name, {}) for name in self._names]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        """Returns the argv training one fold x seed unit."""
        return self._worker_argv("train", run_name)

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        """Returns the argv running inference on one fold x seed unit."""
        return self._worker_argv("infer", run_name)


class CrossValidationLauncher:
    """Parses sweep arguments and builds the cluster scheduler.

    Attributes:
        entry_script : Entry point re-invoked by every job.
    """

    def __init__(self, entry_script: Path) -> None:
        """Binds the launcher to the entry script the jobs re-invoke."""
        self.entry_script = Path(entry_script)

    def build_scheduler(self, argv: list[str]) -> CrossValidationClusterScheduler:
        """Parses the sweep command line and returns the configured scheduler.

        Args:
            argv : Command-line arguments of the sweep invocation.

        Returns:
            A scheduler describing the prepare, unit, and compare jobs.

        Raises:
            SystemExit : If the stage or requested inference splits are invalid.
        """
        cli    = ConfigCli(CrossValidationConfig(), description="Fan k-fold cross-validation out over a cluster: prepare, one job per fold x seed unit, then the cross-fold report")
        config = cli.apply(list(argv))

        CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

        CrossValidationPipeline.validate_sweep(config)

        return CrossValidationClusterScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)
