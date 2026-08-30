"""Cluster launcher fanning the architecture benchmark out as a staged sweep.

Expands the selected models into per-model, per-seed units and frames them with
a prepare stage and a comparison stage.
"""

from __future__ import annotations

from pathlib import Path

from configuration.benchmark      import BenchmarkConfig
from pipelines.benchmark.pipeline import BenchmarkPipeline
from pipelines.benchmark.stages   import SeedExpandedStage
from tools.runtime.config_cli     import ConfigCli
from tools.runtime.run_tag        import RunTag


class BenchmarkClusterScheduler(SeedExpandedStage):
    """Schedules the benchmark on a cluster as prepare, per-unit jobs and compare.

    Attributes:
        config            : Resolved benchmark configuration.
        entry_script      : Entry script each unit job re-invokes.
        run_tag           : Tag identifying this benchmark run.
        run_dir           : Directory holding every artefact of the run.
        forward_overrides : CLI overrides forwarded to the stage jobs, scheduler-only fields removed.
    """

    SCHEDULER_FIELDS = ("stage", "run_tag", "gpu", "gpus", "gpus_file", "poll_interval_s")

    def __init__(self, config: BenchmarkConfig, cli_overrides: dict, entry_script: Path) -> None:
        """Resolves the run directory and expands the selected models into seed units.

        Args:
            config        : Resolved benchmark configuration.
            cli_overrides : Override paths and values parsed from the command line.
            entry_script  : Entry script each unit job re-invokes.
        """
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_tag      = config.run_tag or RunTag.now()
        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self._names = self._expand(config, BenchmarkPipeline.select_models(config))

    @property
    def sweep_name(self) -> str:
        """The cluster sweep name, derived from the benchmarked training type."""
        return f"sweep_benchmark_{self.config.training_type}"

    @property
    def entry_flags(self) -> tuple:
        """The extra entry-script flags, empty because units are selected by worker arguments."""
        return ()

    @property
    def staged_inference(self) -> bool:
        """Whether inference runs as its own stage after all training units."""
        return self.config.runs_inference()

    def setup_argv(self) -> list[str] | None:
        """Returns the argv of the prepare stage, which does size matching and batch probing."""
        return ConfigCli.to_argv({**self.forward_overrides, "run_tag": self.run_tag, "stage": "prepare"})

    def finalize_argv(self) -> list[str] | None:
        """Returns the argv of the compare stage, which writes the comparison reports."""
        return ConfigCli.to_argv({**self.forward_overrides, "run_tag": self.run_tag, "stage": "compare"})

    def units(self) -> list[tuple[str, dict]]:
        """Returns one unit per expanded model, loss component and seed, without overrides."""
        return [(name, {}) for name in self._names]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        """Returns the worker argv training one benchmark unit."""
        return self._worker_argv("train", run_name)

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        """Returns the worker argv running inference over one benchmark unit."""
        return self._worker_argv("infer", run_name)


class BenchmarkLauncher:
    """Builds the cluster scheduler for the architecture benchmark from a command line.

    Attributes:
        entry_script : Entry script the scheduler's jobs re-invoke.
    """

    def __init__(self, entry_script: Path) -> None:
        """Stores the path of the entry script the jobs re-invoke."""
        self.entry_script = Path(entry_script)

    def build_scheduler(self, argv: list[str]) -> BenchmarkClusterScheduler:
        """Parses argv, validates the sweep and returns the configured scheduler.

        Args:
            argv : Command-line arguments without the program name.

        Returns:
            The scheduler for this benchmark sweep.

        Raises:
            SystemExit : If the sweep configuration is invalid.
        """
        cli    = ConfigCli(BenchmarkConfig(), description="Fan the architecture benchmark out over a cluster: prepare, one job per model unit, then compare")
        config = cli.apply(list(argv))

        BenchmarkPipeline.validate_sweep(config)

        return BenchmarkClusterScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)
