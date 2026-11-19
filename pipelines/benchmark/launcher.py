
from __future__ import annotations

from pathlib import Path

from configuration.benchmark                    import BenchmarkConfig
from pipelines.benchmark.pipeline               import BenchmarkPipeline
from pipelines.benchmark.stages                 import SeedExpandedStage
from pipelines.shared.orchestration.resumed_run import ResumedRun
from tools.runtime.config_cli                   import ConfigCli
from tools.runtime.run_tag                      import RunTag


class BenchmarkClusterScheduler(SeedExpandedStage):

    SCHEDULER_FIELDS = ("stage", "run_tag", "gpu", "gpus", "gpus_file", "poll_interval_s", "resume_from")

    def __init__(self, config: BenchmarkConfig, cli_overrides: dict, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_tag      = config.run_tag or RunTag.now()
        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag
        self.unit_dirs    = {}
        self.unidentified = []

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self._names = self._expand(config, BenchmarkPipeline.select_models(config))

        if config.resume:
            self._reconcile(self.run_dir / "training")

    def unit_name(self, name: str) -> str:
        return self.unit_dirs.get(name, name)

    @property
    def sweep_name(self) -> str:
        return f"sweep_benchmark_{self.config.training_type}"

    @property
    def entry_flags(self) -> tuple:
        return ()

    @property
    def staged_inference(self) -> bool:
        return self.config.runs_inference()

    def setup_argv(self) -> list[str] | None:
        return ConfigCli.to_argv({**self.forward_overrides, "run_tag": self.run_tag, "stage": "prepare"})

    def finalize_argv(self) -> list[str] | None:
        return ConfigCli.to_argv({**self.forward_overrides, "run_tag": self.run_tag, "stage": "compare"})

    def units(self) -> list[tuple[str, dict]]:
        return [(name, {}) for name in self._names]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        return self._worker_argv("train", run_name)

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        return self._worker_argv("infer", run_name)


class BenchmarkLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def build_scheduler(self, argv: list[str]) -> BenchmarkClusterScheduler:
        cli    = ConfigCli(BenchmarkConfig(), description="Fan the architecture benchmark out over a cluster: prepare, one job per model unit, then compare")
        config = cli.apply(list(argv))

        if config.resume_from:
            config = ResumedRun(BenchmarkConfig, config, cli.overrides).load()

        BenchmarkPipeline.validate_sweep(config)

        return BenchmarkClusterScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)
