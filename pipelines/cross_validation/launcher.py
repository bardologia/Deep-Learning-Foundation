
from __future__ import annotations

from pathlib import Path

from configuration.cross_validation             import CrossValidationConfig
from configuration.training                     import CurriculumInheritance, default_curriculum
from pipelines.cross_validation.pipeline        import CrossValidationPipeline
from pipelines.cross_validation.stages          import FoldUnitStage
from pipelines.shared.orchestration.resumed_run import ResumedRun
from tools.runtime.config_cli                   import ConfigCli
from tools.runtime.run_tag                      import RunTag


class CrossValidationClusterScheduler(FoldUnitStage):

    SCHEDULER_FIELDS = ("stage", "run_tag", "gpu", "gpus", "gpus_file", "poll_interval_s", "resume_from")

    def __init__(self, config: CrossValidationConfig, cli_overrides: dict, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_tag      = config.run_tag or RunTag.now()
        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag
        self.unit_dirs    = {}
        self.unidentified = []

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self._names = self._expand(config)

        if config.resume:
            self._reconcile(self.run_dir / "folds")

    def unit_name(self, name: str) -> str:
        return self.unit_dirs.get(name, name)

    @property
    def sweep_name(self) -> str:
        return f"sweep_cross_validation_{self.config.training_type}"

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


class CrossValidationLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def build_scheduler(self, argv: list[str]) -> CrossValidationClusterScheduler:
        cli    = ConfigCli(CrossValidationConfig(), description="Fan k-fold cross-validation out over a cluster: prepare, one job per fold x seed unit, then the cross-fold report")
        config = cli.apply(list(argv))

        if config.resume_from:
            config = ResumedRun(CrossValidationConfig, config, cli.overrides).load()

        CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

        CrossValidationPipeline.validate_sweep(config)

        return CrossValidationClusterScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)
