
from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.training.ddp_fanout    import DdpDispatch
from pipelines.shared.training.resume_source import ResumeLauncher
from pipelines.shared.training.seed_sweep    import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from tools.runtime.config_cli                import ConfigCli


class SeedSweepLauncher:

    def __init__(self, entry_config, runner_class, description: str, entry_script: Path, base_attr: str | None = None, supports_infer_at_end: bool = False) -> None:
        self.entry_config          = entry_config
        self.runner_class          = runner_class
        self.description           = description
        self.entry_script          = Path(entry_script)
        self.base_attr             = base_attr
        self.supports_infer_at_end = supports_infer_at_end

    def run(self, argv: list[str] | None = None) -> None:
        tokens = list(sys.argv[1:] if argv is None else argv)

        cli    = ConfigCli(self.entry_config, description=self.description)
        config = cli.apply(argv)
        config = ResumeLauncher.configure(type(self.entry_config), config, cli.overrides, trial_sweep=False, unit="--trial" in tokens)

        infer_at_end = self.supports_infer_at_end and config.infer_at_end
        infer_after  = self.supports_infer_at_end and config.infer_after
        if infer_at_end and infer_after:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        base_label = getattr(config, self.base_attr) if self.base_attr else None
        seeds      = SeedSet.resolve(config.seeds, config.seed)

        if len(seeds) == 1:
            if not DdpDispatch.single_seed(config, cli.overrides, self.entry_script, self.runner_class, base_label=base_label):
                SeedSweepRunner(config, self.runner_class).run()
            return

        DdpDispatch.reject_seed_sweep(config, seeds)
        SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, self.runner_class, base_label=base_label, infer_at_end=infer_at_end, infer_after=infer_after).run()


class SeedSweepClusterScheduler:

    SCHEDULER_FIELDS = ("gpu", "gpus", "gpus_file", "poll_interval_s", "seed", "seeds", "run_name", "infer_at_end")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, sweep_label: str, base_name: str, infer_at_end: bool) -> None:
        if config.resume_from:
            raise SystemExit("resume_from resumes exactly one local run directory and is not supported for cluster seed sweeps; clear resume_from before scheduling on the cluster")

        if config.ddp:
            raise SystemExit("ddp=True is local-only for now: the fan-out launches its ranks as local subprocesses on the machine's own GPUs and has no cluster job wiring; clear ddp before scheduling on the cluster")

        self.config       = config
        self.entry_script = Path(entry_script)
        self.sweep_label  = sweep_label
        self.base_name    = base_name
        self.infer_at_end = infer_at_end
        self.seeds        = SeedSet.resolve(config.seeds, config.seed)

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

    @property
    def sweep_name(self) -> str:
        return f"sweep_seeds_{self.sweep_label}"

    @property
    def entry_flags(self) -> tuple:
        return ()

    @property
    def staged_inference(self) -> bool:
        return self.infer_at_end

    def setup_argv(self) -> list[str] | None:
        return None

    def finalize_argv(self) -> list[str] | None:
        return None

    def units(self) -> list[tuple[str, dict]]:
        return [(SeedSet.run_name(self.base_name, seed), {"seed": seed, "seeds": (seed,)}) for seed in self.seeds]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        if self.infer_at_end and not batched_inference:
            raise ValueError(f"infer_at_end batches inference after every training unit finishes, so '{run_name}' is built with infer_after disabled; the caller must submit the inference pass itself via inference_argv(), or set infer_after instead to run inference inside each unit")

        batched = {"infer_after": False} if self.infer_at_end else {}

        return ConfigCli.to_argv({**self.forward_overrides, **overrides, **batched, "run_name": run_name, "logdir": str(self.config.logdir)})

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        return ConfigCli.to_argv({**self.forward_overrides, **overrides, "infer_after": True, "resume": True, "run_name": run_name, "logdir": str(self.config.logdir)})


class SeedSweepClusterLauncher:

    entry_config_class    = None
    description           = ""
    sweep_label           = ""
    base_attr             = None
    supports_infer_at_end = False

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def build_scheduler(self, argv: list[str]) -> SeedSweepClusterScheduler:
        cli    = ConfigCli(self.entry_config_class(), description=self.description)
        config = cli.apply(list(argv))

        infer_at_end = self.supports_infer_at_end and config.infer_at_end
        if infer_at_end and config.infer_after:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        base_label = getattr(config, self.base_attr) if self.base_attr else None
        base_name  = SeedSweepRunner.base_run_name(config, base_label)

        return SeedSweepClusterScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script, sweep_label=self.sweep_label, base_name=base_name, infer_at_end=infer_at_end)
