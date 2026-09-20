"""Entry-point launchers turning CLI arguments into seed runs or seed sweeps.

Local launches either train the single requested seed in process or fan the
seed set out across the GPU pool. The cluster variants instead emit the per-unit
argument vectors that the Terrabyte scheduler packs into container jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.training.ddp_fanout    import DdpDispatch
from pipelines.shared.training.resume_source import ResumeLauncher
from pipelines.shared.training.seed_sweep    import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from tools.runtime.config_cli                import ConfigCli


class SeedSweepLauncher:
    """Parses an entry's CLI and dispatches a single seed run or a local sweep.

    Attributes:
        entry_config          : Entry configuration instance the CLI parses into.
        runner_class          : Callable turning a config into a single-run runner.
        description           : CLI description shown in the entry's help text.
        entry_script          : Entry script re-invoked by the per-seed jobs.
        base_attr             : Config attribute supplying the base label, or None.
        supports_infer_at_end : Whether this entry offers batched inference.
    """

    def __init__(self, entry_config, runner_class, description: str, entry_script: Path, base_attr: str | None = None, supports_infer_at_end: bool = False) -> None:
        """Stores the entry configuration, runner class and launcher options."""
        self.entry_config          = entry_config
        self.runner_class          = runner_class
        self.description           = description
        self.entry_script          = Path(entry_script)
        self.base_attr             = base_attr
        self.supports_infer_at_end = supports_infer_at_end

    def run(self, argv: list[str] | None = None) -> None:
        """Parses the CLI and runs one seed in process, data-parallel, or fans the seed set out.

        A single resolved seed trains in process, unless ``ddp`` is set, in
        which case it is fanned out as one synchronised rank per ``gpus``
        device; several seeds are fanned out across the GPU pool one seed each.
        A process that is itself a rank of a running group always trains in
        place. With ``resume_from`` set the configuration the run or sweep
        recorded is loaded first, and the typed overrides are applied on top.

        Args:
            argv : Argument vector to parse; None reads ``sys.argv``.

        Raises:
            SystemExit : If ``infer_after`` and ``infer_at_end`` are both set.
            ValueError : If ``ddp`` is combined with a multi-seed sweep.
        """
        tokens = list(sys.argv[1:] if argv is None else argv)

        cli    = ConfigCli(self.entry_config, description=self.description)
        config = cli.apply(argv)
        config = ResumeLauncher.configure(type(self.entry_config), config, cli.overrides, trial_sweep=False, unit="--trial" in tokens)

        infer_at_end = self.supports_infer_at_end and config.infer_at_end
        if infer_at_end and config.infer_after:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        base_label = getattr(config, self.base_attr) if self.base_attr else None
        seeds      = SeedSet.resolve(config.seeds, config.seed)

        if len(seeds) == 1:
            if not DdpDispatch.single_seed(config, cli.overrides, self.entry_script, self.runner_class, base_label=base_label):
                SeedSweepRunner(config, self.runner_class).run()
            return

        DdpDispatch.reject_seed_sweep(config, seeds)
        SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, self.runner_class, base_label=base_label, infer_at_end=infer_at_end).run()


class SeedSweepClusterScheduler:
    """Describes a seed sweep as cluster units for the Terrabyte scheduler.

    Attributes:
        SCHEDULER_FIELDS  : Top-level config fields owned by the scheduler and never forwarded to the units.
        config            : Entry configuration of the sweep.
        entry_script      : Entry script each unit invokes on the cluster.
        sweep_label       : Label identifying the sweep family in its folder name.
        base_name         : Sweep base run name.
        infer_at_end      : Whether inference is submitted as a separate staged pass.
        seeds             : Seeds making up the sweep.
        forward_overrides : CLI overrides forwarded to every unit.
    """

    SCHEDULER_FIELDS = ("gpu", "gpus", "gpus_file", "poll_interval_s", "seed", "seeds", "run_name", "infer_at_end")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, sweep_label: str, base_name: str, infer_at_end: bool) -> None:
        """Stores the sweep description and filters the forwarded overrides.

        Args:
            config        : Entry configuration of the sweep.
            cli_overrides : All CLI overrides seen by the entry.
            entry_script  : Entry script each unit invokes on the cluster.
            sweep_label   : Label identifying the sweep family.
            base_name     : Sweep base run name.
            infer_at_end  : Whether inference runs as a separate staged pass.

        Raises:
            SystemExit : If ``resume_from`` is set, which only resumes a single local run directory, or if ``ddp`` is set, which is local-only.
        """
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
        """Name of the cluster sweep folder for this seed sweep."""
        return f"sweep_seeds_{self.sweep_label}"

    @property
    def entry_flags(self) -> tuple:
        """Extra flags prepended to every unit's argument vector; none here."""
        return ()

    @property
    def staged_inference(self) -> bool:
        """Whether inference is scheduled as its own stage after training."""
        return self.infer_at_end

    def setup_argv(self) -> list[str] | None:
        """Returns the prepare-stage argument vector; this sweep needs none."""
        return None

    def finalize_argv(self) -> list[str] | None:
        """Returns the compare-stage argument vector; this sweep needs none."""
        return None

    def units(self) -> list[tuple[str, dict]]:
        """Returns one (run name, seed overrides) unit per seed in the sweep."""
        return [(SeedSet.run_name(self.base_name, seed), {"seed": seed, "seeds": (seed,)}) for seed in self.seeds]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        """Returns the CLI argument vector training one unit on the cluster.

        Args:
            run_name          : Run name of the unit.
            overrides         : Unit-specific config overrides, such as the seed.
            batched_inference : Set by the caller that also submits the separate inference pass, acknowledging that this vector disables per-unit inference.

        Returns:
            The argument vector for the unit's training job.

        Raises:
            ValueError : If ``infer_at_end`` is set but the caller did not acknowledge the batched inference pass.
        """
        if self.infer_at_end and not batched_inference:
            raise ValueError(f"infer_at_end batches inference after every training unit finishes, so '{run_name}' is built with infer_after disabled; the caller must submit the inference pass itself via inference_argv(), or set infer_after instead to run inference inside each unit")

        batched = {"infer_after": False} if self.infer_at_end else {}

        return ConfigCli.to_argv({**self.forward_overrides, **overrides, **batched, "run_name": run_name, "logdir": str(self.config.logdir)})

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        """Returns the CLI argument vector running inference for a trained unit.

        Args:
            run_name  : Run name of the unit.
            overrides : Unit-specific config overrides, such as the seed.

        Returns:
            The argument vector resuming the unit with inference enabled.
        """
        return ConfigCli.to_argv({**self.forward_overrides, **overrides, "infer_after": True, "resume": True, "run_name": run_name, "logdir": str(self.config.logdir)})


class SeedSweepClusterLauncher:
    """Base launcher building a cluster scheduler for one entry's seed sweep.

    Subclasses set the entry configuration class and the sweep labels; the base
    class parses the CLI and hands back a configured scheduler.

    Attributes:
        entry_config_class    : Entry configuration class the CLI parses into.
        description           : CLI description shown in the entry's help text.
        sweep_label           : Label identifying the sweep family.
        base_attr             : Config attribute supplying the base label, or None.
        supports_infer_at_end : Whether this entry offers batched inference.
        entry_script          : Entry script each unit invokes on the cluster.
    """

    entry_config_class    = None
    description           = ""
    sweep_label           = ""
    base_attr             = None
    supports_infer_at_end = False

    def __init__(self, entry_script: Path) -> None:
        """Stores the entry script each cluster unit invokes."""
        self.entry_script = Path(entry_script)

    def build_scheduler(self, argv: list[str]) -> SeedSweepClusterScheduler:
        """Parses the CLI and builds the cluster scheduler for the seed sweep.

        Args:
            argv : Argument vector to parse.

        Returns:
            The configured cluster scheduler.

        Raises:
            SystemExit : If ``infer_after`` and ``infer_at_end`` are both set.
        """
        cli    = ConfigCli(self.entry_config_class(), description=self.description)
        config = cli.apply(list(argv))

        infer_at_end = self.supports_infer_at_end and config.infer_at_end
        if infer_at_end and config.infer_after:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        base_label = getattr(config, self.base_attr) if self.base_attr else None
        base_name  = SeedSweepRunner.base_run_name(config, base_label)

        return SeedSweepClusterScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script, sweep_label=self.sweep_label, base_name=base_name, infer_at_end=infer_at_end)
