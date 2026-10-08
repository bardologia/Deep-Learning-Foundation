"""Command-line entry wiring for dual-trunk training runs, sweeps and fan-outs.

Holds the single-run runner that names and configures one dual model, the
scheduler that expands the dual-only routing and ratio trial modes on top of the
backbone modes, and the launcher that parses the entry configuration and decides
between a single run, a seed fan-out and a trial sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from configuration.training               import CurriculumInheritance, DualEntryConfig, dual_curriculum
from configuration.training.general.loss  import ParamMatching
from models                               import BACKBONE_CONFIG_REGISTRY
from models.dual                          import DUAL_CONFIG_REGISTRY
from pipelines.backbone.training.experiments import PatchSizeTrialPlanner
from pipelines.backbone.training.launcher    import SingleTrainRunner, TrainScheduler
from pipelines.dual.inference.pipeline    import DUAL_INFERENCE_COMPONENTS
from pipelines.dual.training.experiments  import DualContextTrialPlanner, DualHeadMatchingTrialPlanner, DualRatioTrialPlanner, DualReachTrialPlanner, DualRoutingTrialPlanner
from pipelines.dual.training.pipeline     import DualTrainingPipeline, TrunkChannelMap
from pipelines.shared.model.model_builder import ModelBuilder
from pipelines.shared.training.ddp_fanout    import DdpDispatch
from pipelines.shared.training.resume_source import ResumeLauncher
from pipelines.shared.training.run_naming    import RunNaming
from pipelines.shared.training.seed_sweep import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from tools.runtime.config_cli             import ConfigCli


class DualSingleTrainRunner(SingleTrainRunner):
    """Runs one dual-trunk training job, naming it after both trunks and their inputs."""

    pipeline_class = DualTrainingPipeline

    @property
    def model_name(self) -> str:
        """Registered dual model name selected by the entry configuration."""
        return self.config.model_name

    @property
    def model_head(self) -> str:
        """Output head of the dual model, always set prediction."""
        return "set_pred"

    @property
    def prior_bias_init(self) -> bool:
        """Always False: the set-prediction head has no single output convolution whose bias could carry the label prior."""
        return False

    @property
    def trunk_tag(self) -> str:
        """Run-name fragment combining the params and existence trunk backbones."""
        return RunNaming.dual_model_tag(self.config.params_backbone, self.config.existence_backbone)

    @property
    def label(self) -> str:
        """Full training run label: trunks, head, Gaussian count, augmentation and trunk inputs."""
        n_gaussians = self.factory.gaussian_config().n_default_gaussians
        return RunNaming.training_tag(self.trunk_tag, self.model_head, self.config.curriculum, n_gaussians, self.config.augmentation, extras=(self._input_tag(),))

    def _input_tag(self) -> str:
        """Returns the `<params>.<existence>` label of the channel groups routed into each trunk."""
        return f"{TrunkChannelMap.group_label(tuple(self.config.params_input))}.{TrunkChannelMap.group_label(tuple(self.config.existence_input))}"

    def _model_config(self):
        """Returns the dual model config built from the dedicated trunk entry fields.

        Raises:
            ValueError : If `model_overrides` also sets a field that has its own trunk entry field.
        """
        trunk_overrides = {
            "params_backbone"     : self.config.params_backbone,
            "existence_backbone"  : self.config.existence_backbone,
            "params_input"        : tuple(self.config.params_input),
            "existence_input"     : tuple(self.config.existence_input),
            "params_overrides"    : dict(self.config.params_overrides),
            "existence_overrides" : dict(self.config.existence_overrides),
        }

        duplicated = [key for key in trunk_overrides if key in self.config.model_overrides]
        if duplicated:
            raise ValueError(f"Select {duplicated} via the dedicated entry fields, not model_overrides")

        return ModelBuilder.config_from_registry(self.config.model_name, {**trunk_overrides, **self.config.model_overrides}, registry=DUAL_CONFIG_REGISTRY)

    def _inference_components(self):
        """Returns the dual inference component set used when the run infers after training."""
        return DUAL_INFERENCE_COMPONENTS


class DualTrainScheduler(TrainScheduler):
    """Schedules dual trial sweeps, adding the routing and ratio modes."""

    SCHEDULER_FIELDS = TrainScheduler.SCHEDULER_FIELDS + ("routing_trials", "ratio_trials")

    MODE_SUBDIRS = {
        **TrainScheduler.MODE_SUBDIRS,
        "routing" : "routing",
        "ratio"   : "ratio",
    }

    def _build_planner(self):
        """Returns the trial planner for the configured mode, deferring unknown modes to the backbone scheduler."""
        mode = self.config.trials_mode

        if mode == "routing":
            return DualRoutingTrialPlanner(self.config.routing_trials, self.config.model_overrides, TrunkChannelMap.GROUPS)

        if mode == "ratio":
            return DualRatioTrialPlanner(
                trials_config   = self.config.ratio_trials,
                model_overrides = self.config.model_overrides,
                model_name      = self.config.model_name,
                backbones       = (self.config.params_backbone, self.config.existence_backbone),
                trunk_channels  = self._trunk_channels(),
                trunk_overrides = self._trunk_overrides(),
                patch_size      = self.config.training.patch_size,
                in_channels     = self._in_channels(),
                out_channels    = self._out_channels(),
            )

        if mode == "context":
            return DualContextTrialPlanner(self.config.context_trials, tuple(BACKBONE_CONFIG_REGISTRY))

        if mode == "reach":
            return DualReachTrialPlanner(self.config.reach_trials, tuple(BACKBONE_CONFIG_REGISTRY), self._in_channels(), self._out_channels(), self.config.model_name, self._trunk_channels(), self._trunk_overrides())

        if mode == "head":
            return DualHeadMatchingTrialPlanner(self.config.head_trials, self.config.params_backbone, tuple(BACKBONE_CONFIG_REGISTRY), ("set_pred",), tuple(matching.value for matching in ParamMatching))

        if mode == "patch":
            if self.config.patch_trials.step <= 0:
                raise ValueError("Dual patch trials cannot derive the admissible step from a single encoder pyramid because the two trunks may differ; set patch_trials.step explicitly")
            return PatchSizeTrialPlanner(self.config.patch_trials, self.config.params_backbone, "set_pred", self.config.model_overrides, self.config.training)

        return super()._build_planner()

    def _in_channels(self) -> int:
        """Returns the input channel count the dataset yields for the configured secondary tracks."""
        n_secondaries = len(self.config.paths.secondary_labels)
        return self.config.input.total_channels(n_secondaries, n_secondaries)

    def _trunk_channels(self) -> tuple:
        """Returns the channel index tuples routed into the params and existence trunks."""
        in_channels = self._in_channels()
        return (TrunkChannelMap.resolve(self.config.input, in_channels, self.config.params_input), TrunkChannelMap.resolve(self.config.input, in_channels, self.config.existence_input))

    def _trunk_overrides(self) -> tuple:
        """Returns the entry-level override dicts of the params and existence trunks, which are forwarded to every unit of the sweep."""
        return (dict(self.config.params_overrides), dict(self.config.existence_overrides))

    def _model_key(self) -> str:
        """Returns the model key combining the dual trunk tag with the set-prediction head."""
        return ModelBuilder.model_key(RunNaming.dual_model_tag(self.config.params_backbone, self.config.existence_backbone), "set_pred")


class DualTrainingLauncher:
    """Parses the dual training command line and dispatches to run, fan-out or sweep.

    Attributes:
        entry_script : Path of the entry script re-invoked by the fan-out and sweep schedulers for each spawned job.
    """

    def __init__(self, entry_script: Path) -> None:
        """Stores the entry script path used when re-invoking this training entry."""
        self.entry_script = Path(entry_script)

    def _configure(self, argv: list[str]):
        """Builds the dual entry configuration from the command line.

        Args:
            argv : Command-line arguments without the program name.

        Returns:
            The resolved entry configuration and the `ConfigCli` holding the
            explicit overrides.

        Raises:
            SystemExit : If both `infer_after` and `infer_at_end` are requested.
        """
        cli    = ConfigCli(DualEntryConfig(), description="Train the dual-trunk set-prediction model (the full stack feeds both trunks by default, a 90 % trunk feeds the gaussian heads and a 10 % trunk feeds the existence gate; each trunk is any backbone-zoo architecture), or fan out training trials across GPUs; point resume_from at a run or sweep to finish it with the configuration it recorded")
        config = cli.apply(argv)
        config = ResumeLauncher.configure(DualEntryConfig, config, cli.overrides, trial_sweep=config.trials_enabled and "--trial" not in argv, unit="--trial" in argv)

        CurriculumInheritance(config.curriculum, dual_curriculum(), cli.overrides).apply()

        if config.infer_after and config.infer_at_end:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        return config, cli

    def build_scheduler(self, argv: list[str]) -> DualTrainScheduler:
        """Returns the trial scheduler configured from the given command line, without running it.

        Args:
            argv : Command-line arguments without the program name.
        """
        config, cli = self._configure(list(argv))
        return DualTrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        """Runs the dual training entry as a single run, a rank group, a seed fan-out or a trial sweep.

        A single seed, or an explicit `--trial` flag from a spawned job, runs the
        training in-process, unless `ddp` fans the single seed out as one
        synchronised rank per `gpus` device; several seeds fan out over the GPU
        pool; otherwise the configured trial sweep is scheduled.

        Args:
            argv : Command-line arguments without the program name, defaulting to `sys.argv[1:]`.

        Raises:
            ValueError : If `ddp` is combined with a trial unit, a trial sweep or a multi-seed fan-out, each of which already claims one GPU per unit.
        """
        argv = list(sys.argv[1:] if argv is None else argv)

        config, cli = self._configure(argv)

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        if trial.trial or not config.trials_enabled:
            seeds = SeedSet.resolve(config.seeds, config.seed)

            if trial.trial:
                DdpDispatch.reject_trials(config, trial_unit=True)
                SeedSweepRunner(config, DualSingleTrainRunner).run()
            elif len(seeds) == 1:
                if not DdpDispatch.single_seed(config, cli.overrides, self.entry_script, DualSingleTrainRunner):
                    SeedSweepRunner(config, DualSingleTrainRunner).run()
            else:
                DdpDispatch.reject_seed_sweep(config, seeds)
                SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, DualSingleTrainRunner, infer_at_end=config.infer_at_end, infer_after=config.infer_after).run()
            return

        DualTrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
