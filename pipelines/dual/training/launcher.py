from __future__ import annotations

import argparse
import sys
from pathlib import Path

from configuration.training               import CurriculumInheritance, DualEntryConfig, dual_curriculum
from configuration.training.general.loss  import ParamMatching
from models                               import BACKBONE_CONFIG_REGISTRY
from models.dual                          import DUAL_CONFIG_REGISTRY
from pipelines.backbone.training.launcher import SingleTrainRunner, TrainScheduler
from pipelines.dual.inference.pipeline    import DUAL_INFERENCE_COMPONENTS
from pipelines.dual.training.experiments  import DualContextTrialPlanner, DualHeadMatchingTrialPlanner, DualRatioTrialPlanner, DualReachTrialPlanner, DualRoutingTrialPlanner
from pipelines.dual.training.pipeline     import DualTrainingPipeline, TrunkChannelMap
from pipelines.shared.model.model_builder import ModelBuilder
from pipelines.shared.training.run_naming import RunNaming
from pipelines.shared.training.seed_sweep import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from tools.runtime.config_cli             import ConfigCli


class DualSingleTrainRunner(SingleTrainRunner):

    pipeline_class = DualTrainingPipeline

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def model_head(self) -> str:
        return "set_pred"

    @property
    def trunk_tag(self) -> str:
        return RunNaming.dual_model_tag(self.config.params_backbone, self.config.existence_backbone)

    @property
    def label(self) -> str:
        n_gaussians = self.factory.gaussian_config().n_default_gaussians
        return RunNaming.training_tag(self.trunk_tag, self.model_head, self.config.curriculum, n_gaussians, self.config.augmentation, extras=(self._input_tag(),))

    def _input_tag(self) -> str:
        return f"{TrunkChannelMap.group_label(tuple(self.config.params_input))}.{TrunkChannelMap.group_label(tuple(self.config.existence_input))}"

    def _model_config(self):
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
        return DUAL_INFERENCE_COMPONENTS


class DualTrainScheduler(TrainScheduler):

    SCHEDULER_FIELDS = TrainScheduler.SCHEDULER_FIELDS + ("routing_trials", "ratio_trials")

    MODE_SUBDIRS = {
        **TrainScheduler.MODE_SUBDIRS,
        "routing" : "routing",
        "ratio"   : "ratio",
    }

    def planner(self):
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
                in_channels     = self._in_channels(),
            )

        if mode == "context":
            return DualContextTrialPlanner(self.config.context_trials, tuple(BACKBONE_CONFIG_REGISTRY))

        if mode == "reach":
            return DualReachTrialPlanner(self.config.reach_trials, tuple(BACKBONE_CONFIG_REGISTRY), self._in_channels(), self._out_channels(), self.config.model_name, self._trunk_channels())

        if mode == "head":
            return DualHeadMatchingTrialPlanner(self.config.head_trials, tuple(BACKBONE_CONFIG_REGISTRY), ("set_pred",), tuple(matching.value for matching in ParamMatching))

        return super().planner()

    def _in_channels(self) -> int:
        n_secondaries = len(self.config.paths.secondary_labels)
        return self.config.input.total_channels(n_secondaries, n_secondaries)

    def _trunk_channels(self) -> tuple:
        in_channels = self._in_channels()
        return (TrunkChannelMap.resolve(self.config.input, in_channels, self.config.params_input), TrunkChannelMap.resolve(self.config.input, in_channels, self.config.existence_input))

    def _model_key(self) -> str:
        return ModelBuilder.model_key(RunNaming.dual_model_tag(self.config.params_backbone, self.config.existence_backbone), "set_pred")


class DualTrainingLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        argv = list(sys.argv[1:] if argv is None else argv)

        cli    = ConfigCli(DualEntryConfig(), description="Train the dual-trunk set-prediction model (the full stack feeds both trunks by default, a 90 % trunk feeds the gaussian heads and a 10 % trunk feeds the existence gate; each trunk is any backbone-zoo architecture), or fan out training trials across GPUs")
        config = cli.apply(argv)

        CurriculumInheritance(config.curriculum, dual_curriculum(), cli.overrides).apply()

        if config.infer_after and config.infer_at_end:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        if trial.trial or not config.trials_enabled:
            seeds = SeedSet.resolve(config.seeds, config.seed)

            if trial.trial or len(seeds) == 1:
                SeedSweepRunner(config, DualSingleTrainRunner).run()
            else:
                SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, DualSingleTrainRunner, infer_at_end=config.infer_at_end).run()
            return

        DualTrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
