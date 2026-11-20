
from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from pathlib     import Path

import torch

from configuration.training                       import BackboneEntryConfig, CurriculumInheritance, default_curriculum
from configuration.training.general.loss          import ParamMatching
from configuration.training.general.optimization  import ClipMode, WarmupMode
from models                                       import BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS
from pipelines.backbone.inference.pipeline        import InferencePipeline
from pipelines.backbone.training.experiments      import AblationTrialPlanner, AugmentationTrialPlanner, ClampTrialPlanner, ContextTrialPlanner, CurriculumTrialPlanner, HeadMatchingTrialPlanner, InputTrialPlanner, LogEnergyTrialPlanner, PairLossTrialPlanner, PatchSizeTrialPlanner, ReachTrialPlanner, SecondaryTrialPlanner, SlotPresenceTrialPlanner, WarmupTrialPlanner
from pipelines.backbone.training.patch_report     import PatchTrialsReport
from pipelines.backbone.training.pipeline         import TrainingPipeline
from pipelines.shared.config.config_factory       import ConfigFactory
from pipelines.shared.model.model_builder         import ModelBuilder
from pipelines.shared.training.ddp_fanout         import DdpDispatch
from pipelines.shared.training.pretrain_preflight import PretrainPreflight
from pipelines.shared.training.resume_source      import ResumeLauncher
from pipelines.shared.training.run_naming         import RunNaming
from pipelines.shared.training.seed_sweep         import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from pipelines.shared.training.training_runner    import RunDirectorySettledMarker
from pipelines.shared.training.trial_planners     import AmpTrialPlanner, EmaTrialPlanner, StabilityTrialPlanner
from pipelines.shared.training.trial_sweep        import TrialSweepScheduler
from pipelines.shared.training.unit_resume        import UnitResume
from tools.data.gaussians                         import GaussianHead
from tools.runtime.config_cli                     import ConfigCli
from tools.runtime.run_tag                        import RunTag
from tools.training.checkpoint                    import TrainerState


class SingleTrainRunner:

    def __init__(self, config) -> None:
        self.config        = config
        self.run_directory = None
        self.unit_resume   = None
        self.factory       = ConfigFactory(config)

    @property
    def model_name(self) -> str:
        return self.config.backbone_name

    @property
    def model_head(self) -> str:
        return self.config.backbone_head

    @property
    def prior_bias_init(self) -> bool:
        return self.config.prior_bias_init

    @property
    def label(self) -> str:
        n_gaussians = self.factory.gaussian_config().n_default_gaussians
        return RunNaming.training_tag(self.model_name, self.model_head, self.config.curriculum, n_gaussians, self.config.augmentation)

    def _apply_resume_from(self) -> None:
        target = Path(self.config.resume_from).expanduser()

        self.config.logdir          = target.parent
        self.config.run_name        = target.name
        self.config.resume          = True
        self.config.training.resume = TrainerState.path(target).is_file()

    def _resolve_run_name(self) -> str:
        return RunNaming.compose(self.label, self.config.run_name or RunTag.now())

    def _resolve_run_directory(self) -> None:
        if self.config.resume_from:
            self._apply_resume_from()
        else:
            self.config.run_name = self._resolve_run_name()

        self.run_directory = Path(self.config.logdir) / self.config.run_name

    def _build_unit_resume(self) -> UnitResume:
        self.unit_resume = UnitResume(self.run_directory, enabled=self.config.resume, trainer_resume=self.config.training.resume, config=self.config)
        return self.unit_resume

    def _model_config(self):
        return ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides, head=self.config.backbone_head)

    def _pipeline(self, logdir) -> TrainingPipeline:
        trainer_config            = self.factory.training_trainer_config(logdir=logdir)
        trainer_config.curriculum = self.config.curriculum

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        return TrainingPipeline(
            trainer_config  = trainer_config,
            dataset_config  = dataset_config,
            backbone_name   = self.model_name,
            model_config    = self._model_config(),
            seed            = self.config.seed,
            run_name        = self.config.run_name,
            overfit_check   = self.config.overfit_check,
            prior_bias_init = self.prior_bias_init,
        )

    def _build_pretrain_trainer(self, logger):
        work_dir = self.run_directory / "pretrain" / "context"

        return self._pipeline(work_dir).build_pretrain_trainer(work_dir, logger)

    def _pretrain_preflight(self) -> None:
        PretrainPreflight(
            pretrain_config = self.config.pretrain,
            training_config = self.config.training,
            build_trainer   = self._build_pretrain_trainer,
            run_directory   = self.run_directory,
            label           = self.label,
        ).run()

    def _run_inference(self, run_directory: Path):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)

        return InferencePipeline(inference_config).run()

    def run(self):
        self._resolve_run_directory()

        if self._build_unit_resume().skip_training():
            results = None
        else:
            RunDirectorySettledMarker.write(self.run_directory)

            self._pretrain_preflight()
            pipeline = self._pipeline(self.config.logdir)
            results  = pipeline.run(resolved_entry_config=self.config)

        if (self.config.infer_after or self.config.infer_at_end) and not self.unit_resume.skip_inference(self.config.inference.split):
            self._run_inference(self.run_directory)

        return results


class TrainScheduler(TrialSweepScheduler):

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "warmup_losses", "complete_losses", "presence_trials", "pair_trials", "log_energy_trials", "secondary_trials", "patch_trials", "input_trials", "context_trials", "reach_trials", "head_trials", "augmentation_trials", "clamp_trials", "stability_trials", "ema_trials", "ablation_features", "ablation_include_full", "infer_at_end", "gpu", "gpus", "gpus_file", "poll_interval_s", "resume_from")

    MODE_SUBDIRS = {
        "curriculum"    : "curriculum",
        "warmup"        : "warmup",
        "presence"      : "presence",
        "pair"          : "pair",
        "log_energy"    : "log_energy",
        "secondary"     : "secondary",
        "patch"         : "patch",
        "input"         : "input",
        "context"       : "context",
        "reach"         : "reach",
        "head"          : "head",
        "augmentation"  : "augmentation",
        "clamp"         : "clamp",
        "stability"     : "stability",
        "ema"           : "ema",
        "amp"           : "amp",
        "ablation"      : "ablation",
    }

    @property
    def staged_inference(self) -> bool:
        return self.config.infer_at_end

    @property
    def unit_inference(self) -> bool:
        return self.config.infer_after

    def _inference_rows(self) -> dict:
        return {
            "Infer after"  : self.config.infer_after,
            "Infer at end" : self.config.infer_at_end,
        }

    def _out_channels(self) -> int:
        gaussian = ConfigFactory(self.config).gaussian_config()

        return GaussianHead.total_channels(gaussian.params_per_gaussian, gaussian.n_default_gaussians)

    def _build_planner(self):
        mode = self.config.trials_mode
        if mode == "curriculum":
            return CurriculumTrialPlanner(self.config.warmup_losses, self.config.complete_losses)
        if mode == "warmup":
            return WarmupTrialPlanner(self.config.warmup_losses)
        if mode == "presence":
            return SlotPresenceTrialPlanner(self.config.presence_trials)
        if mode == "pair":
            return PairLossTrialPlanner(self.config.pair_trials)
        if mode == "log_energy":
            return LogEnergyTrialPlanner(self.config.log_energy_trials)
        if mode == "secondary":
            return SecondaryTrialPlanner.from_dataset(self.config.secondary_trials, self.config.paths.dataset_path)
        if mode == "patch":
            return PatchSizeTrialPlanner(self.config.patch_trials, self.config.backbone_name, self.config.backbone_head, self.config.model_overrides, self.config.training)
        if mode == "input":
            return InputTrialPlanner.from_dataset(self.config.input_trials, self.config.paths.dataset_path)
        if mode == "context":
            return ContextTrialPlanner(self.config.context_trials, tuple(BACKBONE_CONFIG_REGISTRY))
        if mode == "reach":
            n_secondaries = len(self.config.paths.secondary_labels)
            return ReachTrialPlanner(self.config.reach_trials, tuple(BACKBONE_CONFIG_REGISTRY), self.config.backbone_head, self.config.input.total_channels(n_secondaries, n_secondaries), self._out_channels())
        if mode == "head":
            return HeadMatchingTrialPlanner(self.config.head_trials, self.config.backbone_name, tuple(BACKBONE_CONFIG_REGISTRY), BACKBONE_HEADS, tuple(matching.value for matching in ParamMatching))
        if mode == "augmentation":
            return AugmentationTrialPlanner(self.config.augmentation_trials)
        if mode == "clamp":
            return ClampTrialPlanner(self.config.clamp_trials)
        if mode == "stability":
            return StabilityTrialPlanner(self.config.stability_trials, tuple(mode.value for mode in WarmupMode), tuple(mode.value for mode in ClipMode))
        if mode == "ema":
            return EmaTrialPlanner(self.config.ema_trials)
        if mode == "amp":
            return AmpTrialPlanner()
        return AblationTrialPlanner(self.config.ablation_features, self.config.ablation_include_full)

    def _model_key(self) -> str:
        return ModelBuilder.model_key(self.config.backbone_name, self.config.backbone_head)

    def _write_mode_report(self) -> None:
        if self.config.trials_mode != "patch":
            return

        out_dir = self.runs_root / "report" / RunTag.now()
        written = PatchTrialsReport(config=self.config, planner=self.planner(), model_label=self._model_key(), runs_root=self.runs_root, out_dir=out_dir, logger=self.logger).write_all()

        self.logger.subsection("Patch sweep report")
        for path in written:
            self.logger.info(f"{path}")

    @staticmethod
    def _same_value(trial_value, forwarded) -> bool:
        if isinstance(trial_value, (list, tuple)) and isinstance(forwarded, (list, tuple)):
            return list(trial_value) == list(forwarded)

        return trial_value == forwarded

    def _warn_shadowed_overrides(self) -> None:
        experiments = self.planner().plan()

        for path, value in self.forward_overrides.items():
            replaced = [overrides[path] for _name, overrides in experiments if path in overrides and not self._same_value(overrides[path], value)]
            if not replaced:
                continue

            trial_values = ", ".join(dict.fromkeys(str(trial_value) for trial_value in replaced))
            self.logger.warning(f"Forwarded override {path}={value} is replaced by the {self.config.trials_mode} planner in {len(replaced)} of {len(experiments)} trial(s) (trial values: {trial_values}); those units train on the trial values")

    def run(self) -> None:
        self._warn_shadowed_overrides()
        super().run()


class BackboneTrainingLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def _configure(self, argv: list[str]):
        cli    = ConfigCli(BackboneEntryConfig(), description="Train one backbone end to end, or fan out curriculum/warmup/secondary trials across GPUs; point resume_from at a run or sweep to finish it with the configuration it recorded")
        config = cli.apply(argv)
        config = ResumeLauncher.configure(BackboneEntryConfig, config, cli.overrides, trial_sweep=config.trials_enabled and "--trial" not in argv, unit="--trial" in argv)

        CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

        return config, cli

    def build_scheduler(self, argv: list[str]) -> TrainScheduler:
        config, cli = self._configure(list(argv))
        return TrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        argv = list(sys.argv[1:] if argv is None else argv)

        config, cli = self._configure(argv)

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        if trial.trial or not config.trials_enabled:
            seeds = SeedSet.resolve(config.seeds, config.seed)

            if trial.trial:
                DdpDispatch.reject_trials(config, trial_unit=True)
                SeedSweepRunner(config, SingleTrainRunner).run()
            elif len(seeds) == 1:
                if not DdpDispatch.single_seed(config, cli.overrides, self.entry_script, SingleTrainRunner):
                    SeedSweepRunner(config, SingleTrainRunner).run()
            else:
                DdpDispatch.reject_seed_sweep(config, seeds)
                SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, SingleTrainRunner, infer_at_end=config.infer_at_end, infer_after=config.infer_after).run()
            return

        TrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
