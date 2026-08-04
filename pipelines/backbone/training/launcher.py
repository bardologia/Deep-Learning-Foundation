from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from pathlib     import Path

import torch

from configuration.normalization               import Presets
from configuration.training                    import BackboneEntryConfig, CurriculumInheritance, LossScaleProbeConfig, default_curriculum
from configuration.training.general.loss       import ParamMatching
from models                                    import BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS
from pipelines.backbone.inference.pipeline     import InferencePipeline
from pipelines.backbone.training.experiments   import AblationTrialPlanner, AugmentationTrialPlanner, ContextTrialPlanner, CurriculumTrialPlanner, HeadMatchingTrialPlanner, InputTrialPlanner, NormalizationTrialPlanner, PairLossTrialPlanner, PatchSizeTrialPlanner, PhysicsLossTrialPlanner, ReachTrialPlanner, SecondaryTrialPlanner, SlotPresenceTrialPlanner, WarmupTrialPlanner
from pipelines.backbone.training.pipeline      import TrainingPipeline
from pipelines.shared.config.config_factory    import ConfigFactory
from pipelines.shared.model.model_builder      import ModelBuilder
from pipelines.shared.training.run_naming      import RunNaming
from pipelines.shared.training.seed_sweep      import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from pipelines.shared.training.training_runner import SingleTrainRunner as BaseSingleTrainRunner
from tools.data.gaussians                      import GaussianHead
from tools.orchestration                       import ExperimentStage, GpuJob
from tools.monitoring.logger                   import Logger
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.run_tag                     import RunTag


class SingleTrainRunner(BaseSingleTrainRunner):

    pipeline_class = TrainingPipeline

    def __init__(self, config) -> None:
        super().__init__(config)
        self.factory = ConfigFactory(config)

    @property
    def model_name(self) -> str:
        return self.config.backbone_name

    @property
    def model_head(self) -> str:
        return self.config.backbone_head

    @property
    def label(self) -> str:
        n_gaussians = self.factory.gaussian_config().n_default_gaussians
        return RunNaming.training_tag(self.model_name, self.model_head, self.config.curriculum, n_gaussians, self.config.augmentation)

    def _resolve_run_name(self) -> str:
        return RunNaming.compose(self.label, self.config.run_name or RunTag.now())

    def _model_config(self):
        return ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides, head=self.config.backbone_head)

    def _pipeline(self, logdir) -> TrainingPipeline:
        trainer_config            = self.factory.training_trainer_config(logdir=logdir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        return self.pipeline_class(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.model_name,
            model_config   = self._model_config(),
            seed           = self.config.seed,
            run_name       = self.config.run_name,
            overfit_check  = self.config.overfit_check,
        )

    def _build_pretrain_trainer(self, logger):
        work_dir = self.run_directory / "pretrain" / "context"

        return self._pipeline(work_dir).build_pretrain_trainer(work_dir, logger)

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = self.config.probe_enabled,
            n_batches      = self.config.probe_n_batches,
            reference      = self.config.probe_reference,
            exit_after     = self.config.probe_exit_after,
            enabled_losses = dict(self.config.probe_enabled_losses),
        )

    def _inference_components(self):
        return None

    def _run_inference(self, run_directory: Path):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)

        return InferencePipeline(inference_config, components=self._inference_components()).run()

    def run(self):
        self._resolve_run_directory()

        if self._build_unit_resume().skip_training():
            results = None
        else:
            self._pretrain_preflight()
            pipeline = self._pipeline(self.config.logdir)
            results  = pipeline.run(probe_config=self._probe_config(), resolved_entry_config=self.config)

        if (self.config.infer_after or self.config.infer_at_end) and not self.unit_resume.skip_inference(self.config.inference.split):
            self._run_inference(self.run_directory)

        return results


class TrainScheduler:

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "warmup_losses", "complete_losses", "presence_trials", "physics_trials", "pair_trials", "secondary_trials", "patch_trials", "input_trials", "context_trials", "reach_trials", "head_trials", "augmentation_trials", "normalization_trials", "ablation_features", "ablation_include_full", "infer_at_end", "gpus", "gpus_file", "poll_interval_s")

    MODE_SUBDIRS = {
        "curriculum"    : "curriculum",
        "warmup"        : "warmup",
        "presence"      : "presence",
        "physics"       : "physics",
        "pair"          : "pair",
        "secondary"     : "secondary",
        "patch"         : "patch",
        "input"         : "input",
        "context"       : "context",
        "reach"         : "reach",
        "head"          : "head",
        "augmentation"  : "augmentation",
        "normalization" : "normalization",
        "ablation"      : "ablation",
    }

    def __init__(self, config, cli_overrides: dict, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)

        if config.trials_mode not in self.MODE_SUBDIRS:
            raise ValueError(f"Unknown trials_mode '{config.trials_mode}', expected one of {sorted(self.MODE_SUBDIRS)}")

        self.runs_root          = Path(config.logdir) / self.MODE_SUBDIRS[config.trials_mode]
        self.log_dir            = self.runs_root / "batch_train_logs"
        self.results_path       = self.log_dir / "train_scheduler_results.json"
        self.infer_results_path = self.log_dir / "infer_scheduler_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self.logger = Logger(log_dir=str(self.log_dir), name="train_scheduler")
        self.stage  = ExperimentStage(config=config, run_tag="batch_train", logger=self.logger, entry_script=self.entry_script, run_dir=self.runs_root)

    def _out_channels(self) -> int:
        gaussian = ConfigFactory(self.config).gaussian_config()

        return GaussianHead.total_channels(gaussian.params_per_gaussian, gaussian.n_default_gaussians)

    def planner(self):
        mode = self.config.trials_mode
        if mode == "curriculum":
            return CurriculumTrialPlanner(self.config.warmup_losses, self.config.complete_losses)
        if mode == "warmup":
            return WarmupTrialPlanner(self.config.warmup_losses)
        if mode == "presence":
            return SlotPresenceTrialPlanner(self.config.presence_trials)
        if mode == "physics":
            return PhysicsLossTrialPlanner(self.config.physics_trials)
        if mode == "pair":
            return PairLossTrialPlanner(self.config.pair_trials)
        if mode == "secondary":
            return SecondaryTrialPlanner.from_dataset(self.config.secondary_trials, self.config.geometry, self.config.paths.dataset_path)
        if mode == "patch":
            return PatchSizeTrialPlanner(self.config.patch_trials)
        if mode == "input":
            return InputTrialPlanner.from_dataset(self.config.input_trials, self.config.geometry, self.config.paths.dataset_path)
        if mode == "context":
            return ContextTrialPlanner(self.config.context_trials, tuple(BACKBONE_CONFIG_REGISTRY))
        if mode == "reach":
            n_secondaries = len(self.config.paths.secondary_labels)
            return ReachTrialPlanner(self.config.reach_trials, tuple(BACKBONE_CONFIG_REGISTRY), self.config.backbone_head, self.config.input.total_channels(n_secondaries, n_secondaries), self._out_channels())
        if mode == "head":
            return HeadMatchingTrialPlanner(self.config.head_trials, tuple(BACKBONE_CONFIG_REGISTRY), BACKBONE_HEADS, tuple(matching.value for matching in ParamMatching))
        if mode == "augmentation":
            return AugmentationTrialPlanner(self.config.augmentation_trials)
        if mode == "normalization":
            return NormalizationTrialPlanner(self.config.normalization_trials, tuple(Presets.names()))
        if mode == "ablation":
            return AblationTrialPlanner(self.config.ablation_features, self.config.ablation_include_full)

        raise ValueError(f"Unknown trials_mode '{mode}', expected one of {sorted(self.MODE_SUBDIRS)}")

    def _seed_units(self, experiments: list[tuple[str, dict]], seeds: list[int]) -> list[tuple[str, dict]]:
        if len(seeds) == 1:
            return list(experiments)

        return [(SeedSet.run_name(base, seed), {**overrides, "seed": seed, "seeds": (seed,)}) for base, overrides in experiments for seed in seeds]

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        batched = {"infer_after": False} if self.config.infer_at_end else {}
        argv    = ConfigCli.to_argv({**self.forward_overrides, **overrides, **batched, "run_name": run_name, "logdir": str(self.runs_root)})

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + argv,
            log_path = self.log_dir / f"{run_name}.log",
        )

    def _model_key(self) -> str:
        return ModelBuilder.model_key(self.config.backbone_name, self.config.backbone_head)

    def _inference_job(self, run_name: str, overrides: dict) -> GpuJob:
        argv = ConfigCli.to_argv({**self.forward_overrides, **overrides, "infer_after": True, "resume": True, "run_name": run_name, "logdir": str(self.runs_root)})

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + argv,
            log_path = self.log_dir / f"{run_name}_infer.log",
        )

    def _run_inference_pass(self, units: list[tuple[str, dict]], training_results: list[dict]) -> list[dict]:
        trained = {result["name"] for result in training_results if result["status"] == "DONE"}
        todo    = [(run_name, overrides) for run_name, overrides in units if run_name in trained]

        self.logger.section("Batched inference")
        self.logger.kv_table({
            "Eligible" : len(todo),
            "Skipped"  : len(units) - len(todo),
            "GPUs"     : self.config.gpus,
            "Log dir"  : str(self.log_dir),
        }, title="Configuration")

        for run_name, _ in units:
            if run_name not in trained:
                self.logger.warning(f"{run_name}: training failed, inference skipped")

        if not todo:
            return []

        jobs    = [self._inference_job(run_name, overrides) for run_name, overrides in todo]
        names   = [run_name for run_name, _ in todo]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.infer_results_path)
        return results

    def run(self) -> None:
        planner     = self.planner()
        experiments = planner.plan()
        seeds       = SeedSet.resolve(self.config.seeds, self.config.seed)
        units       = self._seed_units(experiments, seeds)

        self.logger.section(f"Training trials: {self.config.trials_mode}")
        self.logger.kv_table({
            "Model (base config)"  : f"{self._model_key()} (per-trial settings may override the backbone/head)",
            "Mode"                 : self.config.trials_mode,
            **planner.summary(),
            "Trials"               : len(experiments),
            "Seeds"                : seeds,
            "GPU jobs"             : len(units),
            "GPUs"                 : self.config.gpus,
            "GPU pool file"        : str(self.stage.pool_file),
            "Infer after"          : self.config.infer_after,
            "Infer at end"         : self.config.infer_at_end,
            "Unit resume"          : f"{self.config.resume} (reuse completed run dirs, delete unfinished ones)",
            "Trainer resume"       : self.config.training.resume,
            "Forwarded overrides (scheduler options excluded; per-trial settings win)" : self.forward_overrides or "—",
            "Log dir"              : str(self.log_dir),
        }, title="Configuration")

        jobs    = [self._job(run_name, overrides) for run_name, overrides in units]
        names   = [run_name for run_name, overrides in units]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.results_path)

        infer_results = self._run_inference_pass(units, results) if self.config.infer_at_end else []

        self.logger.section("Summary")
        rows = [{"Trial": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Trial", "Status", "Duration"])

        if infer_results:
            self.logger.subsection("Batched inference")
            infer_rows = [{"Trial": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in infer_results]
            self.logger.metrics_table(infer_rows, columns=["Trial", "Status", "Duration"])

        failed       = [r for r in results if r["status"] != "DONE"]
        infer_failed = [r for r in infer_results if r["status"] != "DONE"]
        self.stage._log_failures(failed + infer_failed)

        self.logger.close()

        problems = []
        if failed:
            problems.append(f"{len(failed)} of {len(results)} training trials failed")
        if infer_failed:
            problems.append(f"{len(infer_failed)} of {len(infer_results)} inference jobs failed")

        if problems:
            raise SystemExit("; ".join(problems) + f"; see {self.log_dir}")


class BackboneTrainingLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        argv = list(sys.argv[1:] if argv is None else argv)

        cli    = ConfigCli(BackboneEntryConfig(), description="Train one backbone end to end, or fan out curriculum/warmup/secondary trials across GPUs")
        config = cli.apply(argv)

        CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

        if config.infer_after and config.infer_at_end:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        if trial.trial or not config.trials_enabled:
            seeds = SeedSet.resolve(config.seeds, config.seed)

            if trial.trial or len(seeds) == 1:
                SeedSweepRunner(config, SingleTrainRunner).run()
            else:
                SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, SingleTrainRunner, infer_at_end=config.infer_at_end).run()
            return

        TrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
