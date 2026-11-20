
from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import numpy as np

from configuration.cross_validation import CrossValidationConfig
from pipelines.shared.comparison.trial_collection import TrainingResultsIndex, TrialCollector, TrialRecord
from pipelines.shared.inference.metadata          import CompletedInference
from pipelines.shared.inference.run_classifier    import InferenceLayout
from pipelines.shared.orchestration.unit_manifest import UnitResultsByRun
from pipelines.shared.training.unit_resume        import UnitResume
from pipelines.shared.training.worker_base        import WorkerBase
from pipelines.cross_validation.folds             import FoldConfigFactory, FoldNaming, FoldPlanner, FoldRunNaming, FoldUnitManifest
from tools.data.io           import FileIO
from tools.metrics.scoring   import SeedAggregation
from tools.data.regions      import SplitRegions
from tools.monitoring.logger import Logger


class FoldCollector(TrialCollector):

    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self, run_dir: Path, splits: list[str], logger: Logger) -> None:
        super().__init__(run_dir=run_dir, logger=logger, inference_split=splits[0] if splits else None)
        self.training_dir    = run_dir / "folds"
        self.splits          = splits
        self.seed_dispersion = {}
        self.scored_splits   = {}

    def _run_dirs(self) -> list[tuple[str, Path]]:
        return [(name, run_dir) for name, run_dir in super()._run_dirs() if FoldNaming.is_fold(FoldNaming.base(name))]

    def _aggregate_sources(self, runs: list[tuple[str, Path]]) -> tuple[dict, dict]:
        return {}, UnitResultsByRun(self.training_dir, self.logger).build(TrainingResultsIndex(self.pipeline_dir, self.logger), runs)

    def _attach_inference(self, record: TrialRecord) -> None:
        scored = []
        for split in self.splits:
            inference_dir = self._split_inference_dir(record.run_dir, split)
            if inference_dir is not None and (inference_dir / "metrics.json").exists():
                scored.append(split)

        self.scored_splits[record.name] = scored

        if not self.splits or self.splits[0] not in scored:
            return

        fields = self._inference_fields(self._split_inference_dir(record.run_dir, self.splits[0]))
        for name, value in fields.items():
            setattr(record, name, value)

    def _group_by_fold(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(FoldNaming.base(record.name), []).append(record)

        return sorted(groups.items(), key=lambda item: FoldNaming.index(item[0]))

    def _log_split_coverage(self) -> None:
        total = len(self.scored_splits)

        for split in self.splits:
            missing = [name for name, scored in self.scored_splits.items() if split not in scored]
            summary = f"split {split}: {total - len(missing)}/{total} fold runs hold a completed inference with metrics"

            if missing:
                self.logger.warning(f"{summary}; missing: {missing}")
            else:
                self.logger.info(summary)

    def _split_inference_dir(self, run_dir: Path, split: str) -> Path | None:
        return CompletedInference.latest(InferenceLayout.base(run_dir), split)

    def _split_view(self, record: TrialRecord, split: str) -> TrialRecord:
        inference_dir = self._split_inference_dir(record.run_dir, split)

        if inference_dir is None or not (inference_dir / "metrics.json").exists():
            return replace(record, inference_dir=None, metrics={}, figures=[], animations=[], report_path=None)

        return replace(record, **self._inference_fields(inference_dir))

    def _inference_fields(self, inference_dir: Path) -> dict:
        report_path = inference_dir / "report.md"

        return {
            "inference_dir" : inference_dir,
            "metrics"       : FileIO.load_json(inference_dir / "metrics.json"),
            "figures"       : sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else [],
            "animations"    : sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else [],
            "report_path"   : report_path if report_path.exists() else None,
        }

    def _fold_split_record(self, fold_name: str, runs: list[TrialRecord], split: str) -> tuple[TrialRecord, dict, dict]:
        views   = [self._split_view(run, split) for run in runs]
        keys    = sorted({key for view in views for key in view.metrics})
        scored  = [view for view in views if view.metrics]
        missing = [self._seed_label(view.name) for view in views if not view.metrics]

        means, stds    = SeedAggregation.aggregate([view.metrics for view in views], keys)
        representative = next((view for view in views if view.inference_dir is not None), views[0])

        if missing:
            self.logger.warning(f"{fold_name} {split}: metrics averaged over {len(scored)}/{len(views)} seeds; missing {missing} (no completed inference with metrics)")

        dropped = [key for key in keys if key not in means]
        if dropped:
            self.logger.warning(f"{fold_name} {split}: {dropped} non-finite in every seed, dropped from the fold mean")

        shortfall = {key: labels for key, labels in SeedAggregation.shortfall([view.metrics for view in scored], keys, [self._seed_label(view.name) for view in scored]).items() if key not in dropped}
        if shortfall:
            self.logger.warning(f"{fold_name} {split}: {len(shortfall)} metric(s) non-finite in some seed and averaged over fewer, e.g. {dict(list(shortfall.items())[:3])}")

        coverage = {"n_used": len(scored), "missing_seeds": missing, "dropped": dropped}

        return replace(representative, name=fold_name, metrics=means), stds, coverage

    def _fold_base_record(self, fold_name: str, runs: list[TrialRecord]) -> tuple[TrialRecord, float | None]:
        checkpoint, checkpoint_std = SeedAggregation.aggregate([run.checkpoint for run in runs], list(self.CHECKPOINT_KEYS))

        durations = [run.training_result.get("duration_s") for run in runs]
        durations = [value for value in durations if isinstance(value, (int, float))]
        not_done  = [self._seed_label(run.name) for run in runs if run.training_result.get("status") != "DONE"]
        status    = "DONE" if not not_done else "PARTIAL"

        if not_done:
            self.logger.warning(f"{fold_name}: training status PARTIAL; seeds not DONE: {not_done}; their checkpoints, where present, still enter the fold's best-val-loss and best-epoch means")

        no_checkpoint = [self._seed_label(run.name) for run in runs if not run.checkpoint]
        if no_checkpoint:
            self.logger.warning(f"{fold_name}: seeds without a checkpoint: {no_checkpoint}; the checkpoint means cover {len(runs) - len(no_checkpoint)}/{len(runs)} seeds")

        record                 = replace(runs[0], name=fold_name, metrics={}, inference_dir=None, figures=[], animations=[], report_path=None)
        record.checkpoint      = checkpoint
        record.training_result = {"status": status, "duration_s": float(np.mean(durations)) if durations else None}

        return record, checkpoint_std.get("best_val_loss")

    @staticmethod
    def _seed_label(run_name: str) -> str:
        return run_name.rsplit("/", 1)[-1]

    def collect_by_split(self) -> tuple[list[TrialRecord], dict[str, list[TrialRecord]]]:
        self.scored_splits = {}

        groups = self._group_by_fold(self.collect())
        self._log_split_coverage()

        base_records         = []
        records_by_split     = {split: [] for split in self.splits}
        dropped              = {split: set() for split in self.splits}
        self.seed_dispersion = {}

        for fold_name, runs in groups:
            base_record, best_val_loss_std = self._fold_base_record(fold_name, runs)
            base_records.append(base_record)

            split_stds    = {}
            n_used        = {}
            missing_seeds = {}
            for split in self.splits:
                record, stds, coverage = self._fold_split_record(fold_name, runs, split)
                records_by_split[split].append(record)
                dropped[split]      |= set(coverage["dropped"])
                split_stds[split]    = stds
                n_used[split]        = coverage["n_used"]
                missing_seeds[split] = coverage["missing_seeds"]

            self.seed_dispersion[fold_name] = {
                "n_seeds"           : len(runs),
                "status"            : base_record.training_result["status"],
                "best_val_loss_std" : best_val_loss_std,
                "splits"            : split_stds,
                "n_used"            : n_used,
                "missing_seeds"     : missing_seeds,
            }

        short = [fold for fold, entry in self.seed_dispersion.items() if entry["status"] != "DONE" or any(count < entry["n_seeds"] for count in entry["n_used"].values())]
        if short:
            self.logger.warning(f"{len(short)} of {len(self.seed_dispersion)} folds hold a seed that did not finish or average fewer seeds than they hold: {short}")

        for split in self.splits:
            kept    = {key for record in records_by_split[split] for key in record.metrics}
            omitted = sorted(dropped[split] - kept)
            if omitted:
                self.logger.warning(f"split {split}: {omitted} non-finite in every fold, omitted from the cross-fold aggregate (cv_aggregate_report.md, cv_summary.json)")

        return base_records, records_by_split


class CrossValidationWorker(WorkerBase):

    def __init__(self, config: CrossValidationConfig, run_tag: str) -> None:
        super().__init__(config=config, run_tag=run_tag)
        self.factory = FoldConfigFactory(config)
        self.naming  = FoldRunNaming(config)

    def fold_name(self, fold_index: int) -> str:
        return FoldNaming.name(fold_index)

    def fold_run_name(self, fold_index: int, seed: int | None) -> str:
        return self.naming.run_name(fold_index, seed)

    def unit_run_name(self, fold_index: int, seed: int | None, run_name: str | None) -> str:
        return run_name if run_name else self.fold_run_name(fold_index, seed)

    def unit_directory(self, run_name: str) -> Path:
        return self.run_dir / "folds" / run_name


class FoldTrainingWorker(CrossValidationWorker):

    def _spatial_configs(self, split_regions: SplitRegions) -> tuple:
        trainer_config            = self.factory.training_trainer_config(logdir=self.run_dir / "folds")
        trainer_config.curriculum = self.config.curriculum

        dataset_config               = self.factory.training_dataset_config()
        dataset_config.split_regions = split_regions
        dataset_config.input_config  = self.config.input

        return trainer_config, dataset_config

    def _write_manifest(self, fold_index: int, seed: int | None, run_name: str) -> Path:
        planner = self.factory.planner()
        regions = {split: planner.inference_region(fold_index, split).as_tuple() for split in FoldPlanner.INFERENCE_SPLITS}

        return FoldUnitManifest.write(self.unit_directory(run_name), run_tag=self.run_tag, unit=run_name, fold_index=fold_index, seed=seed, inference_regions=regions)

    def _run_backbone(self, fold_index: int, seed: int | None, split_regions: SplitRegions, run_name: str) -> None:
        from pipelines.backbone.training.pipeline import TrainingPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        model_config                   = ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides, head=self.config.backbone_head)
        trainer_config, dataset_config = self._spatial_configs(split_regions)

        pipeline = TrainingPipeline(
            trainer_config  = trainer_config,
            dataset_config  = dataset_config,
            backbone_name   = self.config.backbone_name,
            model_config    = model_config,
            seed            = self.config.seed if seed is None else seed,
            run_name        = run_name,
            prior_bias_init = self.config.prior_bias_init,
        )

        pipeline.run()

    def run(self, fold_index: int, seed: int | None = None, run_name: str | None = None) -> None:
        split_regions = self.factory.planner().plan(fold_index).split_regions
        run_name      = self.unit_run_name(fold_index, seed, run_name)

        self._write_manifest(fold_index, seed, run_name)

        if self.skip_complete(self.unit_directory(run_name), "fold_training", self.config.resume):
            return

        self._run_backbone(fold_index, seed, split_regions, run_name)


class FoldInferenceWorker(CrossValidationWorker):

    def _log(self, run_directory: Path, message: str) -> None:
        logger = Logger(log_dir=str(run_directory / "logs"), name="fold_inference", level="INFO")
        logger.info(message)
        logger.close()

    def _run_inference(self, run_directory: Path, split: str, fold_index: int) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline

        InferencePipeline(self.factory.fold_inference_config(run_directory, split, fold_index)).run()

    def run(self, fold_index: int, seed: int | None = None, run_name: str | None = None) -> None:
        run_directory = self.unit_directory(self.unit_run_name(fold_index, seed, run_name))
        resume        = UnitResume(run_directory, enabled=self.config.resume)

        for split in self.config.inference_splits:
            if resume.skip_inference(split):
                continue

            self._run_inference(run_directory, split, fold_index)
