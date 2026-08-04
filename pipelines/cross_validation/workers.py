from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import numpy as np

from configuration.cross_validation import CrossValidationConfig
from pipelines.shared.comparison.trial_collection import TrialCollector, TrialRecord
from pipelines.shared.training.worker_base        import WorkerBase
from pipelines.cross_validation.folds             import FoldConfigFactory, FoldNaming, FoldRunNaming
from tools.data.io           import FileIO
from tools.metrics.scoring   import SeedAggregation
from tools.data.regions      import SplitRegions
from tools.monitoring.logger import Logger


class FoldCollector(TrialCollector):
    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self, run_dir: Path, splits: list[str], logger: Logger) -> None:
        super().__init__(run_dir=run_dir, logger=logger)
        self.training_dir    = run_dir / "folds"
        self.splits          = splits
        self.seed_dispersion = {}

    def _aggregate_sources(self) -> tuple[dict, dict]:
        training_results = {r["name"]: r for r in FileIO.load_json(self.pipeline_dir / "training_results.json")}

        return {}, training_results

    def _group_by_fold(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(FoldNaming.base(record.name), []).append(record)

        return sorted(groups.items(), key=lambda item: FoldNaming.index(item[0]))

    def _split_view(self, record: TrialRecord, split: str) -> TrialRecord:
        inference_dir = record.run_dir / "inference" / split

        if not (inference_dir / "metrics.json").exists():
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

    def _fold_split_record(self, fold_name: str, runs: list[TrialRecord], split: str) -> tuple[TrialRecord, dict]:
        views = [self._split_view(run, split) for run in runs]
        keys  = sorted({key for view in views for key in view.metrics})

        means, stds   = SeedAggregation.aggregate([view.metrics for view in views], keys)
        representative = next((view for view in views if view.inference_dir is not None), views[0])

        return replace(representative, name=fold_name, metrics=means), stds

    def _fold_base_record(self, fold_name: str, runs: list[TrialRecord]) -> tuple[TrialRecord, float | None]:
        checkpoint, checkpoint_std = SeedAggregation.aggregate([run.checkpoint for run in runs], list(self.CHECKPOINT_KEYS))

        durations = [run.training_result.get("duration_s") for run in runs]
        durations = [value for value in durations if isinstance(value, (int, float))]
        status    = "DONE" if all(run.training_result.get("status") == "DONE" for run in runs) else "PARTIAL"

        record                 = replace(runs[0], name=fold_name, metrics={}, inference_dir=None, figures=[], animations=[], report_path=None)
        record.checkpoint      = checkpoint
        record.training_result = {"status": status, "duration_s": float(np.mean(durations)) if durations else None}

        return record, checkpoint_std.get("best_val_loss")

    def collect_by_split(self) -> tuple[list[TrialRecord], dict[str, list[TrialRecord]]]:
        groups = self._group_by_fold(self.collect())

        base_records         = []
        records_by_split     = {split: [] for split in self.splits}
        self.seed_dispersion = {}

        for fold_name, runs in groups:
            base_record, best_val_loss_std = self._fold_base_record(fold_name, runs)
            base_records.append(base_record)

            split_stds = {}
            for split in self.splits:
                record, stds = self._fold_split_record(fold_name, runs, split)
                records_by_split[split].append(record)
                split_stds[split] = stds

            self.seed_dispersion[fold_name] = {
                "n_seeds"           : len(runs),
                "best_val_loss_std" : best_val_loss_std,
                "splits"            : split_stds,
            }

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


class FoldTrainingWorker(CrossValidationWorker):
    def _run_backbone(self, fold_index: int, seed: int | None, split_regions: SplitRegions) -> None:
        from pipelines.backbone.training.pipeline import TrainingPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        model_config = ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides, head=self.config.backbone_head)

        trainer_config            = self.factory.training_trainer_config(logdir=self.run_dir / "folds")
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        dataset_config               = self.factory.training_dataset_config()
        dataset_config.split_regions = split_regions

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.config.backbone_name,
            model_config   = model_config,
            seed           = self.config.seed if seed is None else seed,
            run_name       = self.fold_run_name(fold_index, seed),
        )

        pipeline.run(probe_config=self._probe_config())

    def _run_jepa(self, fold_index: int, seed: int | None, split_regions: SplitRegions) -> None:
        from pipelines.jepa.training.pipeline import TrainingPipeline

        TrainingPipeline(self._jepa_entry_config(self.fold_run_name(fold_index, seed), seed), split_regions=split_regions).run()

    def _jepa_entry_config(self, run_name: str, seed: int | None):
        from configuration.training import JepaEntryConfig

        cv   = self.config
        jepa = cv.jepa

        return JepaEntryConfig(
            run_name                   = run_name,
            backbone_name              = cv.backbone_name,
            backbone_head              = cv.backbone_head,
            seed                       = cv.seed if seed is None else seed,
            logdir                     = self.run_dir / "folds",
            model_overrides            = cv.model_overrides,
            profile_autoencoder_logdir = jepa.profile_autoencoder_logdir,
            profile_autoencoder_run    = jepa.profile_autoencoder_run,
            profile_autoencoder_mode   = jepa.profile_autoencoder_mode,
            image_autoencoder_logdir   = jepa.image_autoencoder_logdir,
            image_autoencoder_run      = jepa.image_autoencoder_run,
            image_autoencoder_mode     = jepa.image_autoencoder_mode,
            target_provider            = jepa.target_provider,
            embedding_loss             = jepa.embedding_loss,
            param_loss                 = jepa.param_loss,
            geometry                   = cv.geometry,
            paths                      = cv.paths,
            training                   = cv.training,
            normalization              = cv.normalization,
            augmentation               = cv.augmentation,
        )

    def _run_profile_autoencoder(self, fold_index: int, seed: int | None, split_regions: SplitRegions) -> None:
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        TrainingPipeline(self._ae_entry_config(self.fold_run_name(fold_index, seed), seed), split_regions=split_regions).run()

    def _ae_entry_config(self, run_name: str, seed: int | None):
        from configuration.training import ProfileAeEntryConfig

        cv = self.config
        ae = cv.autoencoder

        return ProfileAeEntryConfig(
            run_name        = run_name,
            seed            = cv.seed if seed is None else seed,
            logdir          = self.run_dir / "folds",
            pixel_subsample = ae.pixel_subsample,
            keep_empty_frac = ae.keep_empty_frac,
            ae_model_name   = ae.ae_model_name,
            model_overrides = cv.model_overrides,
            ae_loss         = ae.ae_loss,
            geometry        = cv.geometry,
            paths           = cv.paths,
            training        = cv.training,
        )

    def run(self, fold_index: int, seed: int | None = None) -> None:
        dispatch = {
            "backbone"            : self._run_backbone,
            "jepa"                : self._run_jepa,
            "profile_autoencoder" : self._run_profile_autoencoder,
        }

        training_type = self.config.training_type
        if training_type not in dispatch:
            raise ValueError(f"Unknown training_type '{training_type}', expected one of {sorted(dispatch)}")

        split_regions = self.factory.planner().plan(fold_index).split_regions
        dispatch[training_type](fold_index, seed, split_regions)


class FoldInferenceWorker(CrossValidationWorker):
    def run(self, fold_index: int, split: str, seed: int | None = None) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        run_directory = self.run_dir / "folds" / self.fold_run_name(fold_index, seed)

        components = InferenceComponentsResolver.for_run(run_directory)
        pipeline   = InferencePipeline(self.factory.fold_inference_config(run_directory, split), components=components)
        pipeline.run()
