"""Per-unit workers and fold collection for cross-validation.

Holds the collector that folds seed runs back into one record per fold, and the
worker processes that train or run inference on a single fold x seed unit in
whichever model family the sweep selected.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import numpy as np

from configuration.cross_validation import CrossValidationConfig
from pipelines.shared.comparison.trial_collection import TrainingResultsIndex, TrialCollector, TrialRecord
from pipelines.shared.training.unit_resume        import UnitResume
from pipelines.shared.training.worker_base        import WorkerBase
from pipelines.cross_validation.folds             import FoldConfigFactory, FoldFamily, FoldNaming, FoldPlanner, FoldRunNaming, FoldUnitManifest
from tools.data.io           import FileIO
from tools.metrics.scoring   import SeedAggregation
from tools.data.regions      import SplitRegions
from tools.monitoring.logger import Logger


class FoldCollector(TrialCollector):
    """Collects fold runs and aggregates their seed runs into per-fold records.

    Attributes:
        training_dir    : Directory holding the fold run directories.
        splits          : Dataset splits whose inference metrics are collected.
        training_type   : Training family, deciding where inference lives.
        seed_dispersion : Within-fold seed statistics keyed by fold name.
    """

    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self, run_dir: Path, splits: list[str], training_type: str, logger: Logger) -> None:
        """Binds the collector to a sweep run directory.

        Args:
            run_dir       : Sweep run directory holding the `folds/` subtree.
            splits        : Dataset splits whose inference metrics are collected.
            training_type : Training family the folds were trained in.
            logger        : Logger used for progress reporting.
        """
        super().__init__(run_dir=run_dir, logger=logger, inference_split=splits[0] if splits else None)
        self.training_dir    = run_dir / "folds"
        self.splits          = splits
        self.training_type   = training_type
        self.seed_dispersion = {}

    def _run_dirs(self) -> list[tuple[str, Path]]:
        """Returns the fold runs, leaving out directories under the folds directory that are no fold, such as comparison stamps."""
        return [(name, run_dir) for name, run_dir in super()._run_dirs() if FoldNaming.is_fold(FoldNaming.base(name))]

    def _aggregate_sources(self, runs: list[tuple[str, Path]]) -> tuple[dict, dict]:
        """Returns the empty completion index and the training-results index of the runs."""
        return {}, TrainingResultsIndex(self.pipeline_dir, self.logger).build(runs)

    def _group_by_fold(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        """Groups seed runs under their fold name, ordered by fold index."""
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(FoldNaming.base(record.name), []).append(record)

        return sorted(groups.items(), key=lambda item: FoldNaming.index(item[0]))

    def _split_view(self, record: TrialRecord, split: str) -> TrialRecord:
        """Returns a copy of a run record populated from one split's inference run.

        Args:
            record : Record of a single seed run.
            split  : Dataset split to read.

        Returns:
            A copy carrying that split's metrics, figures, and report, with the
            inference fields cleared when the split has no completed run.
        """
        inference_dir = FoldFamily.split_inference_dir(record.run_dir, self.training_type, split)

        if inference_dir is None or not (inference_dir / "metrics.json").exists():
            return replace(record, inference_dir=None, metrics={}, figures=[], animations=[], report_path=None)

        return replace(record, **self._inference_fields(inference_dir))

    def _inference_fields(self, inference_dir: Path) -> dict:
        """Returns the record fields read out of one completed inference directory."""
        report_path = inference_dir / "report.md"

        return {
            "inference_dir" : inference_dir,
            "metrics"       : FileIO.load_json(inference_dir / "metrics.json"),
            "figures"       : sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else [],
            "animations"    : sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else [],
            "report_path"   : report_path if report_path.exists() else None,
        }

    def _fold_split_record(self, fold_name: str, runs: list[TrialRecord], split: str) -> tuple[TrialRecord, dict, dict]:
        """Aggregates a fold's seed runs into one record for a split, naming the seeds that contributed nothing.

        Args:
            fold_name : Name of the fold.
            runs      : Records of the fold's seed runs.
            split     : Dataset split to aggregate.

        Returns:
            A tuple (record, stds, coverage) where the record carries seed-mean
            metrics and the figures of a representative seed, stds maps each
            metric to its across-seed standard deviation, and coverage holds
            ``n_used``, the seeds whose metrics entered the means, and
            ``missing_seeds``, the labels of the seeds without metrics.
        """
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

        coverage = {"n_used": len(scored), "missing_seeds": missing}

        return replace(representative, name=fold_name, metrics=means), stds, coverage

    def _fold_base_record(self, fold_name: str, runs: list[TrialRecord]) -> tuple[TrialRecord, float | None]:
        """Aggregates a fold's seed runs into one training-only record, naming the seeds that did not finish.

        Args:
            fold_name : Name of the fold.
            runs      : Records of the fold's seed runs.

        Returns:
            A tuple (record, best_val_loss_std) where the record carries the
            seed-mean checkpoint fields and mean duration in seconds, and the
            second element is the across-seed standard deviation of the best
            validation loss.
        """
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
        """Returns the seed directory of a run name, or the name itself for an unseeded run."""
        return run_name.rsplit("/", 1)[-1]

    def collect_by_split(self) -> tuple[list[TrialRecord], dict[str, list[TrialRecord]]]:
        """Collects every fold and splits its records by dataset split.

        Returns:
            A tuple (base_records, records_by_split) holding one training-only
            record per fold and, for each split, one metric-carrying record per
            fold. `seed_dispersion` is filled in as a side effect, carrying per
            fold the seed count, the training status, the seeds used per split
            and the seeds missing from each split.
        """
        groups = self._group_by_fold(self.collect())

        base_records         = []
        records_by_split     = {split: [] for split in self.splits}
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

        return base_records, records_by_split


class CrossValidationWorker(WorkerBase):
    """Shared config factory and naming for the per-unit cross-validation workers.

    Attributes:
        factory : Builds the fold planner and per-fold configs.
        naming  : Composes fold run names with the family training tag.
    """

    def __init__(self, config: CrossValidationConfig, run_tag: str) -> None:
        """Binds the worker to a cross-validation sweep run.

        Args:
            config  : Cross-validation configuration.
            run_tag : Tag of the sweep run directory.
        """
        super().__init__(config=config, run_tag=run_tag)
        self.factory = FoldConfigFactory(config)
        self.naming  = FoldRunNaming(config)

    def fold_name(self, fold_index: int) -> str:
        """Returns the seedless name of a fold."""
        return FoldNaming.name(fold_index)

    def fold_run_name(self, fold_index: int, seed: int | None) -> str:
        """Returns the full run name of one fold and seed."""
        return self.naming.run_name(fold_index, seed)

    def unit_run_name(self, fold_index: int, seed: int | None, run_name: str | None) -> str:
        """Returns the run name the unit lives under: the one the scheduler handed over, else the composed one.

        Args:
            fold_index : Zero-based index of the fold.
            seed       : Seed of the unit, or None for an unseeded run.
            run_name   : Directory the scheduler found for this unit relative to the folds directory, or None to compose the name.
        """
        return run_name if run_name else self.fold_run_name(fold_index, seed)

    def unit_directory(self, run_name: str) -> Path:
        """Returns the run directory of a unit from its name relative to the folds directory."""
        return self.run_dir / "folds" / run_name


class FoldTrainingWorker(CrossValidationWorker):
    """Trains one fold x seed unit in the configured model family."""

    def _spatial_configs(self, split_regions: SplitRegions) -> tuple:
        """Builds the trainer and dataset configs of a spatial family fold.

        Args:
            split_regions : Guard-trimmed train, val, and test crop regions.

        Returns:
            A tuple (trainer_config, dataset_config) with the fold's regions
            and resolved acquisition geometry applied.
        """
        trainer_config            = self.factory.training_trainer_config(logdir=self.run_dir / "folds")
        trainer_config.curriculum = self.config.curriculum

        dataset_config               = self.factory.training_dataset_config()
        dataset_config.split_regions = split_regions
        dataset_config.input_config  = self.config.input

        return trainer_config, dataset_config

    def _write_manifest(self, fold_index: int, seed: int | None, run_name: str) -> Path:
        """Records the unit's identity and scored regions in its run directory before training starts.

        Args:
            fold_index : Zero-based index of the fold.
            seed       : Seed of the unit, or None for an unseeded run.
            run_name   : Run name of the unit relative to the folds directory.

        Returns:
            Path of the written manifest.
        """
        planner = self.factory.planner()
        regions = {split: planner.inference_region(fold_index, split).as_tuple() for split in FoldPlanner.INFERENCE_SPLITS}

        return FoldUnitManifest.write(self.unit_directory(run_name), run_tag=self.run_tag, unit=run_name, fold_index=fold_index, seed=seed, training_type=self.config.training_type, inference_regions=regions)

    def _run_backbone(self, fold_index: int, seed: int | None, split_regions: SplitRegions, run_name: str) -> None:
        """Trains one backbone fold unit."""
        from pipelines.backbone.training.pipeline import TrainingPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        model_config                   = ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides, head=self.config.backbone_head)
        trainer_config, dataset_config = self._spatial_configs(split_regions)

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.config.backbone_name,
            model_config   = model_config,
            seed           = self.config.seed if seed is None else seed,
            run_name       = run_name,
            overfit_check  = self.config.overfit_check,
        )

        pipeline.run(probe_config=self.config.probe)

    def _run_dual(self, fold_index: int, seed: int | None, split_regions: SplitRegions, run_name: str) -> None:
        """Trains one dual-trunk fold unit."""
        from pipelines.dual.training.pipeline import DualTrainingPipeline

        model_config                   = self._dual_model_config()
        trainer_config, dataset_config = self._spatial_configs(split_regions)

        pipeline = DualTrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.config.dual.model_name,
            model_config   = model_config,
            seed           = self.config.seed if seed is None else seed,
            run_name       = run_name,
            overfit_check  = self.config.overfit_check,
        )

        pipeline.run(probe_config=self.config.probe)

    def _dual_model_config(self):
        """Builds the dual model config from the dedicated trunk entry fields.

        Returns:
            The dual model configuration.

        Raises:
            ValueError : If a trunk field was also passed through `model_overrides`, which would silently conflict.
        """
        from models.dual                          import DUAL_CONFIG_REGISTRY
        from pipelines.shared.model.model_builder import ModelBuilder

        dual = self.config.dual

        trunk_overrides = {
            "params_backbone"     : dual.params_backbone,
            "existence_backbone"  : dual.existence_backbone,
            "params_input"        : tuple(dual.params_input),
            "existence_input"     : tuple(dual.existence_input),
            "params_overrides"    : dict(dual.params_overrides),
            "existence_overrides" : dict(dual.existence_overrides),
        }

        duplicated = [key for key in trunk_overrides if key in self.config.model_overrides]
        if duplicated:
            raise ValueError(f"Select {duplicated} via the dedicated dual.* entry fields, not model_overrides")

        return ModelBuilder.config_from_registry(dual.model_name, {**trunk_overrides, **self.config.model_overrides}, registry=DUAL_CONFIG_REGISTRY)

    def _run_jepa(self, fold_index: int, seed: int | None, split_regions: SplitRegions, run_name: str) -> None:
        """Trains one JEPA fold unit."""
        from pipelines.jepa.training.pipeline import TrainingPipeline

        TrainingPipeline(self._jepa_entry_config(run_name, seed), split_regions=split_regions).run()

    def _jepa_entry_config(self, run_name: str, seed: int | None):
        """Builds the JEPA training entry config for one fold unit."""
        from configuration.training import JepaEntryConfig

        cv   = self.config
        jepa = cv.jepa

        return JepaEntryConfig(
            run_name        = run_name,
            backbone_name   = cv.backbone_name,
            backbone_head   = cv.backbone_head,
            seed            = cv.seed if seed is None else seed,
            logdir          = self.run_dir / "folds",
            model_overrides = cv.model_overrides,
            input           = cv.input,
            paths           = cv.paths,
            training        = cv.training,
            normalization   = cv.normalization,
            augmentation    = cv.augmentation,
            overfit_check   = cv.overfit_check,
            **jepa.entry_kwargs(),
        )

    def _run_profile_autoencoder(self, fold_index: int, seed: int | None, split_regions: SplitRegions, run_name: str) -> None:
        """Trains one profile-autoencoder fold unit."""
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        TrainingPipeline(self._ae_entry_config(run_name, seed), split_regions=split_regions).run()

    def _ae_entry_config(self, run_name: str, seed: int | None):
        """Builds the profile-autoencoder training entry config for one fold unit."""
        from configuration.training import ProfileAeEntryConfig

        cv = self.config
        ae = cv.autoencoder

        return ProfileAeEntryConfig(
            run_name             = run_name,
            seed                 = cv.seed if seed is None else seed,
            logdir               = self.run_dir / "folds",
            pixel_subsample      = ae.pixel_subsample,
            keep_empty_frac      = ae.keep_empty_frac,
            profile_augmentation = ae.profile_augmentation,
            ae_model_name        = ae.ae_model_name,
            model_overrides      = cv.model_overrides,
            ae_loss              = ae.ae_loss,
            paths                = cv.paths,
            training             = cv.training,
            overfit_check        = cv.overfit_check,
        )

    def _run_image_autoencoder(self, fold_index: int, seed: int | None, split_regions: SplitRegions, run_name: str) -> None:
        """Trains one image-autoencoder fold unit."""
        from pipelines.image_autoencoder.training.pipeline import TrainingPipeline

        TrainingPipeline(self._image_ae_entry_config(run_name, seed), split_regions=split_regions).run()

    def _image_ae_entry_config(self, run_name: str, seed: int | None):
        """Builds the image-autoencoder training entry config for one fold unit."""
        from configuration.training import ImageAeEntryConfig

        cv = self.config
        ae = cv.image_autoencoder

        return ImageAeEntryConfig(
            run_name        = run_name,
            seed            = cv.seed if seed is None else seed,
            logdir          = self.run_dir / "folds",
            ae_model_name   = ae.ae_model_name,
            model_overrides = cv.model_overrides,
            ae_loss         = ae.ae_loss,
            paths           = cv.paths,
            training        = cv.training,
            normalization   = cv.normalization,
            augmentation    = cv.augmentation,
            overfit_check   = cv.overfit_check,
        )

    def run(self, fold_index: int, seed: int | None = None, run_name: str | None = None) -> None:
        """Trains one fold unit, dispatching on the configured training family.

        The unit's manifest is written first, so even a run that dies before
        its first epoch is identifiable when the sweep is resumed; a unit that
        already finished is left as it is on a resume and refused when resume
        is off.

        Args:
            fold_index : Zero-based index of the fold.
            seed       : Seed of the unit, or None for an unseeded run.
            run_name   : Directory of the unit relative to the folds directory, or None to compose it from the fold and seed.

        Raises:
            ValueError : If the training family is unsupported.
            SystemExit : If the unit already holds a completed run while resume is off.
        """
        dispatch = {
            "backbone"            : self._run_backbone,
            "dual"                : self._run_dual,
            "jepa"                : self._run_jepa,
            "profile_autoencoder" : self._run_profile_autoencoder,
            "image_autoencoder"   : self._run_image_autoencoder,
        }

        training_type = FoldFamily.validate(self.config.training_type)
        split_regions = self.factory.planner().plan(fold_index).split_regions
        run_name      = self.unit_run_name(fold_index, seed, run_name)

        self._write_manifest(fold_index, seed, run_name)

        if self.skip_complete(self.unit_directory(run_name), "fold_training", self.config.resume):
            return

        dispatch[training_type](fold_index, seed, split_regions, run_name)


class FoldInferenceWorker(CrossValidationWorker):
    """Runs inference for one fold x seed unit on every configured split."""

    def _log(self, run_directory: Path, message: str) -> None:
        """Writes a single message into the run's fold-inference log."""
        logger = Logger(log_dir=str(run_directory / "logs"), name="fold_inference", level="INFO")
        logger.info(message)
        logger.close()

    def _run_spatial(self, run_directory: Path, split: str, fold_index: int) -> None:
        """Runs backbone-style inference restricted to the fold's row block."""
        from pipelines.backbone.inference.pipeline           import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        components = InferenceComponentsResolver.for_run(run_directory)

        InferencePipeline(self.factory.fold_inference_config(run_directory, split, fold_index), components=components).run()

    def _run_profile_autoencoder(self, run_directory: Path, split: str, fold_index: int) -> None:
        """Runs profile-autoencoder inference for one fold and split."""
        from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferencePipeline

        ProfileAeInferencePipeline(self.factory.fold_profile_ae_inference_config(run_directory, split)).run()

    def _run_image_autoencoder(self, run_directory: Path, split: str, fold_index: int) -> None:
        """Runs image-autoencoder inference for one fold and split."""
        from pipelines.image_autoencoder.inference.pipeline import ImageAeInferencePipeline

        ImageAeInferencePipeline(self.factory.fold_image_ae_inference_config(run_directory, split)).run()

    def run(self, fold_index: int, seed: int | None = None, run_name: str | None = None) -> None:
        """Runs inference for one fold unit on every configured split.

        Splits whose inference is already complete are skipped when resume is
        enabled.

        Args:
            fold_index : Zero-based index of the fold.
            seed       : Seed of the unit, or None for an unseeded run.
            run_name   : Directory of the unit relative to the folds directory, or None to compose it from the fold and seed.

        Raises:
            ValueError : If the training family is unsupported.
        """
        dispatch = {
            "backbone"            : self._run_spatial,
            "dual"                : self._run_spatial,
            "jepa"                : self._run_spatial,
            "profile_autoencoder" : self._run_profile_autoencoder,
            "image_autoencoder"   : self._run_image_autoencoder,
        }

        training_type = FoldFamily.validate(self.config.training_type)
        run_directory = self.unit_directory(self.unit_run_name(fold_index, seed, run_name))
        subdir        = FoldFamily.inference_subdir(training_type)
        resume        = UnitResume(run_directory, enabled=self.config.resume)

        for split in self.config.inference_splits:
            if resume.skip_inference(split, subdir=subdir):
                continue

            dispatch[training_type](run_directory, split, fold_index)
