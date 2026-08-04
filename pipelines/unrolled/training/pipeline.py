from __future__ import annotations

import shutil

from pathlib import Path

import numpy as np

from configuration.sar.gaussian_config            import GaussianConfig
from configuration.training                       import UnrolledEntryConfig, UnrolledTrainerConfig
from models.unrolled                              import UNROLLED_MODEL_REGISTRY, get_unrolled
from pipelines.backbone.dataset.pipeline          import DatasetPipeline
from pipelines.shared.config.config_factory       import ConfigFactory
from pipelines.shared.config.config_persistence   import UnrolledModelConfigIO
from pipelines.shared.config.run_metadata         import TrainingRunMetadata
from pipelines.shared.training.overfit_check      import OverfitCheck
from pipelines.shared.training.seed_sweep         import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from pipelines.shared.training.training_runner    import EntryConfigTrainRunner
from pipelines.unrolled.training.trainer          import UnrolledTrainer
from tools.data.gaussians                         import GaussianAxis
from tools.data.io                                import FileIO
from tools.monitoring.logger                      import Logger
from tools.runtime.config_cli                     import ConfigCli
from tools.runtime.reproducibility                import Reproducibility


class UnrolledOverfitGate:
    def __init__(self, entry_config: UnrolledEntryConfig, run_directory: Path, logger) -> None:
        self.entry  = entry_config
        self.logger = logger
        self.check  = OverfitCheck(entry_config.overfit_check, run_directory, logger)

    def _sanitized_trainer_config(self, trainer_config: UnrolledTrainerConfig) -> UnrolledTrainerConfig:
        gate = self.check.sanitized_trainer_config(trainer_config)

        gate.measurement_noise_std = 0.0
        gate.memory.reserve_vram   = False

        self.check.record("measurement_noise_std", 0.0)
        self.check.record("memory.reserve_vram",   False)

        return gate

    def run(self, model_cfg, x_axis, trainer_config: UnrolledTrainerConfig, norm_stats, train_dataset) -> None:
        if not self.check.enabled:
            return

        self.check.announce()

        gate_trainer_config = self._sanitized_trainer_config(trainer_config)
        gate_model_cfg      = self.check.sanitized_model_config(model_cfg)
        gate_model          = UNROLLED_MODEL_REGISTRY[self.entry.model_name](gate_model_cfg)

        gate_trainer = UnrolledTrainer(gate_model, gate_model_cfg, x_axis, gate_trainer_config, self.check.work_directory, self.logger, norm_stats)

        try:
            self.check.run(gate_trainer, train_dataset)
        finally:
            shutil.rmtree(self.check.work_directory, ignore_errors=True)


class UnrolledTrainingPipeline:
    def __init__(self, config: UnrolledEntryConfig) -> None:
        self.config  = config
        self.factory = ConfigFactory(config)

        Reproducibility.seed_everything(config.seed)

    def _run_directory(self) -> Path:
        run_directory = Path(self.config.logdir) / TrainingRunMetadata.resolve_name(self.config.model_name, self.config.run_name)
        run_directory.mkdir(parents=True, exist_ok=True)

        return run_directory

    def _warn_inert_augmentation(self, logger) -> None:
        if self.config.augmentation.p_noise > 0.0:
            logger.warning("augmentation.p_noise > 0 has no effect for unrolled training: the input stack is discarded and measurements are synthesised from the ground truth; use measurement_noise_std for measurement-space noise.")

        if self.config.augmentation.p_flip_h > 0.0 or self.config.augmentation.p_flip_v > 0.0:
            logger.info("augmentation flips are inert for unrolled training: pixels are processed independently and the loss is a permutation-invariant per-pixel mean.")

    def _build_dataset_pipeline(self, run_directory: Path, logger, gaussian_cfg) -> tuple:
        dataset_config             = self.factory.training_dataset_config()
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        geometry = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        dataset_pipeline = DatasetPipeline(
            config                 = dataset_config,
            training_run_directory = run_directory,
            logger                 = logger,
            seed                   = self.config.seed,
            height_axis_convention = geometry.height_axis_convention,
            build_geometry_field   = True,
        )

        dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, dataset_pipeline.layout.profile_length)

        return dataset_pipeline, dataset_config

    def _build_model(self, logger):
        model, model_cfg = get_unrolled(self.config.model_name, **self.config.model_overrides)

        logger.section("[Model Built]")
        logger.kv_table({
            "Architecture"    : self.config.model_name,
            "Iterations"      : model_cfg.n_iterations,
            "Prox block"      : f"hidden={model_cfg.prox_hidden}, kernel={model_cfg.prox_kernel_size}, activation={model_cfg.activation}",
            "Init"            : f"step_init={model_cfg.step_init:g}, threshold_init={model_cfg.threshold_init:g}",
            "Param-group LRs" : f"steps lr={model_cfg.steps_lr:g} wd={model_cfg.steps_wd:g}  |  prox lr={model_cfg.prox_lr:g} wd={model_cfg.prox_wd:g}",
            "Model overrides" : self.config.model_overrides or "none",
            "Parameters"      : f"{sum(p.numel() for p in model.parameters()):,}",
        })

        return model, model_cfg

    def _trainer_config(self, run_directory: Path, gaussian_cfg) -> UnrolledTrainerConfig:
        return self.factory.unrolled_trainer_config(run_directory, gaussian_cfg.params_per_gaussian)

    def _save_training_summary(self, run_directory: Path, trainer: UnrolledTrainer, results: tuple) -> dict:
        train_losses, val_losses, best_val_loss = results

        summary = {
            "train_losses"  : [float(loss) for loss in train_losses],
            "val_losses"    : [float(loss) for loss in val_losses],
            "best_val_loss" : float(best_val_loss),
            "best_epoch"    : int(trainer.checkpoint.best_epoch) + 1,
            "test"          : {
                "loss"        : float(trainer.test_metrics["avg_loss"]),
                "num_batches" : int(trainer.test_metrics["num_batches"]),
                "peak_mae_m"  : float(trainer.peak_error_m["test"]),
            },
        }

        FileIO.save_json(summary, run_directory / "training_summary.json")

        return summary

    def build_pretrain_trainer(self, work_dir: Path, logger) -> tuple:
        work_dir     = Path(work_dir)
        gaussian_cfg = GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path)

        dataset_pipeline, dataset_config = self._build_dataset_pipeline(work_dir, logger, gaussian_cfg)
        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        dataset = datasets["train"]
        x_axis  = np.asarray(dataset_config.x_axis, dtype=np.float32)

        model, model_cfg = self._build_model(logger)

        trainer = UnrolledTrainer(model, model_cfg, x_axis, self._trainer_config(work_dir, gaussian_cfg), work_dir, logger, dataset.normalizer)

        return trainer, dataset, model

    def run(self) -> dict:
        run_directory = self._run_directory()
        logger        = Logger(log_dir=str(run_directory / "logs"), name="unrolled_training")

        logger.section("[Unrolled Training Pipeline Execution]")

        ConfigCli.save_resolved(self.config, run_directory / "docs" / "resolved_entry_config.json")
        logger.info(f"Resolved entry config saved: {run_directory / 'docs' / 'resolved_entry_config.json'}")

        self._warn_inert_augmentation(logger)

        gaussian_cfg = GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path)

        dataset_pipeline, dataset_config = self._build_dataset_pipeline(run_directory, logger, gaussian_cfg)
        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()

        model, model_cfg = self._build_model(logger)

        meta_directory = run_directory / "meta"
        meta_directory.mkdir(parents=True, exist_ok=True)
        UnrolledModelConfigIO.save(model_cfg, self.config.model_name, meta_directory)

        x_axis         = np.asarray(dataset_config.x_axis, dtype=np.float32)
        trainer_config = self._trainer_config(run_directory, gaussian_cfg)

        UnrolledOverfitGate(self.config, run_directory, logger).run(model_cfg, x_axis, trainer_config, datasets["train"].normalizer, datasets["train"])

        trainer = UnrolledTrainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = x_axis,
            config     = trainer_config,
            run_dir    = run_directory,
            logger     = logger,
            norm_stats = datasets["train"].normalizer,
        )

        try:
            results = trainer.train(train_loader, val_loader, test_loader)
            summary = self._save_training_summary(run_directory, trainer, results)
        finally:
            logger.close()

        return summary


class UnrolledSingleTrainRunner(EntryConfigTrainRunner):

    pipeline_class = UnrolledTrainingPipeline

    @property
    def label(self) -> str:
        return self.config.model_name

    def _resolve_run_name(self) -> str:
        return TrainingRunMetadata.resolve_name(self.config.model_name, self.config.run_name)


class UnrolledTrainingLauncher:
    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        import sys

        argv   = list(sys.argv[1:] if argv is None else argv)
        cli    = ConfigCli(UnrolledEntryConfig(), description="Train the unrolled physics network on synthesised per-pixel coherence measurements")
        config = cli.apply(argv)

        seeds = SeedSet.resolve(config.seeds, config.seed)
        if len(seeds) == 1:
            SeedSweepRunner(config, UnrolledSingleTrainRunner).run()
            return

        SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, UnrolledSingleTrainRunner, base_label=config.model_name).run()
