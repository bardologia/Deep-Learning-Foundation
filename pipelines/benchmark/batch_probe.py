from __future__ import annotations

import gc
import traceback
from pathlib import Path

import numpy as np
import torch

from configuration.benchmark                 import BenchmarkConfig
from models                                  import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.shared.model.model_builder    import ModelBuilder
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.training.loss_terms  import LossComponentCatalog
from pipelines.backbone.training.pipeline    import TrainingPipeline
from pipelines.backbone.training.trainer     import Trainer
from pipelines.shared.config.config_factory  import ConfigFactory
from tools.data.gaussians                    import GaussianAxis, GaussianHead
from tools.monitoring.logger                 import Logger
from tools.runtime.reproducibility           import Reproducibility
from tools.training.pretraining.batch_finder import BatchSizeFinder, TrainStepMemoryProbe


class MaxBatchProbe:
    CONTEXT_WARN_GB = 1.5

    def __init__(self, config: BenchmarkConfig, model_name: str, overrides: dict, work_dir: Path) -> None:
        self.config     = config
        self.model_name = model_name
        self.overrides  = overrides

        self.budget_gb     = config.max_batch.vram_budget_gb
        self.ceiling       = config.max_batch.max_batch
        self.measure_steps = config.max_batch.measure_steps
        self.seed          = config.max_batch.seed

        self.device     = torch.device("cuda")
        self.context_gb = 0.0
        self.work_dir   = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir=str(self.work_dir / "logs"), name="max_batch", level="INFO")

    def _measure_context(self) -> float:
        return TrainStepMemoryProbe.measure_context(self.device)

    def _build_context(self):
        factory        = ConfigFactory(self.config)
        dataset_config = factory.training_dataset_config()
        trainer_config = factory.training_trainer_config(logdir=self.work_dir)

        dataset_config.input_config = self.config.input
        trainer_config.geometry     = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=factory._secondary_labels())
        trainer_config.curriculum   = LossComponentCatalog.combined_curriculum(self.config.sweep_loss_components, base=self.config.loss)

        gaussian_cfg               = trainer_config.gaussian
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(config=dataset_config, training_run_directory=self.work_dir, logger=self.logger, seed=self.seed, height_axis_convention=trainer_config.geometry.height_axis_convention, build_geometry_field=TrainingPipeline.physics_geometry_active(trainer_config))

        profile_length        = dataset_pipeline.layout.profile_length
        dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return trainer_config, dataset_config, datasets["train"], gaussian_cfg

    def _build_model(self, dataset_config, dataset, gaussian_cfg):
        name, head   = ModelBuilder.split_key(self.model_name)
        model_config = BACKBONE_CONFIG_REGISTRY[name]()
        model_config.head = head

        for attribute, value in self.overrides.items():
            setattr(model_config, attribute, value)

        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(name, dataset_config.patch.size))

        return get_backbone(name, config=model_config, **overrides)

    def _build_trainer(self, trainer_config, dataset_config, model, model_cfg, dataset):
        trainer = Trainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = dataset_config.x_axis,
            config     = trainer_config,
            run_dir    = self.work_dir,
            logger     = self.logger,
            norm_stats = dataset.normalizer,
            emit_docs  = False,
        )

        trainer.criterion.set_curriculum(trainer_config.curriculum.complete, context="VRAM probe on the complete stage")
        trainer.model.train()

        return trainer

    def _candidates(self) -> list[int]:
        return BatchSizeFinder.candidates(self.ceiling)

    def _trial(self, trainer, dataset, batch_size: int) -> float:
        return TrainStepMemoryProbe(trainer, dataset, self.measure_steps, self.device, self.context_gb)(batch_size)

    def _release(self, trainer) -> None:
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def run(self) -> dict:
        result = {
            "model"      : self.model_name,
            "status"     : None,
            "batch_size" : None,
            "peak_gb"    : None,
            "budget_gb"  : self.budget_gb,
            "ceiling"    : self.ceiling,
            "context_gb" : None,
            "trials"     : [],
            "error"      : None,
        }

        try:
            Reproducibility.seed_everything(self.seed)

            self.context_gb      = self._measure_context()
            result["context_gb"] = self.context_gb
            self.logger.subsection(f"CUDA context: {self.context_gb:.2f} GB")
            if self.context_gb > self.CONTEXT_WARN_GB:
                self.logger.warning(f"Context {self.context_gb:.2f} GB exceeds a bare CUDA context; other processes are using this GPU and shrink the measured budget")

            trainer_config, dataset_config, dataset, gaussian_cfg = self._build_context()

            if dataset.input_channels != self.config.size_match.in_channels:
                raise RuntimeError(f"size_match.in_channels={self.config.size_match.in_channels} but the dataset provides {dataset.input_channels} input channels; capacity matching counted the wrong width — fix size_match.in_channels and re-run size matching without resume")

            ceiling = min(self.ceiling, len(dataset))
            if ceiling < self.ceiling:
                self.logger.warning(f"Ceiling lowered from {self.ceiling} to {ceiling}: the train split holds {len(dataset)} samples")

            model, model_cfg = self._build_model(dataset_config, dataset, gaussian_cfg)
            trainer          = self._build_trainer(trainer_config, dataset_config, model, model_cfg, dataset)

            finder = BatchSizeFinder(
                trial_step = lambda batch_size: self._trial(trainer, dataset, batch_size),
                budget_gb  = self.budget_gb,
                ceiling    = ceiling,
                device     = self.device,
                logger     = self.logger,
                model_name = self.model_name,
                context_gb = self.context_gb,
                on_oom     = lambda: self._release(trainer),
            )

            result.update(finder.run())

            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        self.logger.close()

        return result
