from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pipelines.backbone.training.loss       import Loss
from pipelines.shared.dataset.dataset_files import DatasetFiles
from tools.diagnostics.step_profiler        import StepProfiler
from tools.runtime.run_tag                  import RunTag


@dataclass
class FeedTarget:

    dataset     : Dataset
    model       : torch.nn.Module
    loss        : Loss
    model_name  : str
    sample_text : str


class FeedAdapter:

    def __init__(self, config, work_dir: Path, logger) -> None:
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name

    def _config_factory(self):
        from configuration.training                 import BackboneEntryConfig
        from pipelines.shared.config.config_factory import ConfigFactory

        entry       = BackboneEntryConfig(backbone_name=self.model_name, seed=self.config.seed)
        entry.paths = self.config.paths

        return ConfigFactory(entry)

    def _dataset(self, factory):
        from pipelines.backbone.dataset.pipeline  import DatasetPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        dataset_config           = factory.training_dataset_config()
        patch                    = dataset_config.patch
        patch.size, patch.stride = ModelBuilder.square_patch(self.model_name, patch.size, patch.stride)

        _train_loader, _val_loader, _test_loader, datasets, _normalizer = DatasetPipeline(dataset_config, self.work_dir, logger=self.logger, seed=self.config.seed).run()

        return datasets["train"], dataset_config

    def _model(self, dataset, dataset_config):
        from models                               import BACKBONE_CONFIG_REGISTRY, get_backbone
        from pipelines.shared.model.model_builder import ModelBuilder

        overrides = {"in_channels": dataset.input_channels, "out_channels": dataset.target_channels}
        overrides.update(ModelBuilder.image_size_override(self.model_name, dataset_config.patch.size))

        model, _ = get_backbone(self.model_name, config=BACKBONE_CONFIG_REGISTRY[self.model_name](), **overrides)
        return model

    def build(self) -> FeedTarget:
        factory                 = self._config_factory()
        dataset, dataset_config = self._dataset(factory)
        model                   = self._model(dataset, dataset_config)
        files                   = DatasetFiles.read(dataset_config.dataset_directory)

        sample = np.asarray(dataset[0][0])

        return FeedTarget(
            dataset     = dataset,
            model       = model,
            loss        = Loss(factory.config.loss, files.target_channels, self.logger),
            model_name  = self.model_name,
            sample_text = f"patches {tuple(sample.shape)}, {len(dataset):,} samples, {dataset.input_channels} input channels",
        )


class TrainingProfilePipeline:

    def __init__(self, entry, logger) -> None:
        self.entry  = entry
        self.logger = logger

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        output_dir = Path(self.entry.output_dir) / RunTag.now()
        feed       = FeedAdapter(self.entry, output_dir / "feed", self.logger).build()

        profiler = StepProfiler(
            feed            = feed,
            device          = self._device(),
            batch_size      = self.entry.batch_size,
            num_workers     = self.entry.num_workers,
            prefetch_factor = self.entry.prefetch_factor,
            use_amp         = self.entry.use_amp,
            warmup_steps    = self.entry.warmup_steps,
            active_steps    = self.entry.active_steps,
            record_shapes   = self.entry.record_shapes,
            profile_memory  = self.entry.profile_memory,
            with_stack      = self.entry.with_stack,
            logger          = self.logger,
        )

        return profiler.run(output_dir)
