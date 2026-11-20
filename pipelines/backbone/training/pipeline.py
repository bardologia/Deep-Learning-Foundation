
from __future__ import annotations

from pathlib import Path

from configuration.dataset                         import DatasetConfig
from configuration.training                        import BackboneTrainerConfig
from models                                        import get_backbone
from pipelines.backbone.dataset.pipeline           import DatasetPipeline
from pipelines.backbone.training.trainer           import Trainer
from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from pipelines.shared.model.model_builder          import ModelBuilder
from tools.runtime.reproducibility                 import Reproducibility


class TrainingPipeline:

    def __init__(
        self,
        trainer_config  : BackboneTrainerConfig,
        dataset_config  : DatasetConfig,
        backbone_name   : str,
        model_config    = None,
        seed            : int = 0,
        run_name        : str | None = None,
    ) -> None:
        self.trainer_config  = trainer_config
        self.dataset_config  = dataset_config
        self.backbone_name   = backbone_name
        self.model_config    = model_config
        self.seed            = seed
        self.run_name        = run_name

        patch                    = self.dataset_config.patch
        self.configured_patch    = (tuple(patch.size), tuple(patch.stride))
        patch.size, patch.stride = ModelBuilder.square_patch(self.backbone_name, patch.size, patch.stride)
        self.patch_size          = patch.size

        Reproducibility.seed_everything(self.seed)

    @property
    def effective_patch(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (tuple(self.patch_size), tuple(self.dataset_config.patch.stride))

    @property
    def patch_squared(self) -> bool:
        return self.effective_patch != self.configured_patch

    def _patch_geometry_label(self) -> str:
        size, stride = self.effective_patch
        effective    = f"patch {size} stride {stride}"

        if not self.patch_squared:
            return effective

        configured_size, configured_stride = self.configured_patch

        return f"configured patch {configured_size} stride {configured_stride} -> effective {effective} ({self.backbone_name} needs a square input)"

    def _patch_geometry_payload(self) -> dict:
        return {
            "configured_size"   : list(self.configured_patch[0]),
            "configured_stride" : list(self.configured_patch[1]),
            "size"              : list(self.effective_patch[0]),
            "stride"            : list(self.effective_patch[1]),
            "squared"           : self.patch_squared,
        }

    def _build_model(self, in_channels: int, out_channels: int):
        model, model_cfg = get_backbone(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        n_total     = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params      = f"{n_trainable:,} trainable" if n_total == n_trainable else f"{n_trainable:,} trainable ({n_total:,} total, {n_total - n_trainable:,} frozen)"

        self.logger.section("[Model Built]")
        self.logger.kv_table({
            "Architecture" : self.backbone_name,
            "Head"         : model_cfg.head,
            "In Channels"  : in_channels,
            "Out Channels" : out_channels,
            "Parameters"   : params,
        })
        return model, model_cfg

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(self.backbone_name, self.patch_size))

        return overrides

    def _make_trainer(self, model, model_cfg, target_channels, run_dir: Path, logger, emit_docs: bool = True, writer=None):
        return Trainer(
            model           = model,
            model_cfg       = model_cfg,
            target_channels = target_channels,
            config          = self.trainer_config,
            run_dir         = run_dir,
            logger          = logger,
            emit_docs       = emit_docs,
            writer          = writer,
        )

    def run(self, resolved_entry_config=None):
        self.run_metadata = TrainingRunMetadata(
            trainer_config = self.trainer_config,
            model_name     = self.backbone_name,
            base_logdir    = Path(self.trainer_config.io.logdir),
            run_name       = self.run_name,
        )
        with self.run_metadata:
            self.logger = self.run_metadata.logger

            if resolved_entry_config is not None:
                ResolvedEntryConfigWriter.save(resolved_entry_config, self.run_metadata.run_directory, self.logger)

            if self.patch_squared:
                self.logger.warning(f"{self.backbone_name} needs a square input: {self._patch_geometry_label()}; pixels per batch and patches per epoch differ from the configured geometry, and the resolved entry config keeps the configured values")

            self.run_metadata.save_dataset_fingerprint(self.dataset_config.dataset_directory, enabled=self.dataset_config.fingerprint_dataset)

            self.logger.section("[PyTorch Training Pipeline Execution]")
            self.logger.kv_table({
                "Seed"           : self.seed,
                "Patch geometry" : self._patch_geometry_label(),
            })

            self.dataset_pipeline = DatasetPipeline(self.dataset_config, self.run_metadata.run_directory, self.logger, seed=self.seed, resume=self.trainer_config.training.resume)

            train_loader, val_loader, test_loader, _datasets, _normalizer = self.dataset_pipeline.run()

            files            = self.dataset_pipeline.files
            model, model_cfg = self._build_model(in_channels=len(files.input_channels), out_channels=len(files.target_channels))

            self.run_metadata.save_trainer_config()
            self.run_metadata.save_model_config(model_cfg, self.backbone_name)
            self.run_metadata.save_run_summary(
                model_name      = self.backbone_name,
                input_channels  = files.input_channels,
                target_channels = files.target_channels,
                seed            = self.seed,
                patch_geometry  = self._patch_geometry_payload(),
            )
            self.logger.ok(f"Run metadata saved : {self.run_metadata.run_directory}")

            trainer = self._make_trainer(model, model_cfg, files.target_channels, self.run_metadata.run_directory, self.logger, writer=self.run_metadata.writer)

            results = trainer.train(train_loader, val_loader, test_loader)
            self.run_metadata.save_test_metrics(trainer.test_metrics)

        return results
