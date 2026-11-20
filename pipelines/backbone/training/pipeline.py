
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configuration.dataset                         import DatasetConfig
from configuration.training                        import BackboneTrainerConfig, OverfitCheckConfig
from models                                        import get_backbone
from pipelines.backbone.dataset.datasets           import MultiRegionDataset
from pipelines.backbone.dataset.pipeline           import DatasetPipeline
from pipelines.backbone.training.loss_terms        import LossComponentCatalog
from pipelines.backbone.training.trainer           import Trainer
from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from pipelines.shared.model.model_builder          import ModelBuilder
from pipelines.shared.training.overfit_check       import OverfitCheck
from tools.data.gaussians                          import GaussianActivity, GaussianAxis, GaussianHead, GaussianSlotSorter
from tools.data.io                                 import FileIO
from tools.runtime.reproducibility                 import Reproducibility


class HeadPriorBias:

    PARAMS    = ("amplitude", "mu", "sigma")
    ROW_CHUNK = 256

    def __init__(self, dataset, n_gaussians: int, activity_threshold: float, logger) -> None:
        self.dataset            = dataset
        self.n_gaussians        = n_gaussians
        self.activity_threshold = activity_threshold
        self.logger             = logger

    @property
    def channel_names(self) -> list[str]:
        return [f"slot_{slot}_{param}" for slot in range(self.n_gaussians) for param in self.PARAMS]

    def _head_bias(self, model) -> torch.nn.Parameter:
        bias = model.output_head.bias

        return bias

    def _parts(self) -> list:
        return list(self.dataset.parts) if isinstance(self.dataset, MultiRegionDataset) else [self.dataset]

    def _accumulate(self, labels: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> None:
        n_params = len(self.PARAMS)

        for start in range(0, labels.shape[1], self.ROW_CHUNK):
            physical   = np.asarray(labels[:, start:start + self.ROW_CHUNK], dtype=np.float32)
            ordered    = GaussianSlotSorter.by_mean(physical, self.n_gaussians, self.activity_threshold)
            normalized = self.dataset.normalizer.normalize_output(ordered)

            for slot in range(self.n_gaussians):
                channels = slice(n_params * slot, n_params * (slot + 1))
                active   = GaussianActivity.is_active(ordered[n_params * slot], self.activity_threshold)

                sums[channels] += normalized[channels][:, active].sum(axis=1, dtype=np.float64)
                counts[slot]   += int(active.sum())

    def prior(self) -> tuple[np.ndarray, np.ndarray, int]:
        n_params   = len(self.PARAMS)
        n_channels = n_params * self.n_gaussians
        sums       = np.zeros(n_channels, dtype=np.float64)
        counts     = np.zeros(self.n_gaussians, dtype=np.int64)
        n_pixels   = 0

        for part in self._parts():
            labels = part.gt_selected

            self._accumulate(labels, sums, counts)
            n_pixels += int(labels.shape[1] * labels.shape[2])

        prior = sums / np.repeat(counts, n_params)

        return prior, counts, n_pixels

    def _log(self, prior: np.ndarray, default_bias: np.ndarray, counts: np.ndarray, n_pixels: int) -> None:
        rows = {
            "Activity threshold" : f"{self.activity_threshold:g}",
            "Training pixels"    : f"{n_pixels:,}",
        }

        for slot in range(self.n_gaussians):
            rows[f"Slot {slot} active"] = f"{int(counts[slot]):,} ({counts[slot] / n_pixels:.1%})"

        for name, value, default in zip(self.channel_names, prior, default_bias):
            rows[name] = f"{value:.4f} (default init {default:.4f})"

        self.logger.section("[Head Prior Bias]")
        self.logger.kv_table(rows)

    def apply(self, model) -> dict:
        bias                    = self._head_bias(model)
        prior, counts, n_pixels = self.prior()
        default_bias            = bias.detach().cpu().numpy().astype(np.float64)

        with torch.no_grad():
            bias.copy_(torch.as_tensor(prior, dtype=bias.dtype, device=bias.device))

        self._log(prior, default_bias, counts, n_pixels)

        return {
            "activity_threshold" : self.activity_threshold,
            "training_pixels"    : n_pixels,
            "active_pixels"      : [int(count) for count in counts],
            "channels"           : self.channel_names,
            "default_bias"       : default_bias.tolist(),
            "prior_bias"         : prior.tolist(),
        }


class TrainingPipeline:

    def __init__(
        self,
        trainer_config  : BackboneTrainerConfig,
        dataset_config  : DatasetConfig,
        backbone_name   : str,
        model_config    = None,
        seed            : int = 0,
        run_name        : str | None = None,
        overfit_check   : OverfitCheckConfig | None = None,
        prior_bias_init : bool = False,
    ) -> None:
        self.trainer_config  = trainer_config
        self.dataset_config  = dataset_config
        self.backbone_name   = backbone_name
        self.model_config    = model_config
        self.seed            = seed
        self.run_name        = run_name
        self.overfit_check   = overfit_check if overfit_check is not None else OverfitCheckConfig()
        self.prior_bias_init = prior_bias_init

        patch                    = self.dataset_config.patch
        self.configured_patch    = (tuple(patch.size), tuple(patch.stride))
        patch.size, patch.stride = ModelBuilder.square_patch(self.backbone_name, patch.size, patch.stride)
        self.patch_size          = patch.size

        Reproducibility.seed_everything(self.seed)

        self.dataset_config.n_gaussians = trainer_config.gaussian.n_default_gaussians

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

    def _build_dataset_pipeline(self, run_directory: Path, logger) -> DatasetPipeline:
        gaussian_cfg = self.trainer_config.gaussian

        dataset_pipeline = DatasetPipeline(
            config                 = self.dataset_config,
            training_run_directory = Path(run_directory),
            activity_threshold     = gaussian_cfg.activity_threshold,
            logger                 = logger,
            seed                   = self.seed,
            resume                 = self.trainer_config.training.resume,
        )

        self.dataset_config.z_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, dataset_pipeline.layout.profile_length)

        return dataset_pipeline

    def _run_overfit_check(self, train_dataset, in_channels: int, out_channels: int, z_axis) -> None:
        check = OverfitCheck(self.overfit_check, self.run_metadata.run_directory, self.logger)
        if not check.enabled:
            return

        check.announce()

        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        gate_stage                          = gate_trainer_config.curriculum.initial_stage
        gate_stage.use_active_normalization = False

        gate_trainer_config.curriculum.enabled  = False
        gate_trainer_config.curriculum.warmup   = gate_stage
        gate_trainer_config.curriculum.complete = gate_stage

        check.record("curriculum.enabled", False)
        check.record("gate_loss_stage", "warmup" if self.trainer_config.curriculum.enabled else "complete")
        check.record("gate_stage.use_active_normalization", False)

        base_config       = self.model_config if self.model_config is not None else ModelBuilder.config_from_registry(self.backbone_name, {})
        gate_model_config = check.sanitized_model_config(base_config)

        gate_model, gate_model_cfg = get_backbone(self.backbone_name, config=gate_model_config, **self._model_overrides(in_channels, out_channels))

        gate_trainer = Trainer(
            model      = gate_model,
            model_cfg  = gate_model_cfg,
            z_axis     = z_axis,
            config     = gate_trainer_config,
            run_dir    = check.work_directory,
            logger     = self.logger,
            norm_stats = train_dataset.normalizer,
            emit_docs  = False,
        )

        check.run(gate_trainer, train_dataset)

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

    def _apply_prior_bias(self, model, train_dataset) -> None:
        gaussian_cfg = self.trainer_config.gaussian
        prior_bias   = HeadPriorBias(train_dataset, gaussian_cfg.n_default_gaussians, gaussian_cfg.activity_threshold, self.logger)
        record       = prior_bias.apply(model)
        out_path     = self.run_metadata.metadata_directory / "head_prior_bias.json"

        FileIO.save_json(record, out_path)
        self.logger.info(f"Head prior bias saved: {out_path}")

    def _make_trainer(self, model, model_cfg, z_axis, norm_stats, run_dir: Path, logger, emit_docs: bool = True, writer=None):
        return Trainer(
            model      = model,
            model_cfg  = model_cfg,
            z_axis     = z_axis,
            config     = self.trainer_config,
            run_dir    = run_dir,
            logger     = logger,
            norm_stats = norm_stats,
            emit_docs  = emit_docs,
            writer     = writer,
        )

    def build_pretrain_trainer(self, work_dir: Path, logger):
        work_dir         = Path(work_dir)
        dataset_pipeline = self._build_dataset_pipeline(work_dir, logger)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        gaussian_cfg = self.trainer_config.gaussian
        dataset      = datasets["train"]
        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)
        z_axis       = np.asarray(self.dataset_config.z_axis, dtype=np.float32)

        model, model_cfg = get_backbone(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        trainer = self._make_trainer(model, model_cfg, z_axis, dataset.normalizer, work_dir, logger, emit_docs=False)
        trainer.criterion.set_curriculum(LossComponentCatalog.probe_union(self.trainer_config.curriculum), context="probe union of all curriculum stages")

        return trainer, dataset, model

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

            self.run_metadata.save_dataset_fingerprint(self.dataset_config.preprocessing_run_directory, self.dataset_config.parameters_path, enabled=self.dataset_config.fingerprint_dataset)

            self.logger.section("[PyTorch Training Pipeline Execution]")
            self.logger.kv_table({
                "Seed"           : self.seed,
                "Gaussian slots" : self.trainer_config.gaussian.n_default_gaussians,
                "Z axis"         : f"[{self.trainer_config.gaussian.x_min:g}, {self.trainer_config.gaussian.x_max:g}] m (sample count set by the dataset profile length)",
                "Patch geometry" : self._patch_geometry_label(),
            })

            self.dataset_pipeline = self._build_dataset_pipeline(self.run_metadata.run_directory, self.logger)

            gaussian_cfg = self.trainer_config.gaussian

            train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

            train_dataset = datasets["train"]
            in_channels   = train_dataset.input_channels
            n_gaussians   = gaussian_cfg.n_default_gaussians
            out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians)
            z_axis        = np.asarray(self.dataset_config.z_axis, dtype=np.float32)
            x_axis_length = int(z_axis.size)

            model, model_cfg = self._build_model(in_channels=in_channels, out_channels=out_channels)

            if self.prior_bias_init:
                self._apply_prior_bias(model, train_dataset)

            self.run_metadata.save_trainer_config()
            self.run_metadata.save_model_config(model_cfg, self.backbone_name)

            self.run_metadata.save_run_summary(
                model_name       = self.backbone_name,
                in_channels      = in_channels,
                out_channels     = out_channels,
                x_axis_length    = x_axis_length,
                n_gaussians      = n_gaussians,
                seed             = self.seed,
                patch_geometry   = self._patch_geometry_payload(),
            )
            self.logger.ok(f"Run metadata saved : {self.run_metadata.run_directory}")

            self._run_overfit_check(train_dataset, in_channels, out_channels, z_axis)

            trainer = self._make_trainer(model, model_cfg, z_axis, train_dataset.normalizer, self.run_metadata.run_directory, self.logger, writer=self.run_metadata.writer)

            results = trainer.train(train_loader, val_loader, test_loader)
            self.run_metadata.save_test_metrics(trainer.test_metrics)

        return results
