from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import numpy as np
import torch

from configuration.training                              import JepaTrainerConfig
from models                                              import get_backbone
from models.profile_autoencoder                          import get_profile_autoencoder
from models.image_autoencoder                            import get_image_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.shared.config.config_factory              import ConfigFactory
from pipelines.shared.config.run_metadata                import TrainingRunMetadata
from pipelines.backbone.dataset.pipeline                 import BackboneDatasetPreparation
from pipelines.shared.model.model_builder                import ModelBuilder
from pipelines.shared.training.overfit_check             import OverfitCheck
from pipelines.shared.training.run_naming                import RunNaming
from pipelines.shared.training.training_runner           import EntryConfigTrainRunner
from pipelines.jepa.training.trainer                     import JepaModule, Trainer
from pipelines.shared.config.config_persistence          import ProfileAutoencoderConfigIO, ImageAutoencoderConfigIO
from tools.runtime.reproducibility                       import Reproducibility
from tools.runtime.run_tag                               import RunTag



class TrainingPipeline:
    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        profile_dir = self._resolve_ae_run(entry_config.profile_autoencoder_logdir, entry_config.profile_autoencoder_run, "profile")
        image_dir   = self._resolve_ae_run(entry_config.image_autoencoder_logdir,   entry_config.image_autoencoder_run,   "image")

        if profile_dir is None and image_dir is None:
            raise ValueError("JEPA requires at least one of profile_autoencoder_run or image_autoencoder_run; with neither, train the plain backbone via main/training/train_backbone.py.")

        self.profile_autoencoder_meta = None
        self.autoencoder_cfg          = None
        self.ae_model_name            = None
        profile_checkpoint            = None
        if profile_dir is not None:
            self.profile_autoencoder_meta            = profile_dir / "meta"
            self.autoencoder_cfg, self.ae_model_name = ProfileAutoencoderConfigIO.load(self.profile_autoencoder_meta)
            profile_checkpoint                       = str(profile_dir / "best_model.pt")

        self.image_autoencoder_meta = None
        self.image_ae_cfg           = None
        self.image_ae_model_name    = None
        image_checkpoint            = None
        if image_dir is not None:
            self.image_autoencoder_meta                 = image_dir / "meta"
            self.image_ae_cfg, self.image_ae_model_name = ImageAutoencoderConfigIO.load(self.image_autoencoder_meta)
            image_checkpoint                            = str(image_dir / "best_model.pt")

        self.trainer_config = JepaTrainerConfig(
            gaussian                       = base.gaussian,
            autoencoder                    = self.autoencoder_cfg,
            embedding_loss                 = entry_config.embedding_loss,
            profile_autoencoder_mode       = entry_config.profile_autoencoder_mode,
            target_provider                = entry_config.target_provider,
            profile_autoencoder_checkpoint = profile_checkpoint,
            ae_finetune_lr                 = entry_config.ae_finetune_lr,
            ae_finetune_wd                 = entry_config.ae_finetune_wd,
            image_autoencoder              = self.image_ae_cfg,
            image_autoencoder_mode         = entry_config.image_autoencoder_mode,
            image_autoencoder_checkpoint   = image_checkpoint,
            image_ae_finetune_lr           = entry_config.image_ae_finetune_lr,
            image_ae_finetune_wd           = entry_config.image_ae_finetune_wd,
            param_loss                     = entry_config.param_loss,
        )
        self.trainer_config.inherit_shared_from(base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

        self.backbone_name = entry_config.backbone_name

    @staticmethod
    def _resolve_ae_run(logdir, run, label):
        if not run:
            return None
        directory = Path(logdir) / run
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} autoencoder run '{run}' not found under {logdir}")
        return directory

    def _physics_geometry_active(self) -> bool:
        if self.autoencoder_cfg is not None:
            return False

        cfg = self.trainer_config.param_loss
        return cfg.use_coherence_resyn or cfg.use_covariance_match or cfg.use_capon_cycle

    def _prepare_datasets(self, run_directory, logger, seed):
        return BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_directory, logger, seed, build_geometry_field=self._physics_geometry_active(), height_axis_convention=self.trainer_config.geometry.height_axis_convention).run()

    def _gaussian_out_channels(self) -> int:
        g = self.trainer_config.gaussian
        return g.params_per_gaussian * g.n_default_gaussians

    def _run_overfit_check(self, run_meta, logger, datasets, x_len: int, x_axis) -> None:
        check = OverfitCheck(self.entry.overfit_check, run_meta.run_directory, logger)
        if not check.enabled:
            return

        check.announce()

        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        if self.autoencoder_cfg is None:
            gate_trainer_config.param_loss.use_active_normalization = False
            check.record("param_loss.use_active_normalization", False)

        base_backbone_config = ModelBuilder.config_from_registry(self.backbone_name, self.entry.model_overrides, head=self.entry.backbone_head)
        gate_backbone_config = check.sanitized_model_config(base_backbone_config)

        gate_module, gate_backbone_cfg = self._build_module(datasets, x_len, x_axis, logger, backbone_config=gate_backbone_config)
        profile_normalizer             = self._profile_normalizer(run_meta.metadata_directory, logger)

        gate_trainer = Trainer(gate_module, gate_backbone_cfg, x_axis, gate_trainer_config, check.work_directory, logger, datasets["train"].normalizer, profile_normalizer)

        check.run(gate_trainer, datasets["train"])

    def _build_module(self, datasets, x_len: int, x_axis, logger, backbone_config=None):
        dataset_in_channels = datasets["train"].input_channels

        image_autoencoder = None
        backbone_in       = dataset_in_channels
        if self.image_ae_cfg is not None:
            if self.image_ae_cfg.in_channels != dataset_in_channels:
                raise ValueError(f"Image autoencoder was trained with in_channels={self.image_ae_cfg.in_channels} but the JEPA dataset produces {dataset_in_channels} input channels; the input representation and secondaries must match the image autoencoder run.")
            image_autoencoder = self._load_image_autoencoder()
            backbone_in       = self.image_ae_cfg.embedding_dim

        profile_autoencoder = None
        if self.autoencoder_cfg is not None:
            if self.autoencoder_cfg.profile_length != x_len:
                raise ValueError(f"Profile autoencoder was trained with profile_length={self.autoencoder_cfg.profile_length} but the JEPA dataset produces curves of length {x_len}; retrain the profile autoencoder on this dataset's elevation axis before coupling it.")

            backbone_out        = self.autoencoder_cfg.embedding_dim
            profile_autoencoder = self._load_profile_autoencoder(x_axis)
        else:
            backbone_out = self._gaussian_out_channels()

        self.backbone_in_channels  = backbone_in
        self.backbone_out_channels = backbone_out

        backbone, backbone_cfg = self._build_backbone(backbone_in, backbone_out, self.dataset_config.patch.size, config=backbone_config)
        module                 = JepaModule(backbone, profile_autoencoder=profile_autoencoder, image_autoencoder=image_autoencoder)

        self._log_module(logger, module, backbone, dataset_in_channels, backbone_in, backbone_out)
        return module, backbone_cfg

    def _log_module(self, logger, module, backbone, dataset_in: int, backbone_in: int, backbone_out: int) -> None:
        backbone_params  = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        info = {
            "Backbone"             : self.backbone_name,
            "Dataset Channels"     : dataset_in,
            "Backbone In"          : backbone_in,
            "Backbone Out"         : backbone_out,
            "Backbone Parameters"  : f"{backbone_params:,}",
            "Trainable Parameters" : f"{trainable_params:,}",
        }

        if self.autoencoder_cfg is not None:
            info["Target Provider"] = self.trainer_config.target_provider
            info["Embedding Loss"]  = self._embedding_loss_label(self.trainer_config.embedding_loss)
            info["Param Loss"]      = "inactive (profile AE coupled; the run-name loss tokens still describe param_loss)"
        else:
            info["Target Provider"] = "inactive (no profile AE)"
            info["Embedding Loss"]  = "inactive (no profile AE; the backbone trains on param_loss, banner below)"

        if self.image_ae_cfg is not None:
            info["Image AE"]            = f"{self.image_ae_model_name}  (embedding_dim={self.image_ae_cfg.embedding_dim}, mode={self.trainer_config.image_autoencoder_mode})"
            info["Image AE Checkpoint"] = self.trainer_config.image_autoencoder_checkpoint
        else:
            info["Image AE"] = "disabled (backbone consumes dataset channels directly)"

        if self.autoencoder_cfg is not None:
            info["Profile AE"]            = f"{self.ae_model_name}  (embedding_dim={self.autoencoder_cfg.embedding_dim}, mode={self.trainer_config.profile_autoencoder_mode})"
            info["Profile AE Checkpoint"] = self.trainer_config.profile_autoencoder_checkpoint
        else:
            info["Profile AE"] = "disabled (backbone predicts Gaussian parameters directly)"

        logger.section("[JEPA Module Built]")
        logger.kv_table(info)

    @staticmethod
    def _embedding_loss_label(emb) -> str:
        terms = []
        if emb.use_embedding_mse:
            terms.append(f"mse w={emb.weight_embedding_mse:g}")
        if emb.use_embedding_cosine:
            terms.append(f"cosine w={emb.weight_embedding_cosine:g}")
        if emb.use_embedding_smoothl1:
            terms.append(f"smoothl1 w={emb.weight_embedding_smoothl1:g} beta={emb.smoothl1_beta:g}")
        if emb.use_curve_recon:
            terms.append(f"curve_recon({emb.curve_kind}) w={emb.weight_curve_recon:g}")

        return "  |  ".join(terms) if terms else "no terms enabled"

    def _build_backbone(self, in_channels: int, out_channels: int, patch_size, config=None):
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(self.backbone_name, patch_size))
        if config is None:
            overrides["head"] = self.entry.backbone_head
            for k, v in self.entry.model_overrides.items():
                overrides[k] = v
        return get_backbone(self.backbone_name, config=config, **overrides)

    def _load_profile_autoencoder(self, x_axis):
        autoencoder, _ = get_profile_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        ckpt_path      = self.trainer_config.profile_autoencoder_checkpoint
        self.validate_checkpoint(ckpt_path, "profile")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._validate_profile_axis(np.asarray(ckpt["x_axis"], dtype=np.float32), np.asarray(x_axis, dtype=np.float32), ckpt_path)
        autoencoder.load_state_dict(ckpt["params"])
        return autoencoder

    @staticmethod
    def _validate_profile_axis(ae_axis, jepa_axis, ckpt_path) -> None:
        if ae_axis.shape != jepa_axis.shape or not np.allclose(ae_axis, jepa_axis, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Profile autoencoder checkpoint '{ckpt_path}' was trained on elevation axis [{ae_axis[0]:.3f}, {ae_axis[-1]:.3f}] with {ae_axis.size} bins, but the JEPA dataset produces axis [{jepa_axis[0]:.3f}, {jepa_axis[-1]:.3f}] with {jepa_axis.size} bins; retrain the profile autoencoder on this dataset's elevation axis before coupling it.")

    def _load_image_autoencoder(self):
        autoencoder, _ = get_image_autoencoder(self.image_ae_model_name, self.image_ae_cfg)
        ckpt_path      = self.trainer_config.image_autoencoder_checkpoint
        self.validate_checkpoint(ckpt_path, "image")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        autoencoder.load_state_dict(ckpt["params"])
        return autoencoder

    @staticmethod
    def validate_checkpoint(ckpt_path, label: str) -> None:
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"{label} autoencoder checkpoint '{ckpt_path}' does not exist; expected 'best_model.pt' under the selected {label} autoencoder run directory.")

    def _save_metadata(self, run_meta, backbone_cfg, x_len: int) -> None:
        gaussian_cfg = self.trainer_config.gaussian

        run_meta.save_trainer_config()
        run_meta.save_model_config(backbone_cfg, self.backbone_name)

        if self.autoencoder_cfg is not None:
            ProfileAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        if self.image_ae_cfg is not None:
            ImageAutoencoderConfigIO.save(self.image_ae_cfg, self.image_ae_model_name, run_meta.metadata_directory)

        run_meta.save_run_summary(self.backbone_name, in_channels=self.backbone_in_channels, out_channels=self.backbone_out_channels, x_axis_length=x_len, n_gaussians=gaussian_cfg.n_default_gaussians, seed=self.entry.seed)

    def _profile_normalizer(self, metadata_directory, logger):
        if self.autoencoder_cfg is None:
            return None

        stats = ProfileStats.load(self.profile_autoencoder_meta, logger=logger)
        stats.save(metadata_directory)

        return ProfileNormalizer(stats)

    def _make_trainer(self, model, backbone_cfg, x_axis, run_dir, logger, norm_stats, profile_normalizer):
        return Trainer(model, backbone_cfg, x_axis, self.trainer_config, run_dir, logger, norm_stats, profile_normalizer)

    def _train(self, run_meta, logger, model, backbone_cfg, x_axis, datasets, loaders, profile_normalizer):
        train_loader, val_loader, test_loader = loaders
        norm_stats                            = datasets["train"].normalizer

        trainer = self._make_trainer(model, backbone_cfg, x_axis, run_meta.run_directory, logger, norm_stats, profile_normalizer)
        try:
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            run_meta.close()
        return results, run_meta.run_directory

    def build_pretrain_trainer(self, work_dir, logger):
        work_dir = Path(work_dir)

        loaders, datasets, x_axis, x_len = self._prepare_datasets(work_dir, logger, self.entry.seed)

        model, backbone_cfg = self._build_module(datasets, x_len, x_axis, logger)
        profile_normalizer  = self._profile_normalizer(work_dir / "meta", logger)

        trainer = self._make_trainer(model, backbone_cfg, x_axis, work_dir, logger, datasets["train"].normalizer, profile_normalizer)

        return trainer, datasets["train"], model

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, self.backbone_name, Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        loaders, datasets, x_axis, x_len = self._prepare_datasets(run_meta.run_directory, logger, self.entry.seed)

        model, backbone_cfg = self._build_module(datasets, x_len, x_axis, logger)

        self._save_metadata(run_meta, backbone_cfg, x_len)

        try:
            self._run_overfit_check(run_meta, logger, datasets, x_len, x_axis)
        except BaseException:
            run_meta.close()
            raise

        profile_normalizer = self._profile_normalizer(run_meta.metadata_directory, logger)

        return self._train(run_meta, logger, model, backbone_cfg, x_axis, datasets, loaders, profile_normalizer)


class SingleTrainRunner(EntryConfigTrainRunner):
    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        n_gaussians = ConfigFactory(self.config).gaussian_config().n_default_gaussians
        return RunNaming.tag(self.config.backbone_name, self.config.backbone_head, self.config.param_loss, n_gaussians, self.config.augmentation)

    def _resolve_run_name(self) -> str:
        return RunNaming.compose(self.label, self.config.run_name or RunTag.now())

    def _run_inference(self, run_directory) -> None:
        from pipelines.backbone.inference.pipeline           import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)
        components       = InferenceComponentsResolver.for_run(Path(run_directory))
        InferencePipeline(inference_config, components=components).run()

    def _skipped_result(self):
        return None, self.run_directory

    def run(self):
        results, run_directory = super().run()

        if self.config.infer_after and not self.unit_resume.skip_inference(self.config.inference.split):
            self._run_inference(run_directory)

        return results
