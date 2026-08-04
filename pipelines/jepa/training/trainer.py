from __future__ import annotations

import torch.nn as nn

from tools.training                   import BaseTrainer
from pipelines.jepa.training.coupling import CouplingMode, TargetProvider
from pipelines.jepa.training.loss     import Loss as EmbeddingLoss
from pipelines.backbone.training.loss import Loss as ParameterLoss


class JepaModule(nn.Module):
    def __init__(self, backbone: nn.Module, profile_autoencoder: nn.Module | None = None, image_autoencoder: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone            = backbone
        self.profile_autoencoder = profile_autoencoder
        self.image_autoencoder   = image_autoencoder

    def forward(self, images):
        if self.image_autoencoder is not None:
            images = self.image_autoencoder.encode_features(images, out_hw=images.shape[-2:])
        return self.backbone(images)


class Trainer(BaseTrainer):
    stage_name    = "JEPA"
    section_title = "[JEPA Predictor Training]"

    def __init__(self, model: JepaModule, backbone_cfg, x_axis, config, run_dir, logger, norm_stats, profile_normalizer):
        self.backbone_cfg       = backbone_cfg
        self.gaussian_cfg       = config.gaussian
        self.norm_stats         = norm_stats
        self.profile_normalizer = profile_normalizer

        self.has_profile = model.profile_autoencoder is not None
        self.has_image   = model.image_autoencoder   is not None

        self.image_mode = CouplingMode(config.image_autoencoder_mode, "image autoencoder") if self.has_image else None
        if self.has_image:
            self.image_mode.apply(model.image_autoencoder)

        self.profile_mode = CouplingMode(config.profile_autoencoder_mode, "profile autoencoder") if self.has_profile else None
        if self.has_profile:
            self.profile_mode.apply(model.profile_autoencoder)
            self.validate_coupling(self.profile_mode, config.target_provider, config.embedding_loss, model.profile_autoencoder)

        super().__init__(model, config, run_dir, logger, x_axis)

    @staticmethod
    def validate_coupling(profile_mode: CouplingMode, target_provider: str, embedding_cfg, autoencoder) -> None:
        anchored = embedding_cfg.use_curve_recon and embedding_cfg.weight_curve_recon > 0.0

        if target_provider == "live" and not profile_mode.trainable:
            raise ValueError(f"target_provider 'live' requires a trainable profile autoencoder (profile_autoencoder_mode 'finetune'), but profile_autoencoder_mode is '{profile_mode.kind}'.")

        if target_provider == "live" and not anchored:
            raise ValueError("target_provider 'live' keeps the target branch differentiable, so the embedding-match loss can collapse to a constant embedding; enable embedding_loss.use_curve_recon with a positive weight_curve_recon to anchor it, or use 'stopgrad'.")

        if profile_mode.trainable and autoencoder.embedding_layernorm is not None and not anchored:
            raise ValueError("profile_autoencoder_mode 'finetune' with embedding_norm 'layernorm' trains the embedding LayerNorm affine on both loss branches, so the embedding-match loss can collapse by driving the affine scale to zero; enable embedding_loss.use_curve_recon with a positive weight_curve_recon to anchor it, or freeze the profile autoencoder.")

    def _build_param_groups(self):
        param_groups = self.backbone_cfg.get_param_groups(self.model.backbone)
        if self.has_profile:
            param_groups += self.profile_mode.param_groups(self.model.profile_autoencoder, self.config.ae_finetune_lr, self.config.ae_finetune_wd)
        if self.has_image:
            param_groups += self.image_mode.param_groups(self.model.image_autoencoder, self.config.image_ae_finetune_lr, self.config.image_ae_finetune_wd)
        return param_groups

    def _build_criterion(self):
        if self.has_profile:
            target_provider = TargetProvider(self.config.target_provider)
            return EmbeddingLoss(self.model.profile_autoencoder, target_provider, self.config.embedding_loss, self.x_axis, self.norm_stats, self.gaussian_cfg.params_per_gaussian, self.profile_normalizer)

        return ParameterLoss(self.x_axis, self.logger, self.tracker, self.gaussian_cfg, self.config.param_loss, norm_stats=self.norm_stats, geometry_cfg=self.config.geometry, log_all_losses=self.config.training.log_all_losses)

    def _set_train_mode(self):
        self.model.backbone.train()
        if self.has_profile and self.profile_mode.trainable:
            self.model.profile_autoencoder.train()
        if self.has_image and self.image_mode.trainable:
            self.model.image_autoencoder.train()

    def _compute_loss(self, batch):
        images    = batch[0].to(self.device)
        gt_params = batch[1].to(self.device)
        pred      = self.model(images)

        if self.has_profile:
            return self.criterion(pred, gt_params)

        kz_map = batch[2].to(self.device) if len(batch) > 2 and batch[2] is not None else None
        return self.criterion(pred, gt_params, kz_map)
