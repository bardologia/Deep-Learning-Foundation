"""Trainer and module wrapper for JEPA predictor training.

JepaModule chains an optional image autoencoder encoder in front of the backbone;
the trainer selects the embedding-match criterion when a profile autoencoder is
coupled, and the ordinary Gaussian parameter loss otherwise.
"""

from __future__ import annotations

import torch.nn as nn

from tools.training                         import BaseTrainer
from pipelines.jepa.training.coupling       import CouplingMode, TargetProvider
from pipelines.jepa.training.loss           import Loss as EmbeddingLoss
from pipelines.jepa.training.loss_probe     import JepaEmbeddingLossProbe
from pipelines.jepa.training.masking        import InputMasker, MaskSampler
from pipelines.backbone.training.loss       import Loss as ParameterLoss
from pipelines.backbone.training.loss_probe import LossScaleProbe
from pipelines.shared.training.curriculum   import CurriculumController


class JepaModule(nn.Module):
    """Backbone with optionally coupled image and profile autoencoders.

    Attributes:
        backbone            : Predictor network consuming either the dataset channels or the image-autoencoder feature map.
        profile_autoencoder : Autoencoder supplying the embedding target and decoder, or None when the backbone predicts Gaussian parameters directly.
        image_autoencoder   : Autoencoder encoding the input patch into features, or None when the backbone consumes the dataset channels.
    """

    def __init__(self, backbone: nn.Module, profile_autoencoder: nn.Module | None = None, image_autoencoder: nn.Module | None = None) -> None:
        """Registers the backbone and the coupled autoencoders as submodules.

        Args:
            backbone            : Predictor network.
            profile_autoencoder : Optional pretrained profile autoencoder.
            image_autoencoder   : Optional pretrained image autoencoder.
        """
        super().__init__()
        self.backbone            = backbone
        self.profile_autoencoder = profile_autoencoder
        self.image_autoencoder   = image_autoencoder

    def forward(self, images):
        """Runs the backbone, first encoding the input with the image autoencoder if present.

        Args:
            images : Input patch batch of shape (batch, channels, rows, cols).

        Returns:
            Backbone output of shape (batch, out_channels, rows, cols), holding
            either the profile embedding or the flattened Gaussian parameter slots.
        """
        if self.image_autoencoder is not None:
            images = self.image_autoencoder.encode_features(images, out_hw=images.shape[-2:])
        return self.backbone(images)


class Trainer(BaseTrainer):
    """Trainer for a JEPA module, wiring coupling modes, param groups and criterion.

    Attributes:
        backbone_cfg       : Backbone model configuration supplying its parameter groups.
        gaussian_cfg       : Gaussian mixture configuration of the run.
        norm_stats         : Output normalizer of the training dataset.
        profile_normalizer : Curve normalizer matching the coupled profile autoencoder.
        has_profile        : Whether a profile autoencoder is coupled.
        has_image          : Whether an image autoencoder is coupled.
        image_mode         : CouplingMode of the image autoencoder, or None.
        profile_mode       : CouplingMode of the profile autoencoder, or None.
    """

    stage_name    = "JEPA"
    section_title = "[JEPA Predictor Training]"

    def __init__(self, model: JepaModule, backbone_cfg, z_axis, config, run_dir, logger, norm_stats, profile_normalizer, writer=None):
        """Applies the coupling modes to the imported autoencoders and builds the trainer.

        Args:
            model              : JepaModule holding the backbone and coupled autoencoders.
            backbone_cfg       : Backbone model configuration.
            z_axis             : Z axis in metres.
            config             : JepaTrainerConfig with coupling modes, losses and optimization settings.
            run_dir            : Directory receiving checkpoints and training artifacts.
            logger             : Logger for the training banners and tables.
            norm_stats         : Output normalizer of the training dataset.
            profile_normalizer : Curve normalizer for the coupled profile autoencoder, or None.
            writer             : Optional TensorBoard writer.
        """
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

        self.curriculum = config.curriculum

        self.masking      = config.masking
        self.mask_sampler = None
        self.input_masker = None
        if self.masking.enabled:
            if not self.has_profile:
                raise ValueError("masking.enabled requires a coupled profile autoencoder: the mask restricts the embedding loss to hidden pixels, which the Gaussian parameter loss does not support. Couple a profile_autoencoder_run or disable masking.")

            if self.masking.loss_on not in ("masked", "all"):
                raise ValueError(f"Unknown masking.loss_on '{self.masking.loss_on}'. Available: masked, all")

            self.mask_sampler = MaskSampler(self.masking)
            self.input_masker = InputMasker(self.masking)

        super().__init__(model, config, run_dir, logger, z_axis, writer)

        if self.masking.enabled:
            self.logger.section("[Input Masking]")
            self.logger.kv_table({
                "Strategy"   : self.masking.strategy,
                "Mask ratio" : f"{self.masking.mask_ratio:g}" + (f" ({self.masking.n_blocks} blocks)" if self.masking.strategy == "block" else ""),
                "Fill"       : self.masking.fill + (f" (std={self.masking.fill_noise_std:g})" if self.masking.fill == "noise" else ""),
                "Loss on"    : self.masking.loss_on,
            })

        if self.curriculum.enabled and self.curriculum.swap_epoch >= self.epochs:
            raise ValueError(f"curriculum.swap_epoch={self.curriculum.swap_epoch} must be below training.epochs={self.epochs}; the main loss stage would never run")

        if self.curriculum.enabled:
            self.logger.section("[Curriculum]")
            self.logger.kv_table({
                "Initial stage"   : "curriculum.warmup_embedding" if self.has_profile else "curriculum.warmup_param",
                "Swap to main"    : f"epoch {self.curriculum.swap_epoch + 1} of {self.epochs} ({'embedding_loss' if self.has_profile else 'param_loss'})",
                "Reset LR"        : self.curriculum.reset_lr,
                "Reset warmup"    : self.curriculum.reset_warmup,
                "Reset optimizer" : self.curriculum.reset_optimizer,
            })

        self.curriculum_controller = CurriculumController(
            curriculum       = self.curriculum,
            complete_stage   = self._main_loss_stage(),
            criterion        = self.criterion,
            early_stopping   = self.early_stopping,
            lr_scheduler     = self.lr_scheduler,
            warmup           = self.warmup,
            optimizer        = self.optimizer,
            update_optimizer = self._update_optimizer,
            logger           = self.logger,
        )

    @staticmethod
    def validate_coupling(profile_mode: CouplingMode, target_provider: str, embedding_cfg, autoencoder) -> None:
        """Rejects coupling settings under which the embedding loss can collapse.

        Args:
            profile_mode    : CouplingMode of the profile autoencoder.
            target_provider : Configured target-provider kind.
            embedding_cfg   : Embedding-loss configuration.
            autoencoder     : Coupled profile autoencoder, inspected for an embedding LayerNorm.

        Raises:
            ValueError : If a live target is requested without a trainable autoencoder, or if a differentiable target branch (live provider, or a fine-tuned autoencoder with an embedding LayerNorm) is left without a curve reconstruction term to anchor it.
        """
        anchored = embedding_cfg.use_curve_recon and embedding_cfg.weight_curve_recon > 0.0

        if target_provider == "live" and not profile_mode.trainable:
            raise ValueError(f"target_provider 'live' requires a trainable profile autoencoder (profile_autoencoder_mode 'finetune'), but profile_autoencoder_mode is '{profile_mode.kind}'.")

        if target_provider == "live" and not anchored:
            raise ValueError("target_provider 'live' keeps the target branch differentiable, so the embedding-match loss can collapse to a constant embedding; enable embedding_loss.use_curve_recon with a positive weight_curve_recon to anchor it, or use 'stopgrad'.")

        if profile_mode.trainable and autoencoder.embedding_layernorm is not None and not anchored:
            raise ValueError("profile_autoencoder_mode 'finetune' with embedding_norm 'layernorm' trains the embedding LayerNorm affine on both loss branches, so the embedding-match loss can collapse by driving the affine scale to zero; enable embedding_loss.use_curve_recon with a positive weight_curve_recon to anchor it, or freeze the profile autoencoder.")

    def _build_param_groups(self):
        """Returns the backbone parameter groups plus any fine-tuned autoencoder groups."""
        param_groups = self.backbone_cfg.get_param_groups(self.model.backbone)
        if self.has_profile:
            param_groups += self.profile_mode.param_groups(self.model.profile_autoencoder, self.config.ae_finetune_lr, self.config.ae_finetune_wd)
        if self.has_image:
            param_groups += self.image_mode.param_groups(self.model.image_autoencoder, self.config.image_ae_finetune_lr, self.config.image_ae_finetune_wd)
        return param_groups

    def _main_loss_stage(self):
        """Returns the run's main loss stage of the active regime."""
        return self.config.embedding_loss if self.has_profile else self.config.param_loss

    def _initial_loss_stage(self):
        """Returns the loss stage training starts on, honoring the curriculum."""
        if not self.curriculum.enabled:
            return self._main_loss_stage()

        return self.curriculum.warmup_embedding if self.has_profile else self.curriculum.warmup_param

    def _build_criterion(self):
        """Returns the embedding-match loss when a profile AE is coupled, else the parameter loss."""
        stage = self._initial_loss_stage()

        if self.has_profile:
            target_provider = TargetProvider(self.config.target_provider)
            return EmbeddingLoss(self.model.profile_autoencoder, target_provider, stage, self.z_axis, self.norm_stats, self.gaussian_cfg.params_per_gaussian, self.profile_normalizer)

        return ParameterLoss(self.z_axis, self.logger, self.tracker, self.gaussian_cfg, stage, norm_stats=self.norm_stats, log_all_losses=self.config.training.log_all_losses)

    def _before_epoch(self, epoch: int) -> None:
        """Performs the curriculum loss swap if this epoch is the swap epoch."""
        self.curriculum_controller.maybe_swap(epoch)

    def _on_state_restored(self, state: dict) -> None:
        """Re-applies the main loss stage when a checkpoint resumes past the swap.

        Args:
            state : Restored trainer state, whose `epoch` is the last completed epoch.
        """
        if self.curriculum.enabled and state["epoch"] + 1 > self.curriculum.swap_epoch:
            self.criterion.set_curriculum(self._main_loss_stage(), context="re-applied on resume")
            self.logger.subsection("Resumed past the curriculum swap; main loss configuration re-applied.")

    def maybe_run_loss_probe(self, train_loader, probe_config=None) -> None:
        """Runs the loss-scale probe of the active regime before training.

        Args:
            train_loader : Loader supplying the probe batches.
            probe_config : Probe configuration; skipped when None or disabled.
        """
        if probe_config is None or not probe_config.enabled:
            return

        if self.has_profile:
            probe = JepaEmbeddingLossProbe(
                probe_cfg           = probe_config,
                embedding_cfg       = self._initial_loss_stage(),
                autoencoder         = self.model.profile_autoencoder,
                target_provider     = TargetProvider(self.config.target_provider),
                norm_stats          = self.norm_stats,
                params_per_gaussian = self.gaussian_cfg.params_per_gaussian,
                profile_normalizer  = self.profile_normalizer,
                logger              = self.logger,
            )
        else:
            probe = LossScaleProbe(
                probe_cfg    = probe_config,
                loss_cfg     = self._initial_loss_stage(),
                gaussian_cfg = self.gaussian_cfg,
                norm_stats   = self.norm_stats,
                logger       = self.logger,
            )

        probe.run(train_loader, self.model, self.device, self.z_axis)

    def _set_train_mode(self):
        """Puts the backbone and every fine-tuned autoencoder into train mode."""
        self.model.backbone.train()
        if self.has_profile and self.profile_mode.trainable:
            self.model.profile_autoencoder.train()
        if self.has_image and self.image_mode.trainable:
            self.model.image_autoencoder.train()

    def _compute_loss(self, batch):
        """Runs the module on a batch and evaluates the active criterion.

        With masking enabled, part of each input patch is hidden before the
        forward pass and the embedding loss is optionally restricted to the
        hidden pixels. Masking is a training-time augmentation only: in
        evaluation mode the batch is passed through unmasked and the embedding
        loss is taken over the full patch regardless of masking.loss_on, so the
        validation metric that drives early stopping and checkpoint selection is
        deterministic and comparable to an unmasked run.

        Args:
            batch : Tuple of the input patch batch of shape (batch, channels, rows, cols), the normalized ground-truth parameter batch, and, for the parameter loss, an optional per-pixel kz map in rad/m.

        Returns:
            The criterion's loss dict for this batch.
        """
        images    = batch[0].to(self.device)
        gt_params = batch[1].to(self.device)

        if self.has_profile:
            if not self.masking.enabled or not self.model.training:
                return self.criterion(self.model(images), gt_params)

            mask   = self.mask_sampler.sample(images.shape[0], images.shape[-2], images.shape[-1], images.device)
            pred   = self.model(self.input_masker.apply(images, mask))
            return self.criterion(pred, gt_params, mask if self.masking.loss_on == "masked" else None)

        return self.criterion(self.model(images), gt_params)
