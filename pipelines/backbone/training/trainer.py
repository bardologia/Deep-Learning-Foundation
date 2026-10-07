"""Backbone training loop with curriculum swapping and per-slot diagnostics.

Extends the shared trainer with the composite Gaussian-profile loss, the
mid-training swap from the warmup to the complete loss stage, and the parameter
histogram, slot-vitals and reconstruction figure diagnostics.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from pipelines.backbone.training.loss        import Loss
from pipelines.backbone.training.loss_probe  import LossScaleProbe
from pipelines.shared.training.curriculum    import CurriculumController
from pipelines.backbone.training.docs        import TrainingDocs
from pipelines.backbone.training.diagnostics import ParamSampler, ReconstructionFigures, SlotVitals
from tools.training                          import BaseTrainer
from tools.training.aggregation              import ScalarsPage


class Trainer(BaseTrainer):
    """Trains a backbone on Gaussian profile parameters under a loss curriculum.

    Attributes:
        model_cfg             : Resolved model configuration, source of the optimizer param groups.
        gaussian_cfg          : Gaussian head configuration.
        curriculum            : Loss curriculum configuration.
        initial_loss_cfg      : Loss stage the run starts from.
        norm_stats            : Normalizer of the training dataset.
        emit_docs             : Whether the architecture documentation is written.
        param_sampler         : Collector of predicted-parameter histograms during validation.
        slot_vitals           : Per-slot activity and gradient diagnostics.
        recon_figures         : Reconstruction figure logger for the validation reference batch.
        docs                  : Architecture documentation emitter.
        curriculum_controller : Controller performing the loss-stage swap.
    """

    stage_name    = "Total"
    section_title = "[PyTorch Training Loop]"

    def __init__(self, model, model_cfg, z_axis, config, run_dir, logger, norm_stats=None, emit_docs=True, writer=None):
        """Builds the trainer, its diagnostics and the curriculum controller.

        Args:
            model      : Backbone to train.
            model_cfg  : Resolved model configuration.
            z_axis     : Z sample positions in metres, of shape (profile_length,).
            config     : Trainer configuration with training, curriculum, gaussian and geometry sections.
            run_dir    : Directory for checkpoints, logs and documentation.
            logger     : Logger for the training loop.
            norm_stats : Normalizer of the training dataset.
            emit_docs  : Whether the architecture documentation is written.
            writer     : Optional TensorBoard writer.

        Raises:
            ValueError : If the curriculum swap epoch is not below the epoch budget, which would leave the complete stage unused.
        """
        self.model_cfg        = model_cfg
        self.gaussian_cfg     = config.gaussian
        self.curriculum       = config.curriculum
        self.initial_loss_cfg = config.curriculum.initial_stage
        self.norm_stats       = norm_stats
        self.emit_docs        = emit_docs
        self.param_sampler    = ParamSampler(config.gaussian.params_per_gaussian)
        self.slot_vitals      = SlotVitals(config.gaussian.params_per_gaussian, grad_scale=float(config.training.gradient_accumulation_steps))

        super().__init__(model, config, run_dir, logger, z_axis, writer)

        self.recon_figures = ReconstructionFigures(self.tracker, self.device)

        if self.curriculum.enabled and self.curriculum.swap_epoch >= self.epochs:
            raise ValueError(f"curriculum.swap_epoch={self.curriculum.swap_epoch} must be below training.epochs={self.epochs}; the complete loss phase would never run")

        if self.curriculum.enabled:
            self.logger.section("[Curriculum]")
            self.logger.kv_table({
                "Initial stage"   : "curriculum.warmup",
                "Swap to complete": f"epoch {self.curriculum.swap_epoch + 1} of {self.epochs}",
                "Reset LR"        : self.curriculum.reset_lr,
                "Reset warmup"    : self.curriculum.reset_warmup,
                "Reset optimizer" : self.curriculum.reset_optimizer,
                "Inherit"         : f"{self.curriculum.inherit} (warmup adopts complete-stage values overridden only on curriculum.complete)",
            })

        self.docs = TrainingDocs(self.model, self.model_cfg, self.logger, self.run_dir, enabled=self.emit_docs)

        self.curriculum_controller = CurriculumController(
            curriculum       = self.curriculum,
            complete_stage   = self.curriculum.complete,
            criterion        = self.criterion,
            early_stopping   = self.early_stopping,
            lr_scheduler     = self.lr_scheduler,
            warmup           = self.warmup,
            optimizer        = self.optimizer,
            update_optimizer = self._update_optimizer,
            logger           = self.logger,
        )

    def _log_init_banner(self) -> None:
        """Logs the backend, device and log directory at the start of training."""
        self.logger.section("[Training Start]")
        self.logger.kv_table({
            "Backend"       : "PyTorch",
            "Device"        : self.device,
            "Log Directory" : self.config.io.logdir,
        })

    def _build_param_groups(self) -> list[dict]:
        """Returns the optimizer parameter groups declared by the model configuration."""
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self):
        """Builds the composite loss over the initial curriculum stage."""
        stage = "curriculum.warmup" if self.curriculum.enabled else "curriculum.complete (curriculum disabled)"
        self.logger.subsection(f"Loss built from {stage}")

        return Loss(self.z_axis, self.logger, self.tracker, self.gaussian_cfg, self.initial_loss_cfg, norm_stats=self.norm_stats, log_all_losses=self.config.training.log_all_losses, sampler=self.param_sampler, slot_vitals=self.slot_vitals)

    def _warn_nonfinite(self, tensor: torch.Tensor, name: str) -> None:
        """Warns when `tensor` holds any NaN or infinite entry.

        Args:
            tensor : Tensor to inspect, of any shape.
            name   : Name used in the warning message.
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.logger.warning(f"NaN or Inf detected in {name} at step {self.global_step}!")

    def _compute_loss(self, batch) -> dict:
        """Runs the model on one batch and evaluates the composite loss.

        Args:
            batch : Tuple of the input patch stack of shape (batch, in_channels, height, width), the target Gaussian parameters of shape (batch, n_gaussians * params_per_gaussian, height, width), and optionally the per-pixel kz map in rad/m of shape (batch, tracks, height, width).

        Returns:
            The loss dict produced by the criterion.
        """
        images    = batch[0].to(self.device)
        gt_params = batch[1].to(self.device) if len(batch) > 1 and batch[1] is not None else None

        if self.tracker.debug:
            self._warn_nonfinite(images, "input images")
            if gt_params is not None:
                self._warn_nonfinite(gt_params, "ground truth parameters")

        pred_params = self.model(images)
        if self.tracker.debug:
            self._warn_nonfinite(pred_params, "model predictions")

        if self.tracker.active and torch.is_tensor(pred_params) and pred_params.requires_grad:
            pred_params.register_hook(self.slot_vitals.observe_gradient)

        return self.criterion(pred_params, gt_params)

    def _scalars_page(self) -> ScalarsPage:
        """Adds the parameter errors, the occupancy statistics and the per slot vitals to the page."""
        return super()._scalars_page().parameter_errors().occupancy().slots()

    def _before_training(self, train_loader: DataLoader) -> None:
        """Emits the architecture documentation from a sample batch."""
        self.docs.emit(train_loader, self.device)

    def _before_epoch(self, epoch: int) -> None:
        """Performs the curriculum loss swap if this epoch is the swap epoch."""
        self.curriculum_controller.maybe_swap(epoch)

    def _on_state_restored(self, state: dict) -> None:
        """Re-applies the complete loss stage when a checkpoint resumes past the swap.

        Args:
            state : Restored trainer state, whose `epoch` is the last completed epoch.
        """
        if self.curriculum.enabled and state["epoch"] + 1 > self.curriculum.swap_epoch:
            self.criterion.set_curriculum(self.curriculum.complete, context="re-applied on resume")
            self.logger.subsection("Resumed past the curriculum swap; complete loss configuration re-applied.")

    def _before_validation(self, epoch: int, val_loader: DataLoader) -> None:
        """Arms the parameter sampler, slot vitals and reconstruction reference batch.

        Args:
            epoch      : Zero-based epoch being validated.
            val_loader : Validation loader the reference batch is taken from.
        """
        if not self.tracker.active:
            return

        self.param_sampler.begin(self.criterion.gaussian_cfg.activity_threshold)
        self.slot_vitals.begin(self.criterion.gaussian_cfg.activity_threshold)
        self.recon_figures.capture_reference(val_loader, self.criterion)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        """Logs the validation diagnostics and warns if early stopping preempted the swap.

        Args:
            val_loss : Validation loss of this epoch.
            epoch    : Zero-based epoch just validated.
        """
        self._log_val_diagnostics(epoch)

        if self.early_stopping.triggered and self.curriculum.enabled and epoch < self.curriculum.swap_epoch:
            self.logger.warning(f"Early stopping fired at epoch {epoch + 1}, before the curriculum swap at epoch {self.curriculum.swap_epoch + 1}; the complete loss phase never ran.")

        self._trial_callback(val_loss, epoch)

    def _log_val_diagnostics(self, epoch: int) -> None:
        """Logs parameter histograms, slot-vital scalars and EMA reconstruction figures.

        Args:
            epoch : Zero-based epoch the diagnostics belong to.
        """
        if not self.tracker.active:
            return

        for name, values in self.param_sampler.histograms().items():
            self.tracker.log_histogram(f"param_hist/{name}/val", values, epoch)

        for name, value in self.slot_vitals.scalars().items():
            self.tracker.log_scalar(f"{name}/val", value, epoch)

        self.param_sampler.end()
        self.slot_vitals.end()

        with self.ema.applied(self.model):
            self.recon_figures.log(self.model, self.criterion, epoch)

    def _log_train_epoch_extra(self, avg_loss: float, epoch: int) -> None:
        """Logs the curriculum phase indicator and the per-slot gradient scalars.

        Args:
            avg_loss : Mean training loss of the epoch.
            epoch    : Zero-based epoch just trained.
        """
        if self.curriculum.enabled:
            in_complete_phase = float(epoch >= self.curriculum.swap_epoch)
            self.tracker.log_scalar("controls/curriculum_phase", in_complete_phase, epoch)

        for name, value in self.slot_vitals.gradient_scalars().items():
            self.tracker.log_scalar(f"{name}/train", value, epoch)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        """Reports the epoch to a hyperparameter trial; a no-op outside tuning runs."""
        pass

    def maybe_run_loss_probe(self, train_loader, probe_config=None) -> None:
        """Runs the loss-scale probe before training when one is configured.

        Args:
            train_loader : Loader supplying the probe batches.
            probe_config : Probe configuration; skipped when None or disabled.
        """
        if probe_config is None or not probe_config.enabled:
            return

        probe = LossScaleProbe(
            probe_cfg    = probe_config,
            loss_cfg     = self.initial_loss_cfg,
            gaussian_cfg = self.gaussian_cfg,
            norm_stats   = self.norm_stats,
            logger       = self.logger,
        )
        probe.run(train_loader, self.model, self.device, self.z_axis)
