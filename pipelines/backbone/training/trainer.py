
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

    stage_name    = "Total"
    section_title = "[PyTorch Training Loop]"

    def __init__(self, model, model_cfg, z_axis, config, run_dir, logger, norm_stats=None, emit_docs=True, writer=None):
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
            Loss.active_weights(self.curriculum.complete)

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
        self.logger.section("[Training Start]")
        self.logger.kv_table({
            "Backend"       : "PyTorch",
            "Device"        : self.device,
            "Log Directory" : self.config.io.logdir,
        })

    def _build_param_groups(self) -> list[dict]:
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self):
        stage = "curriculum.warmup" if self.curriculum.enabled else "curriculum.complete (curriculum disabled)"
        self.logger.subsection(f"Loss built from {stage}")

        return Loss(self.z_axis, self.logger, self.tracker, self.gaussian_cfg, self.initial_loss_cfg, norm_stats=self.norm_stats, log_all_losses=self.config.training.log_all_losses, sampler=self.param_sampler, slot_vitals=self.slot_vitals)

    def _warn_nonfinite(self, tensor: torch.Tensor, name: str) -> None:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.logger.warning(f"NaN or Inf detected in {name} at step {self.global_step}!")

    def _compute_loss(self, batch) -> dict:
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

    def _nonfinite_detail(self, loss_dict: dict, batch) -> str:
        gt_params = batch[1] if len(batch) > 1 and batch[1] is not None else None
        named     = (("inputs", batch[0]), ("targets", gt_params))
        bad_data  = [name for name, tensor in named if tensor is not None and not torch.isfinite(tensor).all()]
        verdict   = f"non-finite {' and '.join(bad_data)} in the batch" if bad_data else "inputs and targets finite, so the model or the loss produced it"

        return f"{super()._nonfinite_detail(loss_dict, batch)}; {verdict}"

    def _scalars_page(self) -> ScalarsPage:
        return super()._scalars_page().parameter_errors().occupancy().slots()

    def _before_training(self, train_loader: DataLoader) -> None:
        self.docs.emit(train_loader, self.device)

    def _before_epoch(self, epoch: int) -> None:
        self.curriculum_controller.maybe_swap(epoch)

    def _on_state_restored(self, state: dict) -> None:
        if self.curriculum.enabled and state["epoch"] + 1 > self.curriculum.swap_epoch:
            self.criterion.set_curriculum(self.curriculum.complete, context="re-applied on resume")
            self.logger.subsection("Resumed past the curriculum swap; complete loss configuration re-applied.")

    def _before_validation(self, epoch: int, val_loader: DataLoader) -> None:
        if not self.tracker.active:
            return

        self.param_sampler.begin(self.criterion.gaussian_cfg.activity_threshold)
        self.slot_vitals.begin(self.criterion.gaussian_cfg.activity_threshold)
        self.recon_figures.capture_reference(val_loader, self.criterion)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        self._log_val_diagnostics(epoch)

        if self.early_stopping.triggered and self.curriculum.enabled and epoch < self.curriculum.swap_epoch:
            self.logger.warning(f"Early stopping fired at epoch {epoch + 1}, before the curriculum swap at epoch {self.curriculum.swap_epoch + 1}; the complete loss phase never ran.")

        self._trial_callback(val_loss, epoch)

    def _log_val_diagnostics(self, epoch: int) -> None:
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
        if self.curriculum.enabled:
            in_complete_phase = float(epoch >= self.curriculum.swap_epoch)
            self.tracker.log_scalar("controls/curriculum_phase", in_complete_phase, epoch)

        for name, value in self.slot_vitals.gradient_scalars().items():
            self.tracker.log_scalar(f"{name}/train", value, epoch)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        pass

    def maybe_run_loss_probe(self, train_loader, probe_config=None) -> None:
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
