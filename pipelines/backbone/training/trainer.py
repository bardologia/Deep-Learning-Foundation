from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from pipelines.backbone.training.docs import TrainingDocs
from pipelines.backbone.training.loss import Loss
from tools.training.trainer           import BaseTrainer


class Trainer(BaseTrainer):

    stage_name    = "Total"
    section_title = "[PyTorch Training Loop]"

    def __init__(self, model, model_cfg, target_channels, config, run_dir, logger, emit_docs=True, writer=None):
        self.model_cfg       = model_cfg
        self.target_channels = list(target_channels)
        self.emit_docs       = emit_docs

        super().__init__(model, config, run_dir, logger, writer)

        self.docs = TrainingDocs(self.model, self.model_cfg, self.logger, self.run_dir, enabled=self.emit_docs)

    def _log_init_banner(self) -> None:
        self.logger.section("[Training Start]")
        self.logger.kv_table({
            "Backend"       : "PyTorch",
            "Device"        : self.device,
            "Log Directory" : self.config.io.logdir,
        })

    def _build_param_groups(self) -> list[dict]:
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self) -> Loss:
        return Loss(self.config.loss, self.target_channels, self.logger)

    def _warn_nonfinite(self, tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            self.logger.warning(f"NaN or Inf detected in {name} at step {self.global_step}!")

    def _compute_loss(self, batch) -> dict:
        inputs, targets, weights = (tensor.to(self.device) for tensor in batch)

        if self.tracker.debug:
            self._warn_nonfinite(inputs, "inputs")
            self._warn_nonfinite(targets, "targets")

        predictions = self.model(inputs)

        if self.tracker.debug:
            self._warn_nonfinite(predictions, "model predictions")

        return self.criterion(predictions, targets, weights)

    def _nonfinite_detail(self, loss_dict: dict, batch) -> str:
        named    = (("inputs", batch[0]), ("targets", batch[1]))
        bad_data = [name for name, tensor in named if not torch.isfinite(tensor).all()]
        verdict  = f"non-finite {' and '.join(bad_data)} in the batch" if bad_data else "inputs and targets finite, so the model or the loss produced it"

        return f"{super()._nonfinite_detail(loss_dict, batch)}; {verdict}"

    def _before_training(self, train_loader: DataLoader) -> None:
        self.docs.emit(train_loader, self.device)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        self._trial_callback(val_loss, epoch)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        pass
