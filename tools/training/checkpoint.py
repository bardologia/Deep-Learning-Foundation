
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib    import Path

import numpy as np
import torch


class TorchIO:

    @staticmethod
    def save_atomic(payload, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        target = path.with_name(path.name + ".tmp")
        torch.save(payload, str(target))
        os.replace(target, path)

        return path


class Checkpoint:

    def __init__(self, logger, tracker, save_path: str):
        self.logger          = logger
        self.tracker         = tracker
        self.save_path       = save_path
        self.best_val_loss   = float("inf")
        self.best_epoch      = -1

    def step(self, val_loss: float, epoch: int, trainer) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.best_epoch    = int(epoch)
            self.save(trainer, self.save_path, epoch)
            self.logger.subsection(f"Checkpoint : new best  val_loss={self.best_val_loss:.4f}  -> {self.save_path}")
            return True
        else:
            self.logger.subsection(f"Checkpoint : no improvement  (best={self.best_val_loss:.4f} @ epoch {self.best_epoch + 1})")
            return False

    def save(self, trainer, path: str, epoch: int) -> None:
        checkpoint = trainer.capture_state(epoch)
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_epoch"]    = self.best_epoch

        weights = "EMA weights" if trainer.ema.enabled else "raw weights"
        self.logger.info(f"Saving checkpoint ({weights}) at epoch {epoch + 1} to {path}")
        TorchIO.save_atomic(checkpoint, path)

    def restore_best(self, model, device) -> None:
        if self.best_epoch < 0:
            self.logger.warning("No improving checkpoint was saved; leaving the model in its final-epoch state.")
            return

        checkpoint = torch.load(self.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["params"])

        self.logger.subsection(f"Restored best parameters from epoch {self.best_epoch + 1} (val_loss={self.best_val_loss:.4f}).")

    def state_dict(self) -> dict:
        return {
            "best_val_loss" : self.best_val_loss,
            "best_epoch"    : self.best_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_val_loss = float(state["best_val_loss"])
        self.best_epoch    = int(state["best_epoch"])


class WeightEma:

    def __init__(self, model, decay: float, enabled: bool, warmup: bool):
        self.decay       = float(decay)
        self.warmup      = bool(warmup)
        self.enabled     = bool(enabled)
        self.num_updates = 0
        self.last_decay  = None
        self.shadow      = {name: p.detach().clone() for name, p in model.named_parameters()} if self.enabled else {}

    def effective_decay(self) -> float:
        if not self.warmup:
            return self.decay

        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model) -> None:
        if not self.enabled:
            return

        decay             = self.effective_decay()
        self.num_updates += 1
        self.last_decay   = decay

        for name, p in model.named_parameters():
            self.shadow[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    @contextmanager
    def applied(self, model):
        if not self.enabled:
            yield
            return

        with torch.no_grad():
            backup = {name: p.detach().clone() for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                p.copy_(self.shadow[name])

        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict) -> None:
        shadow = state["shadow"]

        self.num_updates = int(state["num_updates"])
        self.shadow      = {name: tensor.to(self.shadow[name].device).clone() for name, tensor in shadow.items()}


class WeightSwa:

    def __init__(self, model, enabled: bool):
        self.enabled    = bool(enabled)
        self.n_averaged = 0
        self.shadow     = {name: torch.zeros_like(p, device=p.device) for name, p in model.named_parameters()} if self.enabled else {}

    @torch.no_grad()
    def update(self, model) -> None:
        if not self.enabled:
            return

        self.n_averaged += 1
        weight           = 1.0 / self.n_averaged

        for name, p in model.named_parameters():
            self.shadow[name].mul_(1.0 - weight).add_(p.detach(), alpha=weight)

    @contextmanager
    def applied(self, model):
        if not self.enabled:
            yield
            return

        with torch.no_grad():
            param_backup  = {name: p.detach().clone() for name, p in model.named_parameters()}
            buffer_backup = {name: b.detach().clone() for name, b in model.named_buffers()}
            for name, p in model.named_parameters():
                p.copy_(self.shadow[name])

        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(param_backup[name])
                for name, b in model.named_buffers():
                    b.copy_(buffer_backup[name])

    @staticmethod
    @torch.no_grad()
    def refresh_bn(model, loader, forward) -> int:
        bn_modules = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        if not bn_modules:
            return 0

        momenta = {}
        for module in bn_modules:
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None

        was_training = model.training
        model.train()

        for batch in loader:
            forward(batch)

        for module, momentum in momenta.items():
            module.momentum = momentum

        model.train(was_training)

        return len(bn_modules)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "n_averaged": self.n_averaged}

    def load_state_dict(self, state: dict) -> None:
        shadow = state["shadow"]

        self.n_averaged = int(state["n_averaged"])
        self.shadow     = {name: tensor.to(self.shadow[name].device).clone() for name, tensor in shadow.items()}


class TrainerState:

    FILENAME = "last.pt"

    @staticmethod
    def path(run_dir) -> Path:
        return Path(run_dir) / TrainerState.FILENAME

    @staticmethod
    def batch_geometry(trainer) -> dict:
        return {
            "batch_size"         : trainer.micro_batch,
            "accumulation_steps" : int(trainer.accumulation_steps),
            "world_size"         : int(trainer.distributed.world_size),
            "lr_scale"           : float(trainer.config.optimizer.lr_scale),
        }

    @staticmethod
    def _rng_state(loader_generator) -> dict:
        return {
            "torch"  : torch.get_rng_state(),
            "cuda"   : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy"  : np.random.get_state(),
            "random" : random.getstate(),
            "loader" : loader_generator.get_state() if loader_generator is not None else None,
        }

    @staticmethod
    def _restore_rng(rng: dict, loader_generator) -> None:
        torch.set_rng_state(rng["torch"])

        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        np.random.set_state(rng["numpy"])
        random.setstate(rng["random"])

        if rng["loader"] is not None and loader_generator is not None:
            loader_generator.set_state(rng["loader"])

    @staticmethod
    def capture(trainer, epoch: int, loader_generator=None) -> dict:
        return {
            "epoch"          : int(epoch),
            "global_step"    : int(trainer.global_step),
            "batch_geometry" : TrainerState.batch_geometry(trainer),
            "model"          : trainer.model.state_dict(),
            "optimizer"      : trainer.optimizer.state_dict(),
            "ema"            : trainer.ema.state_dict(),
            "swa"            : trainer.swa.state_dict(),
            "warmup"         : trainer.warmup.state_dict(),
            "scheduler"      : trainer.lr_scheduler.state_dict(),
            "early_stopping" : trainer.early_stopping.state_dict(),
            "checkpoint"     : trainer.checkpoint.state_dict(),
            "grad_clipper"   : trainer.grad_clipper.state_dict(),
            "train_losses"   : list(trainer.train_losses),
            "val_losses"     : list(trainer.val_losses),
            "rng"            : TrainerState._rng_state(loader_generator),
        }

    @staticmethod
    def save(trainer, epoch: int, path, loader_generator=None) -> None:
        TorchIO.save_atomic(TrainerState.capture(trainer, epoch, loader_generator), path)

    @staticmethod
    def restore(trainer, path, loader_generator=None) -> int:
        path = Path(path)
        state = torch.load(str(path), map_location="cpu", weights_only=False)

        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.ema.load_state_dict(state["ema"])
        trainer.swa.load_state_dict(state["swa"])
        trainer.warmup.load_state_dict(state["warmup"])
        trainer.lr_scheduler.load_state_dict(state["scheduler"])
        trainer.early_stopping.load_state_dict(state["early_stopping"])
        trainer.checkpoint.load_state_dict(state["checkpoint"])
        trainer.grad_clipper.load_state_dict(state["grad_clipper"])

        trainer.train_losses = list(state["train_losses"])
        trainer.val_losses   = list(state["val_losses"])
        trainer.global_step  = int(state["global_step"])

        TrainerState._restore_rng(state["rng"], loader_generator)

        return int(state["epoch"]) + 1
