"""Best-checkpoint tracking, weight EMA and full trainer-state capture for resuming runs."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib    import Path

import numpy as np
import torch


class TorchIO:
    """Writes torch payloads so a killed process cannot leave a truncated file."""

    @staticmethod
    def save_atomic(payload, path) -> Path:
        """Saves a payload to a temporary file and renames it over the target.

        Args:
            payload : Object accepted by torch.save.
            path    : Destination path; parent directories are created.

        Returns:
            The destination path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        target = path.with_name(path.name + ".tmp")
        torch.save(payload, str(target))
        os.replace(target, path)

        return path


class Checkpoint:
    """Saves the model whenever validation loss improves, and restores the best weights.

    The best-loss baseline is discarded whenever the criterion reports a new loss
    generation, since losses on either side of a composition change are not
    comparable.

    Attributes:
        logger          : Logger for checkpoint messages.
        tracker         : Experiment tracker carried alongside the checkpoint.
        save_path       : Destination path of the best checkpoint.
        best_val_loss   : Lowest validation loss seen in the current generation.
        best_epoch      : Zero-based epoch of the best checkpoint, -1 if none.
        loss_generation : Loss-composition generation the baseline belongs to.
    """

    def __init__(self, logger, tracker, save_path: str):
        """Initialises an empty best-checkpoint baseline.

        Args:
            logger    : Logger for checkpoint messages.
            tracker   : Experiment tracker.
            save_path : Path the best checkpoint is written to.
        """
        self.logger          = logger
        self.tracker         = tracker
        self.save_path       = save_path
        self.best_val_loss   = float("inf")
        self.best_epoch      = -1
        self.loss_generation = 0

    def reset_baseline(self, loss_generation: int, epoch: int) -> None:
        """Clears the best-loss baseline after a change of loss composition.

        Args:
            loss_generation : Generation counter of the new loss composition.
            epoch           : Zero-based epoch at which the change was observed.
        """
        self.best_val_loss   = float("inf")
        self.best_epoch      = -1
        self.loss_generation = int(loss_generation)
        self.logger.warning(f"Checkpoint baseline reset at epoch {epoch + 1}: loss composition changed (generation {loss_generation}); best_val_loss is not comparable across the swap and tracking restarts on the new scale.")

    def step(self, val_loss: float, epoch: int, trainer) -> bool:
        """Saves a checkpoint when the validation loss improves on the baseline.

        Args:
            val_loss : Validation loss of this round.
            epoch    : Zero-based epoch index.
            trainer  : Trainer whose state is captured on improvement.

        Returns:
            True when a new best checkpoint was written.
        """
        loss_generation = int(trainer.criterion.loss_generation)
        if loss_generation != self.loss_generation:
            self.reset_baseline(loss_generation, epoch)

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
        """Captures the trainer state with the best-loss bookkeeping and writes it atomically.

        Args:
            trainer : Trainer providing capture_state and the EMA flag.
            path    : Destination path of the checkpoint.
            epoch   : Zero-based epoch index being saved.
        """
        checkpoint = trainer.capture_state(epoch)
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_epoch"]    = self.best_epoch

        weights = "EMA weights" if trainer.ema.enabled else "raw weights"
        self.logger.info(f"Saving checkpoint ({weights}) at epoch {epoch + 1} to {path}")
        TorchIO.save_atomic(checkpoint, path)

    def restore_best(self, model, device) -> None:
        """Loads the best saved parameters into a model, leaving it untouched if none exist.

        Args:
            model  : Model whose state dict is overwritten.
            device : Map location used while loading the checkpoint.
        """
        if self.best_epoch < 0:
            self.logger.warning("No improving checkpoint was saved; leaving the model in its final-epoch state.")
            return

        checkpoint = torch.load(self.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["params"])

        self.logger.subsection(f"Restored best parameters from epoch {self.best_epoch + 1} (val_loss={self.best_val_loss:.4f}).")

    def state_dict(self) -> dict:
        """Returns the best-loss, best-epoch and loss-generation bookkeeping."""
        return {
            "best_val_loss"   : self.best_val_loss,
            "best_epoch"      : self.best_epoch,
            "loss_generation" : self.loss_generation,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores the best-loss bookkeeping from a saved state dictionary."""
        self.best_val_loss   = float(state["best_val_loss"])
        self.best_epoch      = int(state["best_epoch"])
        self.loss_generation = int(state["loss_generation"])


class WeightEma:
    """Exponential moving average of the model parameters, applied on demand.

    With decay warmup enabled the effective decay follows the classic schedule
    min(decay, (1 + t) / (10 + t)) over the update count t, so the shadow
    weights track the model closely while it moves away from its random
    initialisation and only then settle at the target decay.

    Attributes:
        decay       : Target EMA decay factor per optimizer step.
        warmup      : Whether the effective decay ramps with the update count.
        enabled     : Whether the shadow weights are maintained at all.
        num_updates : Number of updates blended into the shadow weights so far.
        last_decay  : Effective decay used by the most recent update, None before the first.
        shadow      : Shadow parameter tensors keyed by parameter name, empty when disabled.
    """

    def __init__(self, model, decay: float, enabled: bool, warmup: bool):
        """Clones the model parameters into the shadow copy when EMA is enabled.

        Args:
            model   : Model whose named parameters are shadowed.
            decay   : Target EMA decay factor per update.
            enabled : Whether EMA is active.
            warmup  : Whether the decay-warmup schedule scales the decay by the update count.
        """
        self.decay       = float(decay)
        self.warmup      = bool(warmup)
        self.enabled     = bool(enabled)
        self.num_updates = 0
        self.last_decay  = None
        self.shadow      = {name: p.detach().clone() for name, p in model.named_parameters()} if self.enabled else {}

    def effective_decay(self) -> float:
        """Returns the decay of the next update under the warmup schedule, the target decay without it."""
        if not self.warmup:
            return self.decay

        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model) -> None:
        """Blends the current model parameters into the shadow weights by the effective decay."""
        if not self.enabled:
            return

        decay             = self.effective_decay()
        self.num_updates += 1
        self.last_decay   = decay

        for name, p in model.named_parameters():
            self.shadow[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    @contextmanager
    def applied(self, model):
        """Temporarily swaps the shadow weights into a model for the duration of the block.

        Args:
            model : Model whose parameters are replaced and then restored.

        Yields:
            None, with the model holding the EMA weights inside the block.
        """
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
        """Returns the shadow weights and the update count for checkpointing."""
        return {"shadow": self.shadow, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict) -> None:
        """Restores the shadow weights onto their current devices.

        Args:
            state : State dictionary produced by state_dict().

        Raises:
            ValueError : If the saved state was produced with a different use_ema setting than the current one.
        """
        shadow = state["shadow"]

        if self.enabled != bool(shadow):
            raise ValueError("use_ema does not match the saved trainer state; resume with the same use_ema setting")

        self.num_updates = int(state["num_updates"])
        self.shadow      = {name: tensor.to(self.shadow[name].device).clone() for name, tensor in shadow.items()}


class WeightSwa:
    """Equal-weight running average of the model parameters from a start epoch.

    Stochastic weight averaging: each update blends the current parameters into
    the average with weight 1/(n+1), so every contributing epoch counts equally.
    Applying the averaged weights also backs up and restores the module buffers,
    because the batch-norm refresh that follows an application rewrites running
    statistics that must not leak into the live model.

    Attributes:
        enabled    : Whether the average is maintained at all.
        n_averaged : Number of parameter snapshots blended in so far.
        shadow     : Averaged parameter tensors keyed by parameter name, empty when disabled.
    """

    def __init__(self, model, enabled: bool):
        """Prepares the empty average over the model's named parameters.

        Args:
            model   : Model whose named parameters are averaged.
            enabled : Whether SWA is active.
        """
        self.enabled    = bool(enabled)
        self.n_averaged = 0
        self.shadow     = {name: torch.zeros_like(p, device=p.device) for name, p in model.named_parameters()} if self.enabled else {}

    @torch.no_grad()
    def update(self, model) -> None:
        """Blends the current model parameters into the running average."""
        if not self.enabled:
            return

        self.n_averaged += 1
        weight           = 1.0 / self.n_averaged

        for name, p in model.named_parameters():
            self.shadow[name].mul_(1.0 - weight).add_(p.detach(), alpha=weight)

    @contextmanager
    def applied(self, model):
        """Temporarily swaps the averaged weights into a model, buffers included in the backup.

        Args:
            model : Model whose parameters are replaced and then restored.

        Yields:
            None, with the model holding the averaged weights inside the block.

        Raises:
            ValueError : If no snapshot has been averaged yet.
        """
        if not self.enabled:
            yield
            return

        if self.n_averaged == 0:
            raise ValueError("SWA has averaged no snapshot yet; the swa_start_fraction lies past the last completed epoch")

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
        """Re-estimates every batch-norm running statistic under the current weights.

        Args:
            model   : Model whose _BatchNorm modules are reset and refreshed.
            loader  : Loader whose batches drive the refresh, one full pass.
            forward : Callable running the model on one loader batch, the trainer's loss step.

        Returns:
            The number of batch-norm modules refreshed; zero means the model has none and nothing ran.
        """
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
        """Returns the averaged weights and the snapshot count for checkpointing."""
        return {"shadow": self.shadow, "n_averaged": self.n_averaged}

    def load_state_dict(self, state: dict) -> None:
        """Restores the averaged weights onto their current devices.

        Args:
            state : State dictionary produced by state_dict().

        Raises:
            ValueError : If the saved state was produced with a different use_swa setting than the current one.
        """
        shadow = state["shadow"]

        if self.enabled != bool(shadow):
            raise ValueError("use_swa does not match the saved trainer state; resume with the same use_swa setting")

        self.n_averaged = int(state["n_averaged"])
        self.shadow     = {name: tensor.to(self.shadow[name].device).clone() for name, tensor in shadow.items()}


class TrainerState:
    """Captures and restores the complete trainer state needed to resume a run.

    The saved payload records the batch geometry the restored schedule is only
    valid under: a resume under another micro-batch, accumulation window, rank
    count or LR scale would continue the warmup position, the current learning
    rates and the step counts of a different run, so it is refused.

    Attributes:
        FILENAME : File name of the rolling trainer state inside a run directory.
    """

    FILENAME = "last.pt"

    @staticmethod
    def path(run_dir) -> Path:
        """Returns the trainer-state path inside a run directory."""
        return Path(run_dir) / TrainerState.FILENAME

    @staticmethod
    def batch_geometry(trainer) -> dict:
        """Returns the settings the saved schedule state depends on: the loader's micro-batch (None for an in-memory batch list), accumulation window, rank count and the batch-rule LR scale."""
        return {
            "batch_size"         : trainer.micro_batch,
            "accumulation_steps" : int(trainer.accumulation_steps),
            "world_size"         : int(trainer.distributed.world_size),
            "lr_scale"           : float(trainer.config.optimizer.lr_scale),
        }

    @staticmethod
    def _check_geometry(trainer, state: dict, path: Path) -> None:
        """Refuses a saved state whose batch geometry differs from the trainer's.

        Args:
            trainer : Trainer about to be repopulated.
            state   : Payload read from the trainer-state file.
            path    : Path of that file, named in the errors.

        Raises:
            KeyError   : If the state predates batch-geometry recording, since the schedule it restores cannot be checked.
            ValueError : If any geometry setting differs, naming each one with its saved and current value.
        """
        if "batch_geometry" not in state:
            raise KeyError(f"trainer state {path} records no batch geometry; it was saved before the micro-batch, accumulation window, rank count and LR scale were captured, so the schedule it restores cannot be checked against this run. Start a fresh run instead of resuming it")

        saved   = state["batch_geometry"]
        current = TrainerState.batch_geometry(trainer)
        changed = [f"{key} {saved[key]} -> {current[key]}" for key in current if saved[key] != current[key]]

        if changed:
            raise ValueError(f"resuming from {path} would continue under a different batch geometry ({', '.join(changed)}): the restored warmup position, learning rates and step counts were computed for the saved values. Resume with the recorded settings, or start a fresh run")

    @staticmethod
    def _rng_state(loader_generator) -> dict:
        """Returns the torch, CUDA, numpy, python and data-loader RNG states."""
        return {
            "torch"  : torch.get_rng_state(),
            "cuda"   : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy"  : np.random.get_state(),
            "random" : random.getstate(),
            "loader" : loader_generator.get_state() if loader_generator is not None else None,
        }

    @staticmethod
    def _restore_rng(rng: dict, loader_generator) -> None:
        """Restores every RNG stream captured by _rng_state.

        Args:
            rng              : RNG state dictionary from _rng_state.
            loader_generator : Data-loader generator to reseed, or None.
        """
        torch.set_rng_state(rng["torch"])

        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        np.random.set_state(rng["numpy"])
        random.setstate(rng["random"])

        if rng["loader"] is not None and loader_generator is not None:
            loader_generator.set_state(rng["loader"])

    @staticmethod
    def capture(trainer, epoch: int, loader_generator=None) -> dict:
        """Collects model, optimizer, schedule, control and RNG state into one payload.

        Args:
            trainer          : Trainer to capture.
            epoch            : Zero-based epoch index just completed.
            loader_generator : Data-loader generator whose state is captured, or None.

        Returns:
            Payload consumed by restore().
        """
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
        """Writes the captured trainer state atomically to a path."""
        TorchIO.save_atomic(TrainerState.capture(trainer, epoch, loader_generator), path)

    @staticmethod
    def restore(trainer, path, loader_generator=None) -> int:
        """Loads a saved state into a trainer and returns the epoch to resume from.

        Args:
            trainer          : Trainer to repopulate in place.
            path             : Path of the saved trainer state.
            loader_generator : Data-loader generator to reseed, or None.

        Returns:
            Zero-based index of the first epoch still to run.

        Raises:
            FileNotFoundError : If no trainer state exists at the path.
            KeyError          : If the state predates batch-geometry recording.
            ValueError        : If the state was saved under another micro-batch, accumulation window, rank count or LR scale.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"resume=True but no trainer state exists at {path}; start a fresh run or point run_name at a run that saved state")

        state = torch.load(str(path), map_location="cpu", weights_only=False)

        TrainerState._check_geometry(trainer, state, path)

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
        trainer._on_state_restored(state)

        return int(state["epoch"]) + 1
