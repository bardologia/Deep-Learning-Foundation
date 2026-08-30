"""Generic epoch-driven training loop shared by every training entry point.

`BaseTrainer` wires together the optimizer, LR schedule, warmup, gradient clipping,
early stopping, checkpointing, EMA, resource monitoring and VRAM reservation, and
leaves the loss, parameter groups and criterion to subclasses.
"""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from tools.monitoring.tracker          import Tracker
from tools.monitoring.resource_monitor import ResourceMonitor

from tools.training.aggregation      import MetricAggregator
from tools.training.distributed      import DistributedContext
from tools.training.scheduling       import Scheduler, Warmup
from tools.training.stopping         import EarlyStopping
from tools.training.gradients        import GradientClipper
from tools.training.checkpoint       import Checkpoint, TorchIO, TrainerState, WeightEma
from tools.training.vram_reservation import VramReservation
from tools.runtime.completion        import CompletionMarker


class BaseTrainer:
    """Abstract trainer that runs the epoch loop and owns all training machinery.

    Subclasses supply `_build_param_groups`, `_build_criterion` and `_compute_loss`,
    and may override the hook methods to inject stage-specific logging or state.

    Attributes:
        device       : Device the model and batches live on (CUDA when available).
        distributed  : Data-parallel context; inert outside a DDP fan-out, and otherwise the rank whose gradients and evaluation losses are kept in lockstep with the other ranks.
        run_dir      : Directory holding checkpoints, trainer state and completion markers.
        tracker      : Scalar/metric sink backed by the TensorBoard writer.
        model        : Module being trained, already moved to `device`.
        x_axis       : Elevation sampling axis of shape (n_samples,), in metres.
        optimizer    : AdamW instance built over the subclass parameter groups.
        base_lrs     : Per-group learning rates after `optimizer.lr_scale` is applied.
        global_step  : Number of micro-batches processed so far.
        train_losses : Mean training loss per completed epoch.
        val_losses   : Validation loss per epoch, NaN on epochs without evaluation.
        test_metrics : Result of the final test evaluation, None until `train` ends.
    """

    stage_name    = "Training"
    section_title = "[Training Loop]"

    def __init__(self, model, config, run_dir, logger, z_axis, writer=None):
        """Builds the trainer and every training component from the resolved config.

        Args:
            model   : Module to train; moved to the selected device.
            config  : Resolved configuration exposing training, optimizer, memory, early_stopping and resources sections.
            run_dir : Run directory for checkpoints, trainer state and markers.
            logger  : Logger used for sections, tables and progress bars.
            z_axis  : Z sampling axis of shape (n_samples,), in metres.
            writer  : Optional TensorBoard writer handed to the tracker.

        Raises:
            ValueError : If AMP is requested on a CUDA device without bfloat16 support.
        """
        self.logger = logger
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.distributed = DistributedContext.current()

        if self.distributed.active and self.distributed.device is None:
            self.logger.info(f"DDP rank {self.distributed.rank}/{self.distributed.world_size}: joining the process-group rendezvous at tcp://{self.distributed.master_addr}:{self.distributed.master_port}; this blocks until every rank has built its dataset and arrived, and the other ranks stream their progress into the run's ddp/logs directory, not this console")

        self.distributed.initialise(self.device)

        if self.distributed.active:
            self.logger.info(f"DDP rank {self.distributed.rank}/{self.distributed.world_size}: process group ready on {self.distributed.device}")

        self._log_init_banner()

        self.run_dir         = Path(run_dir)
        self.checkpoint_path = self.run_dir / "best_model.pt"
        self.tracker         = Tracker(writer=writer, debug=config.training.log_debug)

        self.model  = model.to(self.device)
        self.z_axis = torch.tensor(z_axis, device=self.device, dtype=torch.float32)

        self.distributed.broadcast_parameters(self.model)

        self.epochs                  = config.training.epochs
        self.validation_frequency    = config.training.validation_frequency
        self.accumulation_steps      = config.training.gradient_accumulation_steps
        self.use_amp                 = config.training.use_amp
        self.abort_on_nonfinite_loss = config.training.abort_on_nonfinite_loss

        if self.use_amp and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("use_amp=True autocasts to bfloat16, which this GPU does not support; disable use_amp on this device")

        param_groups = self._build_param_groups()

        lr_scale = self.config.optimizer.lr_scale
        if lr_scale != 1.0:
            for pg in param_groups:
                pg["lr"] = float(pg["lr"]) * lr_scale

        self.base_lrs  = [float(pg["lr"]) for pg in param_groups]
        self.optimizer = self._build_optimizer(param_groups)

        self.warmup           = Warmup(config, self.logger)
        self.lr_scheduler     = Scheduler(self.base_lrs, self.warmup, config, self.logger)
        self.early_stopping   = EarlyStopping(config, self.logger, self.tracker)
        self.restore_best     = config.early_stopping.restore_best
        self.grad_clipper     = GradientClipper(config=config, logger=self.logger, tracker=self.tracker, param_groups=self.optimizer.param_groups)
        self.checkpoint       = Checkpoint(self.logger, self.tracker, str(self.checkpoint_path))
        self.ema              = WeightEma(self.model, config.training.ema_decay, config.training.use_ema, config.training.ema_warmup)
        self.resource_monitor = ResourceMonitor(config=config.resources, logger=self.logger, tracker=self.tracker, step_getter=lambda: self.global_step)
        self.vram_reservation = VramReservation(enabled=config.memory.reserve_vram, keep_free_gb=config.memory.vram_keep_free_gb, device=self.device, logger=self.logger)

        self.criterion = self._build_criterion()

        self.resume       = config.training.resume
        self.state_path   = TrainerState.path(self.run_dir)
        self.test_metrics = None

        self.global_step  = 0
        self.train_losses = []
        self.val_losses   = []

        self._ema_decay_logged = None

        self._gt_occupancy_logged = set()

        self._update_optimizer(self.lr_scheduler.effective_lrs())
        self._log_loop_configuration()

    def _log_init_banner(self) -> None:
        """Hook for subclasses to print a banner before components are built."""
        pass

    def _log_loop_configuration(self) -> None:
        """Logs the device, loop settings and the optimizer param-group table."""
        training = self.config.training
        memory   = self.config.memory

        if self.device.type == "cuda":
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            device  = f"cuda ({torch.cuda.get_device_name(self.device)}, CUDA_VISIBLE_DEVICES={visible})"
        else:
            device  = str(self.device)

        self.logger.section("[Training Loop Configuration]")
        self.logger.kv_table({
            "Device"                   : device,
            "Epochs"                   : self.epochs,
            "Validation frequency"     : f"every {self.validation_frequency} epochs",
            "Gradient accumulation"    : f"{self.accumulation_steps} batches per optimizer step",
            "AMP (bfloat16 autocast)"  : self.use_amp,
            "EMA"                      : f"enabled (decay {self.ema.decay}, decay warmup {'min(decay, (1+t)/(10+t))' if self.ema.warmup else 'off'})" if self.ema.enabled else "disabled",
            "Restore best at end"      : self.restore_best,
            "Resume"                   : self.resume,
            "Abort on non-finite loss" : self.abort_on_nonfinite_loss,
            "DDP"                      : f"rank {self.distributed.rank + 1}/{self.distributed.world_size} (gradients averaged, val losses reduced across ranks)" if self.distributed.active else "off",
            "Debug logging"            : training.log_debug,
            "LR scale (batch-size rule)": f"x{self.config.optimizer.lr_scale:.4f} (from the micro-batch, ignores accumulation)",
            "Optimizer"                : f"AdamW (default betas={tuple(self.config.optimizer.betas)}, eps={self.config.optimizer.eps:g}, weight_decay={self.config.optimizer.weight_decay:g}; per-group values below take precedence)",
            "VRAM reservation"         : f"{memory.reserve_vram} (keep free {memory.vram_keep_free_gb:g} GB)",
            "cudnn"                    : f"deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}",
        })

        rows = []
        for i, pg in enumerate(self.optimizer.param_groups):
            rows.append({
                "Group"        : pg.get("name", str(i)),
                "LR"           : f"{pg['lr']:g}",
                "Weight decay" : f"{pg['weight_decay']:g}",
            })

        self.logger.metrics_table(rows, columns=["Group", "LR", "Weight decay"], title="Optimizer Param Groups (after lr_scale)")

    def _build_param_groups(self) -> list[dict]:
        """Returns the optimizer parameter groups; implemented by subclasses.

        Raises:
            NotImplementedError : Always, on the base class.
        """
        raise NotImplementedError

    def _build_criterion(self):
        """Returns the loss module used by `_compute_loss`; implemented by subclasses.

        Raises:
            NotImplementedError : Always, on the base class.
        """
        raise NotImplementedError

    def _compute_loss(self, batch) -> dict:
        """Returns the loss dictionary for one batch; implemented by subclasses.

        Args:
            batch : One batch as yielded by the data loader.

        Returns:
            Dictionary carrying at least "total_loss" as a scalar tensor.

        Raises:
            NotImplementedError : Always, on the base class.
        """
        raise NotImplementedError

    def _eval_step(self, batch) -> dict:
        """Returns the loss dictionary for one evaluation batch.

        Args:
            batch : One batch as yielded by the validation or test loader.

        Returns:
            Dictionary carrying at least "total_loss" as a scalar tensor.
        """
        return self._compute_loss(batch)

    def _set_train_mode(self) -> None:
        """Puts the model into training mode before an epoch."""
        self.model.train()

    def _on_optimizer_step(self) -> None:
        """Hook fired after every optimizer step and EMA update."""
        pass

    def _log_train_banner(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        """Logs loader sizes, micro-batch and effective batch size, and the device.

        Args:
            train_loader : Loader iterated during training epochs.
            val_loader   : Loader iterated during validation.
            test_loader  : Loader iterated for the final test evaluation.
        """
        micro_batch = train_loader.batch_size if isinstance(train_loader, DataLoader) else None

        info = {
            "Train batches"      : len(train_loader),
            "Validation batches" : len(val_loader),
            "Test batches"       : len(test_loader),
        }

        if micro_batch is not None:
            world                        = self.distributed.world_size
            info["Micro-batch size"]     = micro_batch
            info["Effective batch size"] = f"{micro_batch * self.accumulation_steps * world} (micro-batch x {self.accumulation_steps} accumulation steps" + (f" x {world} DDP ranks)" if world > 1 else ")")
        else:
            info["Micro-batch size"]     = "in-memory batch list (gate/probe run)"

        info["Device"] = self.device
        self.logger.kv_table(info)

    def _before_training(self, train_loader: DataLoader) -> None:
        """Hook fired once after resume and before the first epoch.

        Args:
            train_loader : Loader that will drive the training epochs.
        """
        pass

    def _before_epoch(self, epoch: int) -> None:
        """Hook fired at the start of each epoch.

        Args:
            epoch : Zero-based epoch index about to run.
        """
        pass

    def _before_validation(self, epoch: int, val_loader: DataLoader) -> None:
        """Hook fired before each validation pass.

        Args:
            epoch      : Zero-based epoch index that just finished training.
            val_loader : Loader about to be evaluated.
        """
        pass

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        """Hook fired after validation, checkpointing and early stopping.

        Args:
            val_loss : Mean validation loss of the epoch.
            epoch    : Zero-based epoch index.
        """
        pass

    def _log_train_epoch_extra(self, avg_loss: float, epoch: int) -> None:
        """Hook for extra per-epoch training logging.

        Args:
            avg_loss : Mean training loss over the epoch.
            epoch    : Zero-based epoch index.
        """
        pass

    def _on_state_restored(self, state: dict) -> None:
        """Hook fired after trainer state is restored from disk.

        Args:
            state : Payload read back from the trainer state file.
        """
        pass

    def _build_optimizer(self, param_groups: list[dict]):
        """Returns an AdamW optimizer over the given groups, filling shared defaults.

        Args:
            param_groups : Parameter groups from `_build_param_groups`; missing betas, eps and weight_decay entries are filled from the config.

        Returns:
            The configured AdamW optimizer.
        """
        betas        = tuple(self.config.optimizer.betas)
        eps          = self.config.optimizer.eps
        weight_decay = self.config.optimizer.weight_decay

        for pg in param_groups:
            pg.setdefault("betas",        betas)
            pg.setdefault("eps",          eps)
            pg.setdefault("weight_decay", weight_decay)

        optimizer = optim.AdamW(param_groups)
        return optimizer

    def _update_optimizer(self, lrs: list[float]):
        """Writes new learning rates into the optimizer and logs the ones that changed.

        Args:
            lrs : One learning rate per optimizer parameter group, in group order.
        """
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            if param_group['lr'] == lr:
                continue

            param_group['lr'] = lr
            name              = param_group.get('name', str(i))
            self.tracker.log_scalar(f"optim/lr/{name}", lr, self.global_step)

    def _log_ema_decay(self):
        """Logs the effective EMA decay of the last update whenever it changed."""
        if not self.ema.enabled or self.ema.last_decay == self._ema_decay_logged:
            return

        self._ema_decay_logged = self.ema.last_decay
        self.tracker.log_scalar("optim/ema_decay", self.ema.last_decay, self.global_step)

    def _clear_cuda_cache(self) -> None:
        """Collects Python garbage and empties the CUDA allocator cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def capture_state(self, epoch: int) -> dict:
        """Returns the checkpoint payload for the given epoch.

        Args:
            epoch : Zero-based epoch index the weights correspond to.

        Returns:
            Dictionary with the epoch index, the model state dict and the
            elevation axis of shape (n_samples,) as a NumPy array in metres.
        """
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
            "z_axis" : self.z_axis.cpu().numpy(),
        }

    def micro_step(self, batch, window_len: int) -> tuple[dict, float]:
        """Runs one micro-batch under autocast, scales its loss by the accumulation window and backpropagates it.

        Args:
            batch      : One batch as yielded by the training loader.
            window_len : Micro-batches in the accumulation window this batch belongs to.

        Returns:
            Tuple of the loss dictionary and the window-scaled loss as a float,
            NaN when the loss was non-finite and no backward pass ran.
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            loss_dict = self._compute_loss(batch)

        loss = loss_dict["total_loss"] / window_len

        if not torch.isfinite(loss):
            return loss_dict, float("nan")

        loss.backward()

        return loss_dict, float(loss.detach())

    def apply_gradients(self, global_step: int) -> float:
        """Clips the accumulated gradients, takes the optimizer step and clears the gradients.

        Args:
            global_step : Optimizer step index the gradient norm is logged under.

        Returns:
            The gradient norm measured before any clipping.

        Raises:
            FloatingPointError : If the gradient norm is non-finite.
        """
        grad_norm = self.grad_clipper.maybe_clip(self.model, global_step)
        self.grad_clipper.record(grad_norm, global_step)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return grad_norm

    def train_epoch(self, loader: DataLoader, epoch: int):
        """Runs one training epoch with gradient accumulation and returns its mean loss.

        Non-finite batch losses either abort the run or are skipped, depending on
        `abort_on_nonfinite_loss`. Loss terms, throughput and memory are logged.
        In a DDP group the sampler is reshuffled for the epoch, the accumulated
        gradients are averaged across ranks at every optimizer window, and the
        returned mean loss is the cross-rank mean.

        Args:
            loader : Loader yielding training batches.
            epoch  : Zero-based epoch index.

        Returns:
            Mean training loss over the accepted batches, NaN if none were accepted.

        Raises:
            FloatingPointError : If a batch loss is non-finite and aborting is enabled.
        """
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)

        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()
        n_batches   = len(loader)

        window_has_grads = False
        nonfinite_count  = 0
        sample_count     = 0
        data_wait        = 0.0
        epoch_start      = time.perf_counter()

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]Epoch {epoch + 1}/{self.epochs}[/section] - train", total=n_batches)

            fetch_start = time.perf_counter()
            for batch_idx, batch in enumerate(loader):
                data_wait    += time.perf_counter() - fetch_start
                sample_count += batch.shape[0] if isinstance(batch, torch.Tensor) else len(batch[0])

                self._update_optimizer(self.lr_scheduler.effective_lrs())

                window_start = (batch_idx // self.accumulation_steps) * self.accumulation_steps
                window_len   = min(self.accumulation_steps, n_batches - window_start)

                loss_dict, scaled = self.micro_step(batch, window_len)

                if math.isfinite(scaled):
                    window_has_grads = True

                    loss_sum += scaled * window_len
                    n += 1
                    aggregator.add(loss_dict)

                else:
                    if self.abort_on_nonfinite_loss:
                        raise FloatingPointError(f"{self.stage_name} loss is non-finite at step {self.global_step}")

                    nonfinite_count += 1
                    self.logger.warning(f"{self.stage_name} loss is non-finite at step {self.global_step}; skipping batch (abort_on_nonfinite_loss disabled).")

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                    window_has_grads = self.distributed.sync_window(self.model, window_has_grads)

                    if window_has_grads:
                        self.apply_gradients(self.global_step)
                        self.ema.update(self.model)
                        self._log_ema_decay()
                        self._on_optimizer_step()
                        self.warmup.step()

                    window_has_grads = False

                self.global_step += 1
                self.tracker.set_step(self.global_step)

                _prog.update(_task, advance=1)
                fetch_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start

        avg = self.distributed.mean_scalar(loss_sum / n if n > 0 else float("nan"))
        self.tracker.log_scalar("loss/train", avg, epoch)
        self._log_epoch_metrics("train", aggregator, epoch)
        self._log_throughput(epoch_time, data_wait, sample_count, nonfinite_count, epoch)

        self.tracker.log_memory(self.global_step)
        self._log_train_epoch_extra(avg, epoch)
        return avg

    def _log_throughput(self, epoch_time: float, data_wait: float, sample_count: int, nonfinite_count: int, epoch: int) -> None:
        """Logs samples per second, epoch wall time, data-wait fraction and skipped batches.

        Args:
            epoch_time      : Wall-clock duration of the epoch, in seconds.
            data_wait       : Time spent blocked on the loader, in seconds.
            sample_count    : Number of samples seen during the epoch.
            nonfinite_count : Number of batches skipped for a non-finite loss.
            epoch           : Zero-based epoch index used as the x value.
        """
        elapsed = max(epoch_time, 1e-9)

        self.tracker.log_scalar("throughput/samples_per_s",    sample_count / elapsed, epoch)
        self.tracker.log_scalar("throughput/epoch_time_s",     epoch_time,             epoch)
        self.tracker.log_scalar("throughput/data_wait_frac",   data_wait / elapsed,    epoch)
        self.tracker.log_scalar("controls/nonfinite_batches",  float(nonfinite_count), epoch)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage="val"):
        """Evaluates the model over a loader without gradients.

        In a DDP group each rank evaluates its own shard and the loss sums are
        reduced across ranks, so every rank returns the same average and all
        checkpoint and early-stopping decisions stay identical. The aggregator
        metrics remain rank-local.

        Args:
            loader : Loader yielding evaluation batches.
            epoch  : Zero-based epoch index used for logging.
            stage  : Metric stage label, "val" or "test".

        Returns:
            Dictionary with "avg_loss" (NaN when no batch was accepted) and
            "num_batches", the count of finite-loss batches.

        Raises:
            FloatingPointError : If a batch loss is non-finite and aborting is enabled.
        """
        self.model.eval()

        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]Eval {stage}[/section] - epoch {epoch + 1}/{self.epochs}", total=len(loader))

            for batch in loader:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    loss_dict = self._eval_step(batch)
                loss = loss_dict["total_loss"]

                if torch.isfinite(loss):
                    loss_sum += loss.item()
                    n += 1
                    aggregator.add(loss_dict)

                elif self.abort_on_nonfinite_loss:
                    raise FloatingPointError(f"{self.stage_name} {stage} loss is non-finite at epoch {epoch + 1}")

                else:
                    self.logger.warning(f"{self.stage_name} {stage} loss is non-finite at epoch {epoch + 1}; skipping batch (abort_on_nonfinite_loss disabled).")

                _prog.advance(_task)

        loss_sum, n = self.distributed.sum_scalars(loss_sum, float(n))

        avg = loss_sum / n if n > 0 else float("nan")
        self._log_epoch_metrics(stage, aggregator, epoch)

        return {"avg_loss": avg, "num_batches": int(n)}

    def _log_epoch_metrics(self, stage: str, aggregator: MetricAggregator, epoch: int) -> None:
        """Logs the reduced loss terms, monitors, occupancy and parameter errors.

        Ground-truth occupancy statistics are dataset constants, so they are printed
        once per stage as a table instead of being logged every epoch.

        Args:
            stage      : Metric stage label, "train", "val" or "test".
            aggregator : Aggregator holding the epoch's per-batch loss dictionaries.
            epoch      : Zero-based epoch index used as the x value.
        """
        self.tracker.log_staged("loss_terms", aggregator.reduce_components(), stage, epoch)

        monitor = aggregator.reduce_monitor()
        if monitor:
            self.tracker.log_staged("loss_monitor", monitor, stage, epoch)

        occupancy = aggregator.reduce_occupancy()
        if occupancy:
            gt_stats  = {k: v for k, v in occupancy.items() if k.startswith("gt_")}
            pred_side = {k: v for k, v in occupancy.items() if not k.startswith("gt_")}
            self.tracker.log_staged("occupancy", pred_side, stage, epoch)

            if gt_stats and stage not in self._gt_occupancy_logged:
                self._gt_occupancy_logged.add(stage)
                self.logger.kv_table({k: f"{v:.4f}" for k, v in gt_stats.items()}, title=f"GT Occupancy ({stage}, dataset constants)")

        physical = aggregator.reduce_physical()
        if physical:
            self.tracker.log_staged("param_error", physical, stage, epoch)

    def _maybe_snapshot(self, epoch: int) -> None:
        """Writes a periodic epoch snapshot when `snapshot_every_n_epochs` says so.

        Args:
            epoch : Zero-based epoch index that just finished.
        """
        every = self.config.training.snapshot_every_n_epochs
        if every <= 0 or (epoch + 1) % every != 0:
            return

        path = Path(self.run_dir) / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"

        with self.ema.applied(self.model):
            TorchIO.save_atomic({"epoch": epoch, "params": self.model.state_dict()}, path)

        self.logger.subsection(f"Epoch snapshot -> {path}")

    def _maybe_resume(self, loader_generator) -> int:
        """Restores trainer state from disk when resuming and returns the start epoch.

        Args:
            loader_generator : The training loader's RNG generator, whose state is restored alongside the trainer state.

        Returns:
            Zero-based epoch index the loop should start from; 0 when not resuming.
        """
        if not self.resume:
            return 0

        start_epoch = TrainerState.restore(self, self.state_path, loader_generator)

        self.logger.section("[Resumed Trainer State]")
        self.logger.kv_table({
            "State path"  : str(self.state_path),
            "Next epoch"  : f"{start_epoch + 1}/{self.epochs}",
            "Global step" : self.global_step,
            "Best val"    : f"{self.checkpoint.best_val_loss:.4f} @ epoch {self.checkpoint.best_epoch + 1}",
        })

        return start_epoch

    def _evaluate_test(self, test_loader: DataLoader, epoch: int) -> dict:
        """Runs the final test evaluation and logs which weights produced it.

        With `restore_best` the already-restored best checkpoint is used as is;
        otherwise the EMA weights, when enabled, are applied for the pass.

        Args:
            test_loader : Loader yielding test batches.
            epoch       : Zero-based index of the last epoch that ran.

        Returns:
            Dictionary with "avg_loss" and "num_batches" from `evaluate`.
        """
        if self.restore_best and self.checkpoint.best_epoch >= 0:
            kind    = "EMA snapshot" if self.ema.enabled else "raw weights"
            weights = f"restored best checkpoint (epoch {self.checkpoint.best_epoch + 1}, {kind})"
            test    = self.evaluate(test_loader, epoch, stage="test")
        else:
            weights = "EMA of final weights" if self.ema.enabled else "final-epoch weights"
            with self.ema.applied(self.model):
                test = self.evaluate(test_loader, epoch, stage="test")

        self.tracker.log_scalar("loss/test", test["avg_loss"], epoch)
        self.logger.subsection(f"Test : loss={test['avg_loss']:.4f}  (batches={test['num_batches']}, weights: {weights})")

        return test

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Runs the full training schedule and returns the loss history.

        Resolves the warmup fraction into a step count over the run's optimizer-step
        budget, then sequences resume, the epoch loop with periodic validation, checkpointing,
        snapshots and early stopping, then the final-weights checkpoint, the final
        test evaluation and the completion marker. The final-epoch weights (EMA
        when enabled) are always written to final_model.pt before any best-checkpoint
        restore, so the end-of-training model stays loadable regardless of where
        the best validation loss landed. The resource monitor is always stopped on exit.

        Args:
            train_loader : Loader yielding training batches.
            val_loader   : Loader yielding validation batches.
            test_loader  : Loader yielding test batches.

        Returns:
            Tuple of the per-epoch training losses, the per-epoch validation losses
            (NaN on epochs without evaluation) and the best validation loss seen.
        """
        self.logger.section(self.section_title)
        self._log_train_banner(train_loader, val_loader, test_loader)

        steps_per_epoch = math.ceil(len(train_loader) / self.accumulation_steps)
        self.warmup.resolve(self.epochs * steps_per_epoch)

        CompletionMarker.clear(self.run_dir)
        self._clear_cuda_cache()

        if self.resource_monitor is not None:
            self.resource_monitor.start()
        try:
            loader_generator = getattr(train_loader, "generator", None)
            start_epoch      = self._maybe_resume(loader_generator)
            last_epoch       = max(start_epoch - 1, 0)
            stop             = False

            self._before_training(train_loader)
            self.vram_reservation.fill()

            with self.logger.live_monitor("Training Progress") as live_mon:
                with self.logger.track(transient=False) as _prog_epochs:
                    _task_epochs = _prog_epochs.add_task("[section]Training[/section]", total=self.epochs, completed=start_epoch)

                    for epoch in range(start_epoch, self.epochs):
                        epoch_num  = epoch + 1
                        last_epoch = epoch

                        self.logger.section(f"[Epoch {epoch_num}/{self.epochs}]")
                        self._before_epoch(epoch)

                        train_loss = self.train_epoch(train_loader, epoch)
                        self.logger.subsection(f"Train  : loss={train_loss:.4f}")

                        do_eval = (epoch_num % self.validation_frequency == 0) or (epoch_num == self.epochs)
                        if do_eval:
                            self._before_validation(epoch, val_loader)
                            with self.ema.applied(self.model):
                                val      = self.evaluate(val_loader, epoch, stage="val")
                                val_loss = val["avg_loss"]
                                self.checkpoint.step(val_loss, epoch, self)

                            self.logger.subsection(f"Validation : loss={val_loss:.4f}  (batches={val['num_batches']})")
                            self.tracker.log_scalar("loss/val", val_loss, epoch)
                            self.lr_scheduler.step(epoch)
                            stop = self.early_stopping(val_loss, epoch)
                            self._after_eval(val_loss, epoch)
                        else:
                            val_loss = float("nan")
                            self.lr_scheduler.step(epoch)
                            stop = False

                        self.train_losses.append(train_loss)
                        self.val_losses.append(val_loss)

                        effective_lrs = self.lr_scheduler.effective_lrs()
                        self._update_optimizer(effective_lrs)

                        TrainerState.save(self, epoch, self.state_path, loader_generator)
                        self._maybe_snapshot(epoch)

                        live_mon.update(
                            epoch         = f"{epoch_num}/{self.epochs}",
                            train_loss    = train_loss,
                            val_loss      = val_loss,
                            best_val_loss = self.checkpoint.best_val_loss,
                            best_epoch    = self.checkpoint.best_epoch + 1,
                            lr            = effective_lrs[0],
                        )
                        _prog_epochs.update(_task_epochs, advance=1, description=f"[section]Training[/section]  best_val={self.checkpoint.best_val_loss:.4f} @ ep {self.checkpoint.best_epoch + 1}")

                        if stop:
                            break

            last_path = Path(self.run_dir) / "final_model.pt"
            with self.ema.applied(self.model):
                final_payload = self.capture_state(last_epoch)

            final_payload["best_val_loss"] = self.checkpoint.best_val_loss
            final_payload["best_epoch"]    = self.checkpoint.best_epoch

            TorchIO.save_atomic(final_payload, last_path)
            self.logger.subsection(f"Final checkpoint -> {last_path}")

            if self.restore_best:
                self.checkpoint.restore_best(self.model, self.device)

            self.test_metrics = self._evaluate_test(test_loader, last_epoch)

            CompletionMarker.stamp(self.run_dir, {
                "stage"            : "training",
                "epochs_completed" : last_epoch + 1,
                "epochs_total"     : self.epochs,
                "early_stopped"    : bool(stop),
                "best_val_loss"    : float(self.checkpoint.best_val_loss),
                "best_epoch"       : self.checkpoint.best_epoch + 1,
            })

            return self.train_losses, self.val_losses, self.checkpoint.best_val_loss
        finally:
            if self.resource_monitor is not None:
                self.resource_monitor.stop()
