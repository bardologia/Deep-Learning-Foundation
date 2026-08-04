from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tools.monitoring.tracker          import Tracker
from tools.monitoring.resource_monitor import ResourceMonitor

from tools.training.aggregation      import MetricAggregator
from tools.training.scheduling       import Scheduler, Warmup
from tools.training.stopping         import EarlyStopping
from tools.training.gradients        import GradientClipper
from tools.training.checkpoint       import Checkpoint, TorchIO, TrainerState, WeightEma
from tools.training.vram_reservation import VramReservation
from tools.runtime.completion        import CompletionMarker


class BaseTrainer:
    stage_name    = "Training"
    section_title = "[Training Loop]"

    def __init__(self, model, config, run_dir, logger, x_axis):
        self.logger = logger
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log_init_banner()

        self.run_dir         = Path(run_dir)
        self.checkpoint_path = self.run_dir / "best_model.pt"
        self.tracker         = Tracker(writer=self.config.io.writer, debug=config.training.log_debug)

        self.model  = model.to(self.device)
        self.x_axis = torch.tensor(x_axis, device=self.device, dtype=torch.float32)

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
        self.ema              = WeightEma(self.model, config.training.ema_decay, config.training.use_ema)
        self.resource_monitor = ResourceMonitor(config=config.resources, logger=self.logger, tracker=self.tracker, step_getter=lambda: self.global_step)
        self.vram_reservation = VramReservation(enabled=config.memory.reserve_vram, keep_free_gb=config.memory.vram_keep_free_gb, device=self.device, logger=self.logger)

        self.criterion = self._build_criterion()

        self.resume       = config.training.resume
        self.state_path   = TrainerState.path(self.run_dir)
        self.test_metrics = None

        self.global_step  = 0
        self.train_losses = []
        self.val_losses   = []

        self._gt_occupancy_logged = set()

        self._update_optimizer(self.lr_scheduler.effective_lrs())
        self._log_loop_configuration()

    def _log_init_banner(self) -> None:
        pass

    def _log_loop_configuration(self) -> None:
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
            "EMA"                      : f"enabled (decay {self.ema.decay})" if self.ema.enabled else "disabled",
            "Restore best at end"      : self.restore_best,
            "Resume"                   : self.resume,
            "Abort on non-finite loss" : self.abort_on_nonfinite_loss,
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
        raise NotImplementedError

    def _build_criterion(self):
        raise NotImplementedError

    def _compute_loss(self, batch) -> dict:
        raise NotImplementedError

    def _eval_step(self, batch, aggregator: MetricAggregator) -> dict:
        return self._compute_loss(batch)

    def _set_train_mode(self) -> None:
        self.model.train()

    def _on_optimizer_step(self) -> None:
        pass

    def _log_train_banner(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        micro_batch = train_loader.batch_size if isinstance(train_loader, DataLoader) else None

        info = {
            "Train batches"      : len(train_loader),
            "Validation batches" : len(val_loader),
            "Test batches"       : len(test_loader),
        }

        if micro_batch is not None:
            info["Micro-batch size"]     = micro_batch
            info["Effective batch size"] = f"{micro_batch * self.accumulation_steps} (micro-batch x {self.accumulation_steps} accumulation steps)"
        else:
            info["Micro-batch size"]     = "in-memory batch list (gate/probe run)"

        info["Device"] = self.device
        self.logger.kv_table(info)

    def _before_training(self, train_loader: DataLoader) -> None:
        pass

    def _before_epoch(self, epoch: int) -> None:
        pass

    def _before_validation(self, epoch: int, val_loader: DataLoader) -> None:
        pass

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        pass

    def _log_train_epoch_extra(self, avg_loss: float, epoch: int) -> None:
        pass

    def _on_state_restored(self, state: dict) -> None:
        pass

    def _build_optimizer(self, param_groups: list[dict]):
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
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            if param_group['lr'] == lr:
                continue

            param_group['lr'] = lr
            name              = param_group.get('name', str(i))
            self.tracker.log_scalar(f"optim/lr/{name}", lr, self.global_step)

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.vram_reservation.refill()

    def capture_state(self, epoch: int) -> dict:
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
            "x_axis" : self.x_axis.cpu().numpy(),
        }

    def train_epoch(self, loader: DataLoader, epoch: int):
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)

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

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    loss_dict = self._compute_loss(batch)

                window_start = (batch_idx // self.accumulation_steps) * self.accumulation_steps
                window_len   = min(self.accumulation_steps, n_batches - window_start)
                loss         = loss_dict["total_loss"] / window_len

                if torch.isfinite(loss):
                    loss.backward()
                    window_has_grads = True

                    loss_sum += loss.item() * window_len
                    n += 1
                    aggregator.add(loss_dict)

                else:
                    if self.abort_on_nonfinite_loss:
                        raise FloatingPointError(f"{self.stage_name} loss is non-finite at step {self.global_step}")

                    nonfinite_count += 1
                    self.logger.warning(f"{self.stage_name} loss is non-finite at step {self.global_step}; skipping batch (abort_on_nonfinite_loss disabled).")

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                    if window_has_grads:
                        grad_norm = self.grad_clipper.maybe_clip(self.model, self.global_step)
                        self.grad_clipper.record(grad_norm, self.global_step)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.ema.update(self.model)
                        self._on_optimizer_step()
                        self.warmup.step()

                    window_has_grads = False

                self.global_step += 1
                self.tracker.set_step(self.global_step)

                _prog.update(_task, advance=1)
                fetch_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start

        avg = loss_sum / n if n > 0 else float("nan")
        self.tracker.log_scalar("loss/train", avg, epoch)
        self._log_epoch_metrics("train", aggregator, epoch)
        self._log_throughput(epoch_time, data_wait, sample_count, nonfinite_count, epoch)

        self.tracker.log_memory(self.global_step)
        self._log_train_epoch_extra(avg, epoch)
        return avg

    def _log_throughput(self, epoch_time: float, data_wait: float, sample_count: int, nonfinite_count: int, epoch: int) -> None:
        elapsed = max(epoch_time, 1e-9)

        self.tracker.log_scalar("throughput/samples_per_s",    sample_count / elapsed, epoch)
        self.tracker.log_scalar("throughput/epoch_time_s",     epoch_time,             epoch)
        self.tracker.log_scalar("throughput/data_wait_frac",   data_wait / elapsed,    epoch)
        self.tracker.log_scalar("controls/nonfinite_batches",  float(nonfinite_count), epoch)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage="val"):
        self.model.eval()

        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]Eval {stage}[/section] - epoch {epoch + 1}/{self.epochs}", total=len(loader))

            for batch in loader:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    loss_dict = self._eval_step(batch, aggregator)
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

        avg = loss_sum / n if n > 0 else float("nan")
        self._log_epoch_metrics(stage, aggregator, epoch)

        return {"avg_loss": avg, "num_batches": n}

    def _log_epoch_metrics(self, stage: str, aggregator: MetricAggregator, epoch: int) -> None:
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
        every = self.config.training.snapshot_every_n_epochs
        if every <= 0 or (epoch + 1) % every != 0:
            return

        path = Path(self.run_dir) / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"

        with self.ema.applied(self.model):
            TorchIO.save_atomic({"epoch": epoch, "params": self.model.state_dict()}, path)

        self.logger.subsection(f"Epoch snapshot -> {path}")

    def _maybe_resume(self, loader_generator) -> int:
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
        self.logger.section(self.section_title)
        self._log_train_banner(train_loader, val_loader, test_loader)

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
