from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pipelines.backbone.training.docs     import TrainingDocs
from pipelines.backbone.training.loss     import Loss
from tools.monitoring.resource_monitor    import ResourceMonitor
from tools.monitoring.tracker             import Tracker
from tools.runtime.completion             import CompletionMarker
from tools.training.activation_checkpoint import ActivationCheckpointing
from tools.training.aggregation           import MetricAggregator, ScalarsPage
from tools.training.checkpoint            import Checkpoint, TorchIO, TrainerState, WeightEma, WeightSwa
from tools.training.gradients             import GradientClipper
from tools.training.scheduling            import Scheduler, Warmup
from tools.training.stopping              import EarlyStopping
from tools.training.vram_reservation      import VramReservation


class Trainer:

    def __init__(self, model, model_cfg, target_channels, config, run_dir, logger, emit_docs=True, writer=None):
        self.model_cfg       = model_cfg
        self.target_channels = list(target_channels)
        self.logger          = logger
        self.config          = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._warn_cpu_device()
        self._log_init_banner()

        self.run_dir         = Path(run_dir)
        self.checkpoint_path = self.run_dir / "best_model.pt"
        self.tracker         = Tracker(writer=writer, debug=config.training.log_debug)

        self.model = model.to(self.device)

        self.channels_last = config.training.channels_last
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if config.training.activation_checkpointing:
            ActivationCheckpointing.apply(self.model, tuple(config.training.activation_checkpointing), self.logger)

        if config.training.compile_model:
            self.model.forward = torch.compile(self.model.forward, mode=config.training.compile_mode)

        self.epochs                  = config.training.epochs
        self.validation_frequency    = config.training.validation_frequency
        self.accumulation_steps      = config.training.gradient_accumulation_steps
        self.use_amp                 = config.training.use_amp
        self.abort_on_nonfinite_loss = config.training.abort_on_nonfinite_loss

        param_groups = self.model_cfg.get_param_groups(self.model)

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
        self.swa              = WeightSwa(self.model, config.training.use_swa)
        self.swa_start_epoch  = int(config.training.swa_start_fraction * self.epochs)
        self.resource_monitor = ResourceMonitor(config=config.resources, logger=self.logger, tracker=self.tracker, step_getter=lambda: self.global_step)
        self.vram_reservation = VramReservation(enabled=config.memory.reserve_vram, keep_free_gb=config.memory.vram_keep_free_gb, device=self.device, logger=self.logger)

        self.criterion = Loss(self.config.loss, self.target_channels, self.logger)

        self.resume       = config.training.resume
        self.state_path   = TrainerState.path(self.run_dir)
        self.test_metrics = None
        self.micro_batch  = None

        self.global_step  = 0
        self.train_losses = []
        self.val_losses   = []

        self._ema_decay_logged = None

        self._update_optimizer(self.lr_scheduler.effective_lrs())
        self._log_loop_configuration()

        self.docs = TrainingDocs(self.model, self.model_cfg, self.logger, self.run_dir, enabled=emit_docs)

    def _warn_cpu_device(self) -> None:
        if self.device.type == "cuda":
            return

        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        amp  = "; use_amp now autocasts to bfloat16 on the CPU" if self.config.training.use_amp else ""

        self.logger.warning(f"CUDA unavailable (CUDA_VISIBLE_DEVICES={mask}, torch CUDA build {torch.version.cuda}): training runs on the CPU at a fraction of GPU speed{amp}")

    def _log_init_banner(self) -> None:
        self.logger.section("[Training Start]")
        self.logger.kv_table({
            "Backend"       : "PyTorch",
            "Device"        : self.device,
            "Log Directory" : self.config.io.logdir,
        })

    def _build_optimizer(self, param_groups: list[dict]):
        cfg  = self.config.optimizer
        kind = str(cfg.type)

        for pg in param_groups:
            pg.setdefault("weight_decay", cfg.weight_decay)

        if kind in ("adamw", "adam"):
            for pg in param_groups:
                pg.setdefault("betas", tuple(cfg.betas))
                pg.setdefault("eps",   cfg.eps)
            return optim.AdamW(param_groups) if kind == "adamw" else optim.Adam(param_groups)

        if kind == "sgd":
            for pg in param_groups:
                pg.setdefault("momentum", cfg.momentum)
                pg.setdefault("nesterov", cfg.nesterov and cfg.momentum > 0.0)
            return optim.SGD(param_groups)

        for pg in param_groups:
            pg.setdefault("momentum", cfg.momentum)
            pg.setdefault("eps",      cfg.eps)
        return {"rmsprop": optim.RMSprop}[kind](param_groups)

    def _update_optimizer(self, lrs: list[float]):
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            if param_group['lr'] == lr:
                continue

            param_group['lr'] = lr
            name              = param_group.get('name', str(i))
            self.tracker.log_scalar(f"optim/lr/{name}", lr, self.global_step)

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
            "torch.compile"            : f"enabled (mode {self.config.training.compile_mode})" if self.config.training.compile_model else "disabled",
            "channels_last"            : self.channels_last,
            "Activation checkpointing" : f"patterns {tuple(self.config.training.activation_checkpointing)}" if self.config.training.activation_checkpointing else "disabled",
            "EMA"                      : f"enabled (decay {self.ema.decay}, decay warmup {'min(decay, (1+t)/(10+t))' if self.ema.warmup else 'off'})" if self.ema.enabled else "disabled",
            "Restore best at end"      : self.restore_best,
            "Resume"                   : self.resume,
            "Abort on non-finite loss" : self.abort_on_nonfinite_loss,
            "Debug logging"            : training.log_debug,
            "LR scale (batch-size rule)": f"x{self.config.optimizer.lr_scale:.4f} (from the micro-batch, ignores accumulation)",
            "Optimizer"                : self._optimizer_banner(),
            "SWA"                      : f"enabled (averaging from epoch {self.swa_start_epoch + 1})" if self.swa.enabled else "disabled",
            "VRAM reservation"         : f"{memory.reserve_vram} (keep free {memory.vram_keep_free_gb:g} GB)",
            "cudnn"                    : f"deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}",
        })

        self._log_param_groups()

    def _optimizer_banner(self) -> str:
        cfg  = self.config.optimizer
        kind = str(cfg.type)

        if kind in ("adamw", "adam"):
            return f"{kind} (default betas={tuple(cfg.betas)}, eps={cfg.eps:g}, weight_decay={cfg.weight_decay:g}; per-group values below take precedence)"

        return f"{kind} (momentum={cfg.momentum:g}, nesterov={cfg.nesterov}, weight_decay={cfg.weight_decay:g}; per-group values below take precedence)"

    def _log_param_groups(self) -> None:
        rows = []
        for i, pg in enumerate(self.optimizer.param_groups):
            rows.append({
                "Group"        : pg.get("name", str(i)),
                "Base LR"      : f"{self.base_lrs[i]:g}",
                "Step-0 LR"    : f"{pg['lr']:g}",
                "Weight decay" : f"{pg['weight_decay']:g}",
            })

        self.logger.metrics_table(rows, columns=["Group", "Base LR", "Step-0 LR", "Weight decay"], title="Optimizer Param Groups (base LR after lr_scale; step-0 LR includes the warmup start factor)")

    def _log_train_banner(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        self.logger.kv_table({
            "Train batches"        : len(train_loader),
            "Validation batches"   : len(val_loader),
            "Test batches"         : len(test_loader),
            "Micro-batch size"     : self.micro_batch,
            "Effective batch size" : f"{self.micro_batch * self.accumulation_steps} (micro-batch x {self.accumulation_steps} accumulation steps)",
            "Device"               : self.device,
        })

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _maybe_resume(self, loader_generator) -> int:
        if not self.resume:
            return 0

        start_epoch = TrainerState.restore(self, self.state_path, loader_generator)
        geometry    = TrainerState.batch_geometry(self)
        stopping    = self.early_stopping
        warmup      = self.warmup

        self.logger.section("[Resumed Trainer State]")
        self.logger.kv_table({
            "State path"     : str(self.state_path),
            "Next epoch"     : f"{start_epoch + 1}/{self.epochs}",
            "Global step"    : self.global_step,
            "Best val"       : f"{self.checkpoint.best_val_loss:.4f} @ epoch {self.checkpoint.best_epoch + 1}",
            "Batch geometry" : f"micro-batch {geometry['batch_size']}, {geometry['accumulation_steps']} accumulation step(s), lr_scale x{geometry['lr_scale']:.4g}",
            "Early stopping" : f"{stopping.counter}/{stopping.patience} validation rounds without improvement, triggered={stopping.triggered}",
            "Warmup"         : f"step {warmup.current_step}/{warmup.warmup_steps}, {'finished' if warmup.is_finished() else 'ramping'}",
            "Current LRs"    : ", ".join(f"{lr:.4g}" for lr in self.lr_scheduler.effective_lrs()),
        })

        if stopping.triggered:
            self.logger.warning(f"the restored state had already early-stopped at epoch {start_epoch} (best epoch {stopping.best_epoch + 1}); no further epoch is trained, the run goes straight to the final checkpoint, the best-weight restore and the test evaluation")

        return start_epoch

    def _apply_memory_format(self, batch):
        if not self.channels_last:
            return batch

        return type(batch)(item.contiguous(memory_format=torch.channels_last) if item.dim() == 4 else item for item in batch)

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

    def micro_step(self, batch, window_len: int) -> tuple[dict, float]:
        batch = self._apply_memory_format(batch)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            loss_dict = self._compute_loss(batch)

        loss = loss_dict["total_loss"].value / window_len

        if not torch.isfinite(loss):
            return loss_dict, float("nan")

        loss.backward()

        return loss_dict, float(loss.detach())

    def apply_gradients(self, global_step: int) -> float:
        grad_norm = self.grad_clipper.maybe_clip(self.model, global_step)
        self.grad_clipper.record(grad_norm, global_step)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return grad_norm

    def _log_ema_decay(self):
        if not self.ema.enabled or self.ema.last_decay == self._ema_decay_logged:
            return

        self._ema_decay_logged = self.ema.last_decay
        self.tracker.log_scalar("optim/ema_decay", self.ema.last_decay, self.global_step)

    def _log_throughput(self, epoch_time: float, data_wait: float, sample_count: int, nonfinite_count: int, epoch: int) -> None:
        elapsed = max(epoch_time, 1e-9)

        self.tracker.log_scalar("throughput/samples_per_s",    sample_count / elapsed, epoch)
        self.tracker.log_scalar("throughput/epoch_time_s",     epoch_time,             epoch)
        self.tracker.log_scalar("throughput/data_wait_frac",   data_wait / elapsed,    epoch)
        self.tracker.log_scalar("controls/nonfinite_batches",  float(nonfinite_count), epoch)

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        n          = 0
        aggregator = MetricAggregator()
        n_batches  = len(loader)

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
                sample_count += len(batch[0])

                self._update_optimizer(self.lr_scheduler.effective_lrs())

                window_start = (batch_idx // self.accumulation_steps) * self.accumulation_steps
                window_len   = min(self.accumulation_steps, n_batches - window_start)

                loss_dict, scaled = self.micro_step(batch, window_len)

                if math.isfinite(scaled):
                    window_has_grads = True

                    n += 1
                    aggregator.add(loss_dict)

                else:
                    if self.abort_on_nonfinite_loss:
                        raise FloatingPointError(f"Training loss is non-finite at step {self.global_step}{self._nonfinite_detail(loss_dict, batch)}")

                    nonfinite_count += 1
                    self.logger.warning(f"Training loss is non-finite at step {self.global_step}{self._nonfinite_detail(loss_dict, batch)}; skipping batch (abort_on_nonfinite_loss disabled).")

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                    if window_has_grads:
                        self.apply_gradients(self.global_step)
                        self.ema.update(self.model)
                        self._log_ema_decay()
                        self.warmup.step()

                    window_has_grads = False

                self.global_step += 1
                self.tracker.set_step(self.global_step)

                _prog.update(_task, advance=1)
                fetch_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start

        avg = aggregator.reduce_total() if n > 0 else float("nan")
        self.tracker.log_scalar("loss/train", avg, epoch)
        self._log_epoch_metrics("train", aggregator, epoch)
        self._log_throughput(epoch_time, data_wait, sample_count, nonfinite_count, epoch)

        self.tracker.log_memory(self.global_step)
        return avg

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage="val"):
        self.model.eval()

        batch_count, sample_count = 0, 0

        aggregator = MetricAggregator()

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]Eval {stage}[/section] - epoch {epoch + 1}/{self.epochs}", total=len(loader))

            for batch in loader:
                batch = self._apply_memory_format(batch)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    loss_dict = self._compute_loss(batch)
                loss = loss_dict["total_loss"].value

                if torch.isfinite(loss):
                    batch_count  += 1
                    sample_count += int(batch[0].shape[0])
                    aggregator.add(loss_dict)

                elif self.abort_on_nonfinite_loss:
                    raise FloatingPointError(f"Training {stage} loss is non-finite at epoch {epoch + 1}{self._nonfinite_detail(loss_dict, batch)}")

                else:
                    self.logger.warning(f"Training {stage} loss is non-finite at epoch {epoch + 1}{self._nonfinite_detail(loss_dict, batch)}; skipping batch (abort_on_nonfinite_loss disabled).")

                _prog.advance(_task)

        if sample_count == 0:
            self.logger.warning(f"Training {stage} epoch {epoch + 1}: no finite batch out of {len(loader)}; the {stage} loss is NaN")

        avg = aggregator.reduce_total() if sample_count > 0 else float("nan")
        self._log_epoch_metrics(stage, aggregator, epoch)

        return {"avg_loss": avg, "num_batches": int(batch_count), "num_samples": int(sample_count)}

    def _nonfinite_detail(self, loss_dict: dict, batch) -> str:
        bad      = [name for name, mean in loss_dict["channels"].items() if not torch.isfinite(mean.value)]
        named    = (("inputs", batch[0]), ("targets", batch[1]))
        bad_data = [name for name, tensor in named if not torch.isfinite(tensor).all()]
        verdict  = f"non-finite {' and '.join(bad_data)} in the batch" if bad_data else "inputs and targets finite, so the model or the loss produced it"

        return f"; non-finite target channels: {bad if bad else 'none (every channel loss is finite)'}; {verdict}"

    def _log_epoch_metrics(self, stage: str, aggregator: MetricAggregator, epoch: int) -> None:
        self.tracker.log_staged("loss_channels", aggregator.reduce_channels(), stage, epoch)

    def capture_state(self, epoch: int) -> dict:
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
        }

    def _maybe_snapshot(self, epoch: int) -> None:
        every = self.config.training.snapshot_every_n_epochs
        if every <= 0 or (epoch + 1) % every != 0:
            return

        path = Path(self.run_dir) / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"

        with self.ema.applied(self.model):
            TorchIO.save_atomic({"epoch": epoch, "params": self.model.state_dict()}, path)

        self.logger.subsection(f"Epoch snapshot -> {path}")

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        pass

    def _finalize_swa(self, train_loader: DataLoader, val_loader: DataLoader, last_epoch: int) -> float | None:
        if not self.swa.enabled or self.swa.n_averaged == 0:
            if self.swa.enabled:
                self.logger.warning(f"SWA averaged no epoch: the start epoch {self.swa_start_epoch + 1} lies past the last completed epoch {last_epoch + 1}; no swa_model.pt written")
            return None

        with self.swa.applied(self.model):
            refreshed = WeightSwa.refresh_bn(self.model, train_loader, self._compute_loss)
            swa_val   = self.evaluate(val_loader, last_epoch, stage="swa_val")
            payload   = self.capture_state(last_epoch)

        payload["swa_epochs"]   = self.swa.n_averaged
        payload["swa_val_loss"] = swa_val["avg_loss"]

        swa_path = Path(self.run_dir) / "swa_model.pt"
        TorchIO.save_atomic(payload, swa_path)

        self.tracker.log_scalar("loss/swa_val", swa_val["avg_loss"], last_epoch)
        self.logger.subsection(f"SWA : val loss={swa_val['avg_loss']:.4f} over {self.swa.n_averaged} averaged epochs ({refreshed} BN modules refreshed) -> {swa_path}")

        return float(swa_val["avg_loss"])

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
        self.logger.subsection(f"Test : loss={test['avg_loss']:.4f}  (batches={test['num_batches']}, samples={test['num_samples']}, weights: {weights})")

        return test

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.micro_batch = train_loader.batch_size

        self.logger.section("[Training Loop]")
        self._log_train_banner(train_loader, val_loader, test_loader)
        self.tracker.log_layout(ScalarsPage().loss().optimiser().layout())

        steps_per_epoch = math.ceil(len(train_loader) / self.accumulation_steps)
        self.warmup.resolve(self.epochs * steps_per_epoch)

        CompletionMarker.clear(self.run_dir)
        self._clear_cuda_cache()

        self.resource_monitor.start()
        try:
            loader_generator = train_loader.generator
            start_epoch      = self._maybe_resume(loader_generator)
            last_epoch       = max(start_epoch - 1, 0)
            stop             = self.early_stopping.triggered
            end_epoch        = start_epoch if stop else self.epochs

            self.docs.emit(train_loader, self.device)
            self.vram_reservation.fill()

            with self.logger.live_monitor("Training Progress") as live_mon:
                with self.logger.track(transient=False) as _prog_epochs:
                    _task_epochs = _prog_epochs.add_task("[section]Training[/section]", total=self.epochs, completed=start_epoch)

                    for epoch in range(start_epoch, end_epoch):
                        epoch_num  = epoch + 1
                        last_epoch = epoch

                        self.logger.section(f"[Epoch {epoch_num}/{self.epochs}]")

                        train_loss = self.train_epoch(train_loader, epoch)
                        self.logger.subsection(f"Train  : loss={train_loss:.4f}")

                        do_eval = (epoch_num % self.validation_frequency == 0) or (epoch_num == self.epochs)
                        if do_eval:
                            with self.ema.applied(self.model):
                                val      = self.evaluate(val_loader, epoch, stage="val")
                                val_loss = val["avg_loss"]
                                self.checkpoint.step(val_loss, epoch, self)

                            self.logger.subsection(f"Validation : loss={val_loss:.4f}  (batches={val['num_batches']}, samples={val['num_samples']})")
                            self.tracker.log_scalar("loss/val", val_loss, epoch)
                            self.lr_scheduler.step(epoch)
                            stop = self.early_stopping(val_loss, epoch)
                            self._trial_callback(val_loss, epoch)
                        else:
                            val_loss = float("nan")
                            self.lr_scheduler.step(epoch)
                            stop = False

                        self.train_losses.append(train_loss)
                        self.val_losses.append(val_loss)

                        effective_lrs = self.lr_scheduler.effective_lrs()
                        self._update_optimizer(effective_lrs)

                        if self.swa.enabled and epoch >= self.swa_start_epoch:
                            self.swa.update(self.model)

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

            swa_val_loss = self._finalize_swa(train_loader, val_loader, last_epoch)

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
                "swa_val_loss"     : swa_val_loss,
            })

            return self.train_losses, self.val_losses, self.checkpoint.best_val_loss
        finally:
            self.resource_monitor.stop()
