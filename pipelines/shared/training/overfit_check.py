from __future__ import annotations

import copy
import math
import time
from dataclasses import fields
from pathlib import Path

from torch.utils.data import default_collate

from configuration.training.general.runtime import IOConfig
from tools.data.io                          import FileIO
from tools.runtime.completion               import CompletionMarker
from tools.runtime.reproducibility          import RngSnapshot


class OverfitCheck:
    REPORT_FILENAME = "overfit_report.json"

    def __init__(self, config, run_directory: Path, logger) -> None:
        self.config        = config
        self.run_directory = Path(run_directory)
        self.logger        = logger

        self.work_directory = self.run_directory / "overfit_check"
        self.report_path    = self.run_directory / "meta" / self.REPORT_FILENAME
        self.overrides      = {}
        self.rng            = RngSnapshot() if config.enabled else None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def epoch_steps(self) -> int:
        return min(self.config.steps_per_epoch, self.config.max_steps)

    @property
    def planned_epochs(self) -> int:
        return math.ceil(self.config.max_steps / self.epoch_steps)

    def announce(self) -> None:
        self.logger.section("[Overfit Gate]")
        self.logger.subsection("Gate run: the trainer banners below belong to the overfit sanity gate (sanitized throwaway config), not the training run.")

    def record(self, key: str, value) -> None:
        self.overrides[key] = value

    def sanitized_trainer_config(self, trainer_config):
        writer                   = trainer_config.io.writer
        trainer_config.io.writer = None
        try:
            cfg = copy.deepcopy(trainer_config)
        finally:
            trainer_config.io.writer = writer

        cfg.io = IOConfig(logdir=str(self.work_directory))

        cfg.training.epochs               = self.planned_epochs
        cfg.training.validation_frequency = self.planned_epochs
        cfg.training.use_ema              = False
        cfg.training.resume               = False

        cfg.optimizer.weight_decay      = 0.0
        cfg.warmup.warmup_enabled       = False
        cfg.scheduler.type              = "constant"
        cfg.early_stopping.restore_best = False
        cfg.resources.enabled           = False

        self.record("io.logdir",                    str(self.work_directory))
        self.record("training.epochs",              self.planned_epochs)
        self.record("training.validation_frequency", self.planned_epochs)
        self.record("training.use_ema",             False)
        self.record("training.resume",              False)
        self.record("optimizer.weight_decay",       0.0)
        self.record("warmup.warmup_enabled",        False)
        self.record("scheduler.type",               "constant")
        self.record("early_stopping.restore_best",  False)
        self.record("resources.enabled",            False)

        return cfg

    def sanitized_model_config(self, model_config):
        cfg = copy.deepcopy(model_config)

        for field_spec in fields(cfg):
            if field_spec.name.endswith("dropout") or field_spec.name.endswith("_wd") or field_spec.name.endswith("stochastic_depth_rate"):
                setattr(cfg, field_spec.name, 0.0)
                self.record(f"model.{field_spec.name}", 0.0)

        return cfg

    def _gate_batch(self, train_dataset):
        parts = getattr(train_dataset, "parts", None) or [train_dataset]
        saved = [(part, part.augmenter) for part in parts]

        for part in parts:
            part.augmenter = None

        try:
            n_examples = min(self.config.n_examples, len(train_dataset))
            indices    = [(i * len(train_dataset)) // n_examples for i in range(n_examples)]
            items      = [train_dataset[i] for i in indices]
        finally:
            for part, augmenter in saved:
                part.augmenter = augmenter

        self.record("augmentation", "disabled")

        return default_collate(items), indices

    def _verdict(self, train_losses: list[float]) -> dict:
        initial = float(train_losses[0])
        best    = float(min(train_losses))
        final   = float(train_losses[-1])
        ratio   = best / initial if initial > 0 else 0.0
        passed  = best <= self.config.stop_threshold or ratio <= self.config.pass_loss_ratio

        return {
            "passed"       : passed,
            "initial_loss" : initial,
            "best_loss"    : best,
            "final_loss"   : final,
            "loss_ratio"   : ratio,
        }

    def _write_report(self, verdict: dict, train_losses: list[float], indices: list[int], duration_s: float) -> None:
        report = {
            **verdict,
            "n_examples"          : len(indices),
            "example_indices"     : indices,
            "max_steps"           : self.config.max_steps,
            "steps_per_epoch"     : self.epoch_steps,
            "epochs_run"          : len(train_losses),
            "steps_run"           : len(train_losses) * self.epoch_steps,
            "pass_loss_ratio"     : self.config.pass_loss_ratio,
            "stop_threshold"      : self.config.stop_threshold,
            "epoch_losses"        : [float(loss) for loss in train_losses],
            "sanitized_overrides" : self.overrides,
            "duration_s"          : duration_s,
        }

        FileIO.save_json(report, self.report_path)

    def _cleanup(self) -> None:
        for name in ("best_model.pt", "last.pt", CompletionMarker.FILENAME):
            (self.work_directory / name).unlink(missing_ok=True)

        if self.work_directory.is_dir() and not any(self.work_directory.iterdir()):
            self.work_directory.rmdir()

    def _emit(self, verdict: dict) -> None:
        self.logger.section("[Overfit Check Verdict]")
        self.logger.kv_table({
            "Passed"       : verdict["passed"],
            "Initial Loss" : f"{verdict['initial_loss']:.6f}",
            "Best Loss"    : f"{verdict['best_loss']:.6f}",
            "Loss Ratio"   : f"{verdict['loss_ratio']:.4f}",
            "Pass Ratio"   : self.config.pass_loss_ratio,
            "Report"       : str(self.report_path),
        })

        if not verdict["passed"]:
            raise RuntimeError(f"Overfit check failed: best loss {verdict['best_loss']:.6f} vs initial {verdict['initial_loss']:.6f} (ratio {verdict['loss_ratio']:.4f} > {self.config.pass_loss_ratio}); the model cannot fit {self.config.n_examples} examples with regularization disabled, so normal training was aborted.")

        self.logger.subsection("Overfit check passed; continuing with normal training.")

    def run(self, trainer, train_dataset) -> dict:
        self.work_directory.mkdir(parents=True, exist_ok=True)

        start          = time.perf_counter()
        batch, indices = self._gate_batch(train_dataset)
        loader         = [batch] * self.epoch_steps

        self.logger.section("[Overfit Check]")
        self.logger.kv_table({
            "Examples"        : self.config.n_examples,
            "Max Steps"       : self.config.max_steps,
            "Steps per Epoch" : self.epoch_steps,
            "Pass Ratio"      : self.config.pass_loss_ratio,
            "Stop Threshold"  : self.config.stop_threshold,
            "Work Directory"  : str(self.work_directory),
            "Sanitized"       : ", ".join(sorted(self.overrides)),
        })

        try:
            train_losses, _, _ = trainer.train(loader, loader, loader)
        finally:
            self.rng.restore()

        verdict = self._verdict(train_losses)

        self._write_report(verdict, train_losses, indices, time.perf_counter() - start)
        self._cleanup()
        self._emit(verdict)

        return verdict
