"""Pre-training sanity gate that overfits a handful of examples.

Trains the model repeatedly on one fixed collated batch with regularization,
augmentation, EMA and scheduling stripped out. A model that cannot drive the
loss down on that batch is broken, so the gate aborts the real training run and
writes a JSON verdict report into the run's ``meta`` directory.
"""

from __future__ import annotations

import copy
import math
import shutil
import time
from dataclasses import fields
from pathlib import Path

from torch.utils.data import default_collate

from configuration.training.general.runtime import IOConfig
from tools.data.io                          import FileIO
from tools.runtime.reproducibility          import RngSnapshot


class OverfitCheck:
    """Runs the overfit sanity gate and reports its verdict.

    Attributes:
        config         : Overfit-check settings (example count, step budget, pass thresholds).
        run_directory  : Directory of the training run being gated.
        logger         : Logger receiving the gate banners and verdict table.
        work_directory : Throwaway directory holding the gate's own trainer output.
        report_path    : Path of the JSON verdict report.
        overrides      : Sanitizations applied to the throwaway configs, recorded for the report.
        rng            : Snapshot of the RNG state restored after the gate, or None when the gate is disabled.
    """

    REPORT_FILENAME = "overfit_report.json"

    def __init__(self, config, run_directory: Path, logger) -> None:
        """Prepares the gate's paths and snapshots the RNG state when enabled.

        Args:
            config        : Overfit-check settings.
            run_directory : Directory of the training run being gated.
            logger        : Logger receiving the gate output.
        """
        self.config        = config
        self.run_directory = Path(run_directory)
        self.logger        = logger

        self.work_directory = self.run_directory / "overfit_check"
        self.report_path    = self.run_directory / "meta" / self.REPORT_FILENAME
        self.overrides      = {}
        self.rng            = RngSnapshot() if config.enabled else None

    @property
    def enabled(self) -> bool:
        """Whether the overfit gate is switched on."""
        return bool(self.config.enabled)

    @property
    def epoch_steps(self) -> int:
        """Number of repeated batch steps making up one gate epoch."""
        return min(self.config.steps_per_epoch, self.config.max_steps)

    @property
    def planned_epochs(self) -> int:
        """Number of gate epochs needed to cover the configured step budget."""
        return math.ceil(self.config.max_steps / self.epoch_steps)

    def announce(self) -> None:
        """Logs the banner clarifying that the trainer output below is the gate."""
        self.logger.section("[Overfit Gate]")
        self.logger.subsection("Gate run: the trainer banners below belong to the overfit sanity gate (sanitized throwaway config), not the training run.")

    def record(self, key: str, value) -> None:
        """Records one sanitization applied to a throwaway config, for the report."""
        self.overrides[key] = value

    def sanitized_trainer_config(self, trainer_config):
        """Returns a gate copy of the trainer config with regularization removed.

        Weight decay, warmup, LR scheduling, EMA, resume, best-checkpoint
        restore and resource monitoring are all switched off, the epoch budget
        is set to the planned gate epochs, and the log directory is redirected
        to the throwaway work directory.

        Args:
            trainer_config : Trainer configuration of the real training run.

        Returns:
            A deep copy of the trainer config sanitized for the gate.
        """
        cfg    = copy.deepcopy(trainer_config)
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
        """Returns a gate copy of the model config with every regularizer zeroed.

        Args:
            model_config : Model configuration dataclass of the real run.

        Returns:
            A deep copy in which all fields whose names end in ``dropout``,
            ``_wd`` or ``stochastic_depth_rate`` are set to zero.
        """
        cfg = copy.deepcopy(model_config)

        for field_spec in fields(cfg):
            if field_spec.name.endswith("dropout") or field_spec.name.endswith("_wd") or field_spec.name.endswith("stochastic_depth_rate"):
                setattr(cfg, field_spec.name, 0.0)
                self.record(f"model.{field_spec.name}", 0.0)

        return cfg

    def _gate_batch(self, train_dataset):
        """Collates the fixed gate batch from evenly spaced training examples.

        Augmenters on every dataset part are disabled while the examples are
        drawn and restored afterwards.

        Args:
            train_dataset : Training dataset, optionally a concatenation exposing its constituent ``parts``.

        Returns:
            Tuple of the default-collated batch and the dataset indices it was
            drawn from.
        """
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
        """Judges the gate from its per-epoch training losses.

        Args:
            train_losses : Mean training loss of each gate epoch, in order.

        Returns:
            Dict with the pass flag and the initial, best and final losses plus
            the best-to-initial loss ratio.

        Raises:
            RuntimeError : If the initial loss is not positive, which leaves the loss-reduction ratio undefined.
        """
        initial = float(train_losses[0])
        best    = float(min(train_losses))
        final   = float(train_losses[-1])
        if initial <= 0.0:
            raise RuntimeError(f"Overfit check is inconclusive: the initial epoch loss is {initial:.6f}, which is not positive, so the loss-reduction ratio is undefined. A zero or negative starting loss means the forward pass or the loss definition is broken; fix that before training.")

        ratio  = best / initial
        passed = best <= self.config.stop_threshold or ratio <= self.config.pass_loss_ratio

        return {
            "passed"       : passed,
            "initial_loss" : initial,
            "best_loss"    : best,
            "final_loss"   : final,
            "loss_ratio"   : ratio,
        }

    def _write_report(self, verdict: dict, train_losses: list[float], indices: list[int], duration_s: float) -> None:
        """Writes the gate report JSON next to the run's metadata.

        Args:
            verdict      : Verdict dict produced by ``_verdict``.
            train_losses : Mean training loss of each gate epoch, in order.
            indices      : Dataset indices making up the fixed gate batch.
            duration_s   : Wall-clock duration of the gate, in seconds.
        """
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
        """Deletes the gate's throwaway work directory with every trainer output inside it."""
        if self.work_directory.is_dir():
            shutil.rmtree(self.work_directory)

    def _emit(self, verdict: dict) -> None:
        """Logs the verdict table and aborts training when the gate failed.

        Args:
            verdict : Verdict dict produced by ``_verdict``.

        Raises:
            RuntimeError : If the gate did not pass, which aborts the run.
        """
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
        """Runs the gate end to end and returns its verdict.

        Trains on the repeated fixed batch, restores the RNG state, writes the
        report, removes the throwaway artifacts and emits the verdict.

        Args:
            trainer       : Trainer built from the sanitized configs; its ``train`` method is called with the repeated gate batch as train, val and test loader.
            train_dataset : Training dataset the gate batch is drawn from.

        Returns:
            The verdict dict produced by ``_verdict``.

        Raises:
            RuntimeError : If the gate is inconclusive or does not pass.
        """
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
