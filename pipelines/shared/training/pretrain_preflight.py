"""Pre-training preflight that tunes batch size, loader settings and the learning rate.

Wraps the shared pretraining orchestrator: it builds a throwaway trainer,
measures the CUDA context footprint and probes real training steps to pick a
batch size, data-loader configuration and learning-rate multiplier before the
run starts.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Callable, Optional

import torch

from tools.monitoring.logger    import Logger
from tools.training.checkpoint  import TrainerState
from tools.training.pretraining import PretrainContext, PretrainOrchestrator, TrainStepMemoryProbe, TrainerFeed


class PretrainPreflight:
    """Runs batch-size search, loader tuning and the LR range test ahead of a training run.

    Attributes:
        pretrain      : Preflight settings (which probes to run, measurement steps).
        training      : Training configuration whose batch size and loader fields the preflight updates.
        build_trainer : Callable taking a logger and returning (trainer, dataset, model) for the probe.
        run_directory : Run directory under which preflight output is written.
        label         : Optional label identifying the run in the preflight logs.
    """

    def __init__(self, pretrain_config, training_config, build_trainer: Callable[[Logger], tuple], run_directory: Path, label: Optional[str] = None) -> None:
        """Stores the preflight configuration and the trainer factory.

        Args:
            pretrain_config : Preflight settings.
            training_config : Training configuration to tune.
            build_trainer   : Callable taking a logger and returning (trainer, dataset, model).
            run_directory   : Run directory for preflight output.
            label           : Optional run label used in the preflight logs.
        """
        self.pretrain      = pretrain_config
        self.training      = training_config
        self.build_trainer = build_trainer
        self.run_directory = Path(run_directory)
        self.label         = label

    def _enabled_probes(self) -> list[str]:
        """Returns the names of the probes switched on, in the order they run."""
        flags = {"find_batch_size": self.pretrain.find_batch_size, "tune_loader": self.pretrain.tune_loader, "find_lr": self.pretrain.find_lr}

        return [name for name, enabled in flags.items() if enabled]

    def _enabled(self) -> bool:
        """Returns whether any of the batch-size, loader or LR probes is on."""
        return bool(self._enabled_probes())

    def _refuse_on_resume(self) -> None:
        """Refuses to probe a run that continues from its trainer state.

        Raises:
            RuntimeError : If the trainer resumes while a probe is on, since the probe would rewrite the recorded batch size, loader settings or learning-rate multiplier under the restored optimizer, scheduler and warmup state.
        """
        if not self.training.resume:
            return

        probes = ", ".join(self._enabled_probes())

        raise RuntimeError(f"pretraining probes ({probes}) cannot run on a run resuming from {TrainerState.FILENAME}: they would rewrite the recorded training.batch_size={self.training.batch_size}, num_workers/prefetch_factor or lr_multiplier under the restored optimizer, scheduler and warmup state. A run that probed records its probes switched off; pass --pretrain.<probe> false for each probe typed or recorded before that, or start a fresh run")

    def _build_context(self, logger: Logger, device: torch.device, context_gb: float) -> PretrainContext:
        """Builds the probe context around a freshly constructed trainer.

        Args:
            logger     : Logger the probe trainer writes into.
            device     : Device the probe steps run on.
            context_gb : CUDA context footprint measured before anything was built, in gigabytes.

        Returns:
            The PretrainContext wiring the model, dataset, forward/loss hooks
            and the memory-probing trial step.
        """
        trainer, dataset, model = self.build_trainer(logger)
        trainer.model.train()

        feed = TrainerFeed(trainer)

        return PretrainContext(
            dataset        = dataset,
            model          = model,
            to_model_input = feed.to_model_input,
            forward_loss   = feed.forward_loss,
            trial_step     = TrainStepMemoryProbe(trainer, dataset, self.pretrain.measure_steps, device, context_gb),
            device         = device,
            trainer        = trainer,
            use_amp        = trainer.use_amp,
            context_gb     = context_gb,
            on_oom         = lambda: self._release(trainer),
        )

    def _release(self, trainer) -> None:
        """Frees gradients and the CUDA cache after an out-of-memory probe step."""
        trainer.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> None:
        """Runs the preflight probes, writing their log and evidence records under ``<run>/pretrain``.

        Returns immediately when none of the batch-size search, loader tuning
        and LR range test is enabled. The preflight log appends across
        launches and closes with an ABORTED banner naming the error when a
        probe fails or the run turns out to be a resume.

        Raises:
            RuntimeError : If the trainer resumes while a probe is on, or a probe fails; propagated with its message recorded in the preflight log.
        """
        if not self._enabled():
            return

        result_dir = self.run_directory / "pretrain"
        result_dir.mkdir(parents=True, exist_ok=True)

        with Logger(log_dir=str(result_dir / "logs"), name="pretrain", level="INFO") as logger:
            self._refuse_on_resume()

            device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            context_gb = TrainStepMemoryProbe.measure_context(device)

            logger.subsection(f"CUDA context before the probe builds anything: {context_gb:.2f} GB")

            orchestrator = PretrainOrchestrator(
                pretrain_config = self.pretrain,
                training_config = self.training,
                build_context   = lambda: self._build_context(logger, device, context_gb),
                logger          = logger,
                result_dir      = result_dir,
                label           = self.label,
            )

            orchestrator.run()
