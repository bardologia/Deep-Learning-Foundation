
from __future__ import annotations

from pathlib import Path
from typing  import Callable, Optional

import os

import torch

from tools.monitoring.logger    import Logger
from tools.training.checkpoint  import TrainerState
from tools.training.pretraining import PretrainContext, PretrainOrchestrator, TrainStepMemoryProbe, TrainerFeed


class PretrainPreflight:

    def __init__(self, pretrain_config, training_config, build_trainer: Callable[[Logger], tuple], run_directory: Path, label: Optional[str] = None) -> None:
        self.pretrain      = pretrain_config
        self.training      = training_config
        self.build_trainer = build_trainer
        self.run_directory = Path(run_directory)
        self.label         = label

    def _enabled_probes(self) -> list[str]:
        flags = {"find_batch_size": self.pretrain.find_batch_size, "tune_loader": self.pretrain.tune_loader, "find_lr": self.pretrain.find_lr}

        return [name for name, enabled in flags.items() if enabled]

    def _enabled(self) -> bool:
        return bool(self._enabled_probes())

    def _refuse_on_resume(self) -> None:
        if not self.training.resume:
            return

        probes = ", ".join(self._enabled_probes())

        raise RuntimeError(f"pretraining probes ({probes}) cannot run on a run resuming from {TrainerState.FILENAME}: they would rewrite the recorded training.batch_size={self.training.batch_size}, num_workers/prefetch_factor or lr_multiplier under the restored optimizer, scheduler and warmup state. A run that probed records its probes switched off; pass --pretrain.<probe> false for each probe typed or recorded before that, or start a fresh run")

    def _warn_occupied_device(self, logger: Logger, context_gb: float) -> None:
        if context_gb <= TrainStepMemoryProbe.CONTEXT_WARN_GB:
            return

        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")

        logger.warning(f"CUDA context {context_gb:.2f} GB on CUDA_VISIBLE_DEVICES={mask} exceeds a bare context ({TrainStepMemoryProbe.CONTEXT_WARN_GB:g} GB): other processes are using this GPU and their memory counts against the probe budget, so the batch-size finder can settle on a smaller training.batch_size, and with scale_lr_with_batch={self.training.scale_lr_with_batch} a smaller learning-rate scale, than on a free GPU")

    def _build_context(self, logger: Logger, device: torch.device, context_gb: float) -> PretrainContext:
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
        trainer.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> None:
        if not self._enabled():
            return

        result_dir = self.run_directory / "pretrain"
        result_dir.mkdir(parents=True, exist_ok=True)

        with Logger(log_dir=str(result_dir / "logs"), name="pretrain", level="INFO") as logger:
            self._refuse_on_resume()

            device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            context_gb = TrainStepMemoryProbe.measure_context(device)

            logger.subsection(f"CUDA context before the probe builds anything: {context_gb:.2f} GB")
            self._warn_occupied_device(logger, context_gb)

            orchestrator = PretrainOrchestrator(
                pretrain_config = self.pretrain,
                training_config = self.training,
                build_context   = lambda: self._build_context(logger, device, context_gb),
                logger          = logger,
                result_dir      = result_dir,
                label           = self.label,
            )

            orchestrator.run()
