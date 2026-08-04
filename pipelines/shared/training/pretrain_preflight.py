from __future__ import annotations

from pathlib import Path
from typing  import Callable, Optional

import torch

from tools.monitoring.logger    import Logger
from tools.training.pretraining import PretrainContext, PretrainOrchestrator, TrainStepMemoryProbe, TrainerFeed


class PretrainPreflight:
    def __init__(self, pretrain_config, training_config, build_trainer: Callable[[Logger], tuple], run_directory: Path, label: Optional[str] = None) -> None:
        self.pretrain      = pretrain_config
        self.training      = training_config
        self.build_trainer = build_trainer
        self.run_directory = Path(run_directory)
        self.label         = label

    def _enabled(self) -> bool:
        return bool(self.pretrain.find_batch_size or self.pretrain.tune_loader)

    def _build_context(self, logger: Logger, device: torch.device) -> PretrainContext:
        trainer, dataset, model = self.build_trainer(logger)
        trainer.model.train()

        feed       = TrainerFeed(trainer)
        context_gb = TrainStepMemoryProbe.measure_context(device)

        return PretrainContext(
            dataset        = dataset,
            model          = model,
            to_model_input = feed.to_model_input,
            forward_loss   = feed.forward_loss,
            trial_step     = TrainStepMemoryProbe(trainer, dataset, self.pretrain.measure_steps, device, context_gb),
            device         = device,
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

        logger = Logger(log_dir=str(result_dir / "logs"), name="pretrain", level="INFO")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        orchestrator = PretrainOrchestrator(
            pretrain_config = self.pretrain,
            training_config = self.training,
            build_context   = lambda: self._build_context(logger, device),
            logger          = logger,
            label           = self.label,
        )

        try:
            orchestrator.run()
        finally:
            logger.close()
