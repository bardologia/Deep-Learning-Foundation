from __future__ import annotations

import gc
from dataclasses import dataclass
from typing      import Callable, Optional

import torch

from tools.monitoring.logger                 import Logger
from tools.training.pretraining.batch_finder import BatchSizeFinder
from tools.training.pretraining.loader_tuner import LoaderTuner


@dataclass
class PretrainContext:
    dataset        : object
    model          : object
    to_model_input : Callable
    forward_loss   : Callable
    trial_step     : Callable[[int], float]
    device         : torch.device
    use_amp        : bool                         = False
    context_gb     : float                        = 0.0
    on_oom         : Optional[Callable[[], None]] = None


class PretrainOrchestrator:
    def __init__(self, pretrain_config, training_config, build_context: Callable[[], PretrainContext], logger: Logger, label: Optional[str] = None) -> None:
        self.pretrain = pretrain_config
        self.training = training_config
        self.build    = build_context
        self.logger   = logger
        self.label    = label

    def _enabled(self) -> bool:
        return bool(self.pretrain.find_batch_size or self.pretrain.tune_loader)

    def _release_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_batch_size(self, context: PretrainContext) -> None:
        ceiling = min(self.pretrain.max_batch, len(context.dataset))
        if ceiling < self.pretrain.max_batch:
            self.logger.subsection(f"Ceiling lowered from {self.pretrain.max_batch} to {ceiling}: the dataset holds {len(context.dataset)} samples")

        finder = BatchSizeFinder(
            trial_step = context.trial_step,
            budget_gb  = self.pretrain.vram_budget_gb,
            ceiling    = ceiling,
            device     = context.device,
            logger     = self.logger,
            model_name = self.label,
            context_gb = context.context_gb,
            on_oom     = context.on_oom,
        )

        result = finder.run()

        if result["status"] != "PASS":
            raise SystemExit(f"max batch-size finder failed: {result['error']}")

        self.training.batch_size = int(result["batch_size"])
        self.logger.subsection(f"Resolved batch size: {self.training.batch_size} overrides training.batch_size for this run (peak {result['peak_gb']:.2f} GB, scale_lr_with_batch={self.training.scale_lr_with_batch})")

    def _tune_loader(self, context: PretrainContext) -> None:
        tuner = LoaderTuner(
            dataset          = context.dataset,
            model            = context.model,
            to_model_input   = context.to_model_input,
            forward_loss     = context.forward_loss,
            device           = context.device,
            logger           = self.logger,
            use_amp          = context.use_amp,
            seed             = self.pretrain.seed,
            warmup_batches   = self.pretrain.warmup_batches,
            timed_batches    = self.pretrain.timed_batches,
            worker_counts    = self.pretrain.worker_counts,
            prefetch_factors = self.pretrain.prefetch_factors,
            data_wait_target = self.pretrain.data_wait_target,
        )

        choice = tuner.run(self.training.batch_size)

        if choice is None:
            return

        self.training.num_workers     = int(choice["num_workers"])
        self.training.prefetch_factor = int(choice["prefetch_factor"])
        self.logger.subsection(f"Resolved loader: workers={self.training.num_workers} prefetch={self.training.prefetch_factor} override training.num_workers/prefetch_factor for this run (pin_memory recommendation {choice['pin_memory']} is not applied; the training loader always pins memory)")

    def run(self) -> None:
        if not self._enabled():
            return

        if self.pretrain.find_batch_size:
            self.logger.section("[Pretrain] Max batch-size finder")
            self.logger.subsection("Probe run: the dataset and model below belong to the batch-size probe, not the training run.")
            self.logger.kv_table({
                "VRAM budget"   : f"{self.pretrain.vram_budget_gb:g} GB",
                "Batch ceiling" : f"{self.pretrain.max_batch} (trials are powers of two up to the ceiling)",
                "Measure steps" : self.pretrain.measure_steps,
            })
            self._find_batch_size(self.build())
            self._release_cache()

        if self.pretrain.tune_loader:
            self.logger.section("[Pretrain] DataLoader tuner")
            self.logger.subsection("Probe run: the dataset and model below belong to the loader-timing probe, not the training run.")
            self.logger.kv_table({
                "Worker counts"    : list(self.pretrain.worker_counts),
                "Prefetch factors" : list(self.pretrain.prefetch_factors),
                "Warmup batches"   : self.pretrain.warmup_batches,
                "Timed batches"    : self.pretrain.timed_batches,
                "Data-wait target" : self.pretrain.data_wait_target,
                "Probe seed"       : self.pretrain.seed,
            })
            self._tune_loader(self.build())
            self._release_cache()
