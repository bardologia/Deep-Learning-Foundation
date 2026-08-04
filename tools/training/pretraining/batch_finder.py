from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from tools.monitoring.logger import Logger


class TrainStepMemoryProbe:
    def __init__(self, trainer, dataset: Dataset, measure_steps: int, device: torch.device, context_gb: float = 0.0) -> None:
        self.trainer       = trainer
        self.dataset       = dataset
        self.measure_steps = int(measure_steps)
        self.device        = device
        self.context_gb    = float(context_gb)

    @staticmethod
    def measure_context(device: torch.device) -> float:
        if device.type != "cuda":
            return 0.0

        warm = torch.zeros(1, device=device)
        del warm
        torch.cuda.empty_cache()

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

        return (total_bytes - free_bytes) / (1024.0 ** 3)

    def __call__(self, batch_size: int) -> float:
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
        if len(loader) == 0:
            raise RuntimeError(f"dataset holds {len(self.dataset)} samples, fewer than one full batch of {batch_size} with drop_last=True")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        iterator = iter(loader)

        for _ in range(self.measure_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch    = next(iterator)

            self.trainer.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.trainer.use_amp):
                loss = self.trainer._compute_loss(batch)["total_loss"]

            loss.backward()
            self.trainer.optimizer.step()

        peak_reserved = torch.cuda.max_memory_reserved(self.device)

        del loss, batch, iterator, loader

        return self.context_gb + peak_reserved / (1024.0 ** 3)


class BatchSizeFinder:
    def __init__(
        self,
        trial_step : Callable[[int], float],
        budget_gb  : float,
        ceiling    : int,
        device     : torch.device,
        logger     : Logger,
        model_name : Optional[str]                = None,
        context_gb : float                        = 0.0,
        on_oom     : Optional[Callable[[], None]] = None,
    ) -> None:
        self.trial_step = trial_step
        self.budget_gb  = float(budget_gb)
        self.ceiling    = int(ceiling)
        self.device     = device
        self.logger     = logger
        self.model_name = model_name
        self.context_gb = float(context_gb)
        self.on_oom     = on_oom

    @staticmethod
    def candidates(ceiling: int) -> list[int]:
        sizes = []
        size  = 1

        while size <= ceiling:
            sizes.append(size)
            size *= 2

        return sizes

    def _record(self, batch_size: int, peak_gb: Optional[float], status: str) -> dict:
        return {"batch_size": batch_size, "peak_gb": peak_gb, "status": status}

    def run(self) -> dict:
        result = {
            "model"      : self.model_name,
            "status"     : None,
            "batch_size" : None,
            "peak_gb"    : None,
            "budget_gb"  : self.budget_gb,
            "ceiling"    : self.ceiling,
            "context_gb" : self.context_gb,
            "trials"     : [],
            "error"      : None,
        }

        best = None

        for batch_size in self.candidates(self.ceiling):
            try:
                peak_gb = self.trial_step(batch_size)
            except torch.cuda.OutOfMemoryError:
                if self.on_oom is not None:
                    self.on_oom()
                result["trials"].append(self._record(batch_size, None, "OOM"))
                self.logger.subsection(f"batch {batch_size:>5} -> OOM")
                break

            fits = peak_gb <= self.budget_gb
            result["trials"].append(self._record(batch_size, peak_gb, "FIT" if fits else "OVER"))
            self.logger.subsection(f"batch {batch_size:>5} -> {peak_gb:.2f} GB {'FIT' if fits else 'OVER'}")

            if not fits:
                break

            best = (batch_size, peak_gb)

        if best is None:
            result["status"] = "FAIL"
            result["error"]  = f"batch size 1 exceeds the {self.budget_gb:g} GB budget"
        else:
            result["batch_size"], result["peak_gb"] = best
            result["status"]                        = "PASS"

        return result
