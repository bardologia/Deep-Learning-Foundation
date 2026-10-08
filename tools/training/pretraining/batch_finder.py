"""Measurement of per-batch VRAM peaks and the largest batch fitting a memory budget."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from tools.monitoring.logger import Logger


class TrainStepMemoryProbe:
    """Measures the peak GPU memory of real training steps at a given batch size.

    Full forward, backward and optimizer steps are run, because surrogate probes
    understate the true peak by roughly an order of magnitude.

    Attributes:
        CONTEXT_WARN_GB : Fixed device memory in GB above which the measured context exceeds a bare CUDA context, meaning other processes hold memory on the device.
        trainer         : Trainer supplying the optimizer, the loss and the AMP setting.
        dataset         : Dataset the probe batches are drawn from.
        measure_steps   : Number of training steps taken per trial.
        device          : CUDA device the peak is measured on.
        context_gb      : Fixed device memory in GB already held outside the probe, added to every reported peak.
    """

    CONTEXT_WARN_GB = 1.5

    def __init__(self, trainer, dataset: Dataset, measure_steps: int, device: torch.device, context_gb: float = 0.0) -> None:
        """Stores the probe inputs.

        Args:
            trainer       : Trainer whose optimizer and loss are exercised.
            dataset       : Dataset the probe batches come from.
            measure_steps : Training steps per trial, at least one.
            device        : CUDA device to measure.
            context_gb    : Fixed device memory in GB added to each measured peak.

        Raises:
            ValueError : If measure_steps is below one, since no allocation peak would be reached.
        """
        if int(measure_steps) < 1:
            raise ValueError(f"measure_steps={measure_steps} is below 1; the memory probe needs at least one training step to reach a peak allocation")

        self.trainer       = trainer
        self.dataset       = dataset
        self.measure_steps = int(measure_steps)
        self.device        = device
        self.context_gb    = float(context_gb)

    @staticmethod
    def measure_context(device: torch.device) -> float:
        """Returns the device memory in GB already held by other processes and the driver.

        Args:
            device : Device to inspect; non-CUDA devices report 0.0.

        Returns:
            Fixed occupied memory in GB, read as total minus free.

        Raises:
            RuntimeError : If this process already allocated memory on the device, since those bytes would then be counted twice.
        """
        if device.type != "cuda":
            return 0.0

        allocated_gb = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
        if allocated_gb > 0.0:
            raise RuntimeError(f"{allocated_gb:.3f} GB is already allocated on {device}; the fixed context is read as (total - free) and must be measured before any model or batch reaches the device, or those bytes are counted a second time in the probe's peak and the batch-size ladder stops early")

        warm = torch.zeros(1, device=device)
        del warm
        torch.cuda.empty_cache()

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

        return (total_bytes - free_bytes) / (1024.0 ** 3)

    def __call__(self, batch_size: int) -> float:
        """Runs the measured training steps at a batch size and returns the peak memory.

        Args:
            batch_size : Batch size of the probe data loader.

        Returns:
            Peak reserved memory in GB, including the fixed context.

        Raises:
            RuntimeError : If the dataset holds fewer samples than one full batch.
        """
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
    """Searches powers of two for the largest batch whose measured peak fits a VRAM budget.

    Attributes:
        trial_step : Callable returning the peak memory in GB for a batch size.
        budget_gb  : VRAM budget in GB a trial must stay within.
        ceiling    : Largest batch size considered.
        device     : Device the trials run on.
        logger     : Logger receiving one line per trial.
        model_name : Label of the probed model, reported in the result.
        context_gb : Fixed device memory in GB reported alongside the result.
        on_oom     : Callback invoked when a trial raises an out-of-memory error.
    """

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
        """Stores the search settings.

        Args:
            trial_step : Callable mapping a batch size to its peak memory in GB.
            budget_gb  : VRAM budget in GB.
            ceiling    : Largest batch size to try.
            device     : Device the trials run on.
            logger     : Logger for the per-trial lines.
            model_name : Label of the probed model.
            context_gb : Fixed device memory in GB recorded in the result.
            on_oom     : Callback invoked on an out-of-memory trial, or None.
        """
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
        """Returns the powers of two from 1 up to and including the ceiling."""
        sizes = []
        size  = 1

        while size <= ceiling:
            sizes.append(size)
            size *= 2

        return sizes

    def _record(self, batch_size: int, peak_gb: Optional[float], status: str) -> dict:
        """Returns one trial entry holding the batch size, peak memory and FIT/OVER/OOM status."""
        return {"batch_size": batch_size, "peak_gb": peak_gb, "status": status}

    def run(self) -> dict:
        """Runs the ladder of batch sizes and reports the largest one within budget.

        The ladder stops at the first trial that exceeds the budget or raises an
        out-of-memory error.

        Returns:
            Result dictionary with the model label, PASS/FAIL status, the chosen
            batch size and its peak memory in GB, the budget, ceiling and fixed
            context, the per-trial records, and an error message when no batch
            size fits.
        """
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
