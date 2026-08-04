from __future__ import annotations

import gc
import math
import os
import threading
import time
from dataclasses import dataclass
from typing      import Callable, Optional

import numpy as np
import pynvml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LoaderSpec:
    batch_size         : int
    num_workers        : int
    prefetch_factor    : int  = 4
    pin_memory         : bool = True
    persistent_workers : bool = True

    @property
    def label(self) -> str:
        return f"bs{self.batch_size}_w{self.num_workers}_pf{self.prefetch_factor}_pin{int(self.pin_memory)}_persist{int(self.persistent_workers)}"

    def as_record(self) -> dict:
        return {
            "batch_size"         : self.batch_size,
            "num_workers"        : self.num_workers,
            "prefetch_factor"    : self.prefetch_factor if self.num_workers > 0 else 0,
            "pin_memory"         : self.pin_memory,
            "persistent_workers" : self.persistent_workers and self.num_workers > 0,
        }


class GpuUtilizationSampler:
    def __init__(self, gpu_index: Optional[int] = None, interval_s: float = 0.05) -> None:
        self.interval_s = float(interval_s)
        self.gpu_index  = self._resolve_index(gpu_index)

        self._handle       = None
        self._nvml_ok      = False
        self._stop_evt     = threading.Event()
        self._thread       = None
        self._util_samples = []
        self._vram_samples = []

    @staticmethod
    def _resolve_index(gpu_index: Optional[int]) -> int:
        if gpu_index is not None:
            return int(gpu_index)

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            return int(visible.split(",")[0])

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())

        return 0

    def start(self) -> None:
        try:
            pynvml.nvmlInit()
            self._handle  = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_ok = True
        except Exception:
            self._nvml_ok = False
            return

        self._util_samples = []
        self._vram_samples = []
        self._stop_evt.clear()

        self._thread = threading.Thread(target=self._run, name="GpuUtilizationSampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._sample()
            self._stop_evt.wait(self.interval_s)

    def _sample(self) -> None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

            self._util_samples.append(float(util.gpu))
            self._vram_samples.append(float(mem.used) / (1024.0 ** 3))
        except Exception:
            pass

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=max(self.interval_s * 4, 1.0))
            self._thread = None

        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ok = False

        if not self._util_samples:
            return {"gpu_available": False, "gpu_util_mean": float("nan"), "gpu_util_max": float("nan"), "vram_peak_gb": float("nan"), "gpu_n_samples": 0}

        return {
            "gpu_available" : True,
            "gpu_util_mean" : float(np.mean(self._util_samples)),
            "gpu_util_max"  : float(np.max(self._util_samples)),
            "vram_peak_gb"  : float(np.max(self._vram_samples)) if self._vram_samples else float("nan"),
            "gpu_n_samples" : int(len(self._util_samples)),
        }


class GpuFeedBenchmark:
    def __init__(
        self,
        dataset        : Dataset,
        model          : nn.Module,
        to_model_input : Callable[[object, torch.device], torch.Tensor],
        forward_loss   : Callable[[nn.Module, torch.Tensor], torch.Tensor],
        device         : torch.device,
        use_amp        : bool          = False,
        seed           : int           = 0,
        warmup_batches : int           = 8,
        timed_batches  : int           = 60,
        gpu_index      : Optional[int] = None,
        cpu_threads    : Optional[int] = None,
    ) -> None:
        self.dataset        = dataset
        self.model          = model.to(device)
        self.to_model_input = to_model_input
        self.forward_loss   = forward_loss
        self.device         = device
        self.use_amp        = bool(use_amp)
        self.seed           = int(seed)
        self.warmup_batches = int(warmup_batches)
        self.timed_batches  = int(timed_batches)
        self.gpu_index      = gpu_index

        if cpu_threads is not None:
            torch.set_num_threads(max(1, int(cpu_threads)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    @staticmethod
    def _worker_init(worker_id: int) -> None:
        torch.set_num_threads(1)

    @staticmethod
    def _iterate(loader: DataLoader, n_batches: int):
        produced = 0
        while produced < n_batches:
            for batch in loader:
                yield batch
                produced += 1
                if produced >= n_batches:
                    return
            if len(loader) == 0:
                raise RuntimeError("DataLoader produced zero batches; dataset smaller than one batch with drop_last=True")

    def _build_loader(self, spec: LoaderSpec) -> DataLoader:
        generator   = torch.Generator()
        generator.manual_seed(self.seed)

        loader_kwargs = dict(
            batch_size  = spec.batch_size,
            num_workers = spec.num_workers,
            pin_memory  = spec.pin_memory,
            shuffle     = True,
            drop_last   = True,
            generator   = generator,
        )

        if spec.num_workers > 0:
            loader_kwargs["persistent_workers"] = spec.persistent_workers
            loader_kwargs["prefetch_factor"]    = spec.prefetch_factor
            loader_kwargs["worker_init_fn"]     = GpuFeedBenchmark._worker_init

        return DataLoader(self.dataset, **loader_kwargs)

    def _measure_loader_only(self, loader: DataLoader, spec: LoaderSpec) -> float:
        iterator = self._iterate(loader, self.warmup_batches + self.timed_batches)

        for _ in range(self.warmup_batches):
            next(iterator)

        start = time.perf_counter()
        count = 0
        for _ in range(self.timed_batches):
            next(iterator)
            count += 1
        elapsed = time.perf_counter() - start

        return count * spec.batch_size / max(elapsed, 1e-9)

    def _reference_batch(self, loader: DataLoader) -> torch.Tensor:
        batch = next(self._iterate(loader, 1))
        return self.to_model_input(batch, self.device)

    def _train_step(self, model_input: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            loss = self.forward_loss(self.model, model_input)

        loss.backward()
        self.optimizer.step()

        return loss

    def _measure_compute_ceiling(self, reference_input: torch.Tensor, spec: LoaderSpec) -> float:
        self.model.train()

        for _ in range(self.warmup_batches):
            self._train_step(reference_input)
        self._synchronize()

        start = time.perf_counter()
        for _ in range(self.timed_batches):
            self._train_step(reference_input)
        self._synchronize()
        elapsed = time.perf_counter() - start

        return self.timed_batches * spec.batch_size / max(elapsed, 1e-9)

    def _measure_end_to_end(self, loader: DataLoader, spec: LoaderSpec) -> dict:
        self.model.train()

        sampler = GpuUtilizationSampler(gpu_index=self.gpu_index)
        sampler.start()

        self._synchronize()

        data_seconds    = 0.0
        compute_seconds = 0.0
        counted         = 0
        index           = 0
        previous        = time.perf_counter()

        for batch in self._iterate(loader, self.warmup_batches + self.timed_batches):
            after_fetch  = time.perf_counter()
            model_input  = self.to_model_input(batch, self.device)
            self._train_step(model_input)
            self._synchronize()
            after_step   = time.perf_counter()

            if index >= self.warmup_batches:
                data_seconds    += after_fetch - previous
                compute_seconds += after_step - after_fetch
                counted         += 1

            index   += 1
            previous = time.perf_counter()

        gpu_summary = sampler.stop()

        total_seconds = data_seconds + compute_seconds
        throughput    = counted * spec.batch_size / max(total_seconds, 1e-9)
        wait_fraction = data_seconds / max(total_seconds, 1e-9)

        return {
            "end_to_end_samples_per_s" : throughput,
            "data_wait_fraction"       : wait_fraction,
            **gpu_summary,
        }

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, spec: LoaderSpec) -> dict:
        loader = self._build_loader(spec)

        loader_only     = self._measure_loader_only(loader, spec)
        reference       = self._reference_batch(loader)
        compute_ceiling = self._measure_compute_ceiling(reference, spec)
        end_to_end      = self._measure_end_to_end(loader, spec)

        feed_ratio = loader_only / max(compute_ceiling, 1e-9)
        efficiency = end_to_end["end_to_end_samples_per_s"] / max(compute_ceiling, 1e-9)

        record = {
            **spec.as_record(),
            "loader_only_samples_per_s"     : loader_only,
            "compute_ceiling_samples_per_s" : compute_ceiling,
            "feed_ratio"                    : feed_ratio,
            "compute_efficiency"            : efficiency,
            **end_to_end,
        }

        del reference
        del loader
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return record


class DataLoaderSweep:
    def __init__(self, benchmark: GpuFeedBenchmark, specs: list[LoaderSpec], on_result: Optional[Callable[[dict], None]] = None) -> None:
        self.benchmark = benchmark
        self.specs     = list(specs)
        self.on_result = on_result

    def _measure_spec(self, spec: LoaderSpec) -> Optional[dict]:
        try:
            record = self.benchmark.measure(spec)
            record["status"] = "ok"
            return record
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {**spec.as_record(), "status": "oom"}
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                return {**spec.as_record(), "status": "oom"}
            raise

    def run(self) -> list[dict]:
        results = []

        for spec in self.specs:
            record = self._measure_spec(spec)
            if record is None:
                continue

            results.append(record)
            if self.on_result is not None:
                self.on_result(record)

        return results
