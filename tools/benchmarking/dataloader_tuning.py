"""Measurement of how well a DataLoader configuration keeps the GPU fed.

Times the loader in isolation, the GPU compute ceiling on a fixed batch and the
end-to-end training step, while sampling NVML utilisation, so that a sweep over
batch sizes and worker counts can be judged as CPU-bound or GPU-bound.
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing      import Callable, Optional

import numpy as np
import pynvml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tools.monitoring.logger import Logger


@dataclass(frozen=True)
class LoaderSpec:
    """One DataLoader configuration under test.

    Attributes:
        batch_size         : Samples per batch.
        num_workers        : Worker processes feeding the loader.
        prefetch_factor    : Batches prefetched per worker; ignored when num_workers is 0.
        pin_memory         : Whether host batches are pinned before transfer.
        persistent_workers : Whether workers survive between epochs.
    """
    batch_size         : int
    num_workers        : int
    prefetch_factor    : int  = 4
    pin_memory         : bool = True
    persistent_workers : bool = True

    @property
    def label(self) -> str:
        """Filesystem-safe label encoding every field of the spec."""
        return f"bs{self.batch_size}_w{self.num_workers}_pf{self.prefetch_factor}_pin{int(self.pin_memory)}_persist{int(self.persistent_workers)}"

    def as_record(self) -> dict:
        """Returns the spec as a record, zeroing the fields that workerless loaders ignore."""
        return {
            "batch_size"         : self.batch_size,
            "num_workers"        : self.num_workers,
            "prefetch_factor"    : self.prefetch_factor if self.num_workers > 0 else 0,
            "pin_memory"         : self.pin_memory,
            "persistent_workers" : self.persistent_workers and self.num_workers > 0,
        }


class GpuUtilizationSampler:
    """Background sampler of GPU utilisation and VRAM use through NVML.

    NVML numbers the physical devices, while torch numbers the ones
    CUDA_VISIBLE_DEVICES exposes, possibly a share of the machine and in
    another order; without an explicit NVML index the sampler therefore
    finds the device torch computes on by its UUID, which both name alike.

    Attributes:
        logger     : Logger receiving NVML warnings.
        interval_s : Seconds between samples.
        gpu_index  : NVML index of the sampled device; None until start() has found the device torch computes on.
    """
    def __init__(self, logger: Logger, gpu_index: Optional[int] = None, interval_s: float = 0.05) -> None:
        """Prepares a sampler for one device without starting it.

        Args:
            logger     : Logger receiving NVML warnings.
            gpu_index  : NVML device index, or None to sample the CUDA device torch computes on.
            interval_s : Seconds between samples.
        """
        self.logger     = logger
        self.interval_s = float(interval_s)
        self.gpu_index  = None if gpu_index is None else int(gpu_index)

        self._handle        = None
        self._nvml_ok       = False
        self._nvml_started  = False
        self._stop_evt      = threading.Event()
        self._thread        = None
        self._util_samples  = []
        self._vram_samples  = []
        self._failed_reads  = 0
        self._first_failure = None

    @staticmethod
    def torch_uuid() -> str:
        """Returns the NVML UUID of the CUDA device torch computes on."""
        return f"GPU-{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}"

    @classmethod
    def nvml_device(cls) -> dict:
        """Returns the NVML index, name and UUID of the CUDA device torch computes on.

        Raises:
            pynvml.NVMLError : If NVML cannot be initialised or does not know the device.
        """
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(cls.torch_uuid())
            return {"index": int(pynvml.nvmlDeviceGetIndex(handle)), "name": str(pynvml.nvmlDeviceGetName(handle)), "uuid": cls.torch_uuid()}
        finally:
            pynvml.nvmlShutdown()

    def _label(self) -> str:
        """Returns the sampled device as named in the warnings."""
        return "the CUDA device in use" if self.gpu_index is None else f"GPU {self.gpu_index}"

    def start(self) -> None:
        """Starts NVML and the sampling thread, warning and giving up if there is no CUDA device or NVML is unavailable."""
        if self.gpu_index is None and not torch.cuda.is_available():
            self.logger.warning("[GpuUtilizationSampler] No CUDA device is visible and no NVML index was given; this sweep entry reports no GPU utilisation")
            return

        try:
            pynvml.nvmlInit()
            if self.gpu_index is None:
                self._handle   = pynvml.nvmlDeviceGetHandleByUUID(self.torch_uuid())
                self.gpu_index = int(pynvml.nvmlDeviceGetIndex(self._handle))
            else:
                self._handle   = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_ok      = True
            self._nvml_started = True
        except Exception as error:
            self._nvml_ok = False
            self.logger.warning(f"[GpuUtilizationSampler] NVML unavailable for {self._label()} ({type(error).__name__}: {error}); this sweep entry reports no GPU utilisation")
            return

        self._util_samples  = []
        self._vram_samples  = []
        self._failed_reads  = 0
        self._first_failure = None
        self._stop_evt.clear()

        self._thread = threading.Thread(target=self._run, name="GpuUtilizationSampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Samples until the stop event is set."""
        while not self._stop_evt.is_set():
            self._sample()
            self._stop_evt.wait(self.interval_s)

    def _sample(self) -> None:
        """Appends one utilisation percentage and VRAM reading, counting failed reads."""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

            self._util_samples.append(float(util.gpu))
            self._vram_samples.append(float(mem.used) / (1024.0 ** 3))
        except Exception as error:
            self._failed_reads += 1
            if self._first_failure is None:
                self._first_failure = f"{type(error).__name__}: {error}"

    def _report_failures(self) -> None:
        """Warns about failed or entirely absent NVML reads."""
        if self._failed_reads:
            self.logger.warning(f"[GpuUtilizationSampler] {self._failed_reads} NVML reads failed on GPU {self.gpu_index}, first as {self._first_failure}; the utilisation figures below cover only the successful reads")

        if self._nvml_started and not self._util_samples:
            self.logger.warning(f"[GpuUtilizationSampler] No NVML read succeeded on GPU {self.gpu_index}; utilisation and VRAM columns are NaN for this sweep entry")

    def stop(self) -> dict:
        """Stops sampling and returns the utilisation summary.

        Returns:
            Mapping with the availability flag, mean and maximum GPU utilisation in
            percent, peak VRAM in GiB and the number of successful samples; the
            numeric fields are NaN when no read succeeded.
        """
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=max(self.interval_s * 4, 1.0))
            self._thread = None

        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception as error:
                self.logger.warning(f"[GpuUtilizationSampler] NVML shutdown failed on GPU {self.gpu_index} ({type(error).__name__}: {error})")
            self._nvml_ok = False

        self._report_failures()

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
    """Times loader throughput, GPU compute ceiling and end-to-end steps for one spec.

    Attributes:
        dataset   : Dataset the loaders are built over.
        model     : Model trained during the timed steps.
        device    : Device the steps run on.
        optimizer : AdamW optimizer driving the timed steps.
    """
    def __init__(
        self,
        dataset        : Dataset,
        model          : nn.Module,
        to_model_input : Callable[[object, torch.device], torch.Tensor],
        forward_loss   : Callable[[nn.Module, torch.Tensor], torch.Tensor],
        device         : torch.device,
        logger         : Logger,
        use_amp        : bool          = False,
        seed           : int           = 0,
        warmup_batches : int           = 8,
        timed_batches  : int           = 60,
        gpu_index      : Optional[int] = None,
        cpu_threads    : Optional[int] = None,
    ) -> None:
        """Moves the model to the device and prepares the timing budget.

        Args:
            dataset        : Dataset the loaders iterate.
            model          : Model whose training step defines the compute ceiling.
            to_model_input : Callable turning a loader batch and a device into the model input tensor.
            forward_loss   : Callable running the model on that tensor and returning a scalar loss.
            device         : Device the steps run on.
            logger         : Logger receiving NVML warnings.
            use_amp        : Whether the forward pass runs under bfloat16 autocast.
            seed           : Seed of the loader shuffling generator.
            warmup_batches : Batches consumed before timing starts.
            timed_batches  : Batches counted in each timing.
            gpu_index      : NVML device index to sample, or None to sample the CUDA device torch computes on.
            cpu_threads    : Torch intra-op thread count, or None to leave it untouched.
        """
        self.dataset        = dataset
        self.model          = model.to(device)
        self.to_model_input = to_model_input
        self.forward_loss   = forward_loss
        self.device         = device
        self.logger         = logger
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
        """Pins every DataLoader worker to a single torch thread."""
        torch.set_num_threads(1)

    @staticmethod
    def _iterate(loader: DataLoader, n_batches: int):
        """Yields exactly n_batches batches, cycling the loader as often as needed."""
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
        """Builds a shuffled, drop-last DataLoader for one spec."""
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
        """Returns loader-only throughput in samples per second, with no model attached."""
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
        """Returns one batch already converted to the model input tensor."""
        batch = next(self._iterate(loader, 1))
        return self.to_model_input(batch, self.device)

    def _train_step(self, model_input: torch.Tensor) -> torch.Tensor:
        """Runs one optimizer step on a fixed input and returns the scalar loss tensor."""
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            loss = self.forward_loss(self.model, model_input)

        loss.backward()
        self.optimizer.step()

        return loss

    def _measure_compute_ceiling(self, reference_input: torch.Tensor, spec: LoaderSpec) -> float:
        """Returns the GPU compute ceiling in samples per second, replaying one cached batch."""
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
        """Times full training steps and splits the step time into fetch and compute.

        Args:
            loader : Loader supplying the batches.
            spec   : Configuration being measured, used to convert batches to samples.

        Returns:
            Mapping with end-to-end throughput in samples per second, the fraction of
            step time spent waiting for data, and the NVML utilisation summary.
        """
        self.model.train()

        sampler = GpuUtilizationSampler(logger=self.logger, gpu_index=self.gpu_index)
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
        """Blocks until queued CUDA work has finished, on CUDA devices only."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, spec: LoaderSpec) -> dict:
        """Measures one loader configuration end to end.

        Args:
            spec : Configuration to benchmark.

        Returns:
            Record holding the spec fields, loader-only and compute-ceiling
            throughputs in samples per second, their ratio, the achieved compute
            efficiency and the end-to-end measurements.
        """
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
    """Runs a list of loader specs, tolerating out-of-memory configurations.

    Attributes:
        benchmark : Benchmark used to measure each spec.
        specs     : Configurations to measure, in order.
        on_result : Callback invoked with each finished record, or None.
        on_start  : Callback invoked with each spec before it is measured, or None.
    """
    def __init__(self, benchmark: GpuFeedBenchmark, specs: list[LoaderSpec], on_result: Optional[Callable[[dict], None]] = None, on_start: Optional[Callable[[LoaderSpec], None]] = None) -> None:
        """Stores the benchmark, the specs to run and the per-spec callbacks.

        Args:
            benchmark : Benchmark used to measure each spec.
            specs     : Configurations to measure, in order.
            on_result : Callback invoked with each finished record, or None.
            on_start  : Callback invoked with each spec before it is measured, or None, so a hung loader shows which spec it belongs to.
        """
        self.benchmark = benchmark
        self.specs     = list(specs)
        self.on_result = on_result
        self.on_start  = on_start

    def _measure_spec(self, spec: LoaderSpec) -> Optional[dict]:
        """Measures one spec, returning an 'oom' record instead of raising on out-of-memory."""
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
        """Measures every spec in order.

        Returns:
            One record per spec, each carrying a 'status' of 'ok' or 'oom'.
        """
        results = []

        for spec in self.specs:
            if self.on_start is not None:
                self.on_start(spec)

            record = self._measure_spec(spec)
            if record is None:
                continue

            results.append(record)
            if self.on_result is not None:
                self.on_result(record)

        return results
