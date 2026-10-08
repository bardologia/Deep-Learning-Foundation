"""Tests covering the dataloader tuning sweep: loader specs, OOM handling, recommendation selection from the sweep report, and NVML utilisation sampling."""

from __future__ import annotations

import time

import pytest

import tools.benchmarking.dataloader_tuning as tuning_module

from tools.benchmarking.dataloader_tuning import DataLoaderSweep, GpuUtilizationSampler, LoaderSpec
from tools.benchmarking.sweep_report      import SweepReport


def _ok_record(spec: LoaderSpec, throughput: float, wait: float = 0.0, feed: float = 2.0, util: float = 90.0) -> dict:
    """Builds a successful sweep record for one loader spec.

    Args:
        spec       : Loader configuration the record describes.
        throughput : End-to-end samples per second.
        wait       : Fraction of step time spent waiting on data.
        feed       : Ratio of loader-only throughput to the compute ceiling.
        util       : GPU utilisation percentage reported for the run.

    Returns:
        Record dict in the shape the sweep emits, with status 'ok'.
    """
    return {
        **spec.as_record(),
        "loader_only_samples_per_s"     : throughput * feed,
        "compute_ceiling_samples_per_s" : throughput,
        "feed_ratio"                    : feed,
        "compute_efficiency"            : 1.0,
        "end_to_end_samples_per_s"      : throughput,
        "data_wait_fraction"            : wait,
        "gpu_available"                 : True,
        "gpu_util_mean"                 : util,
        "gpu_util_max"                  : util,
        "vram_peak_gb"                  : 1.0,
        "gpu_n_samples"                 : 10,
        "status"                        : "ok",
    }


class FakeBenchmark:
    """Benchmark stub returning canned timings and raising on selected specs.

    Attributes:
        timings    : End-to-end throughput keyed by spec label.
        fail_specs : Labels that raise a generic runtime error.
        oom_specs  : Labels that raise a CUDA out-of-memory error.
        seen       : Specs measured, in call order.
    """
    def __init__(self, timings: dict, fail_specs=None, oom_specs=None):
        """Stores the canned timings and the labels that must fail."""
        self.timings    = timings
        self.fail_specs = set(fail_specs or [])
        self.oom_specs  = set(oom_specs or [])
        self.seen       = []

    def measure(self, spec: LoaderSpec) -> dict:
        """Returns the canned record for a spec, or raises for the failing labels.

        Args:
            spec : Loader configuration to measure.

        Returns:
            Sweep record dict for the spec.

        Raises:
            RuntimeError : If the spec label is registered as failing or out of memory.
        """
        self.seen.append(spec)

        if spec.label in self.oom_specs:
            raise RuntimeError("CUDA out of memory: tried to allocate")
        if spec.label in self.fail_specs:
            raise RuntimeError("kernel exploded")

        return _ok_record(spec, self.timings[spec.label])


def test_loaderspec_label_is_deterministic():
    """Verifies the spec label encodes batch size, workers, prefetch, pinning, and persistence."""
    spec = LoaderSpec(batch_size=64, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    assert spec.label == "bs64_w4_pf2_pin1_persist1"


def test_loaderspec_as_record_zeroes_prefetch_for_zero_workers():
    """Verifies prefetch and persistence collapse to zero and False without workers."""
    spec   = LoaderSpec(batch_size=32, num_workers=0, prefetch_factor=4, pin_memory=False, persistent_workers=True)
    record = spec.as_record()

    assert record["prefetch_factor"]    == 0
    assert record["persistent_workers"] is False
    assert record["num_workers"]        == 0


def test_loaderspec_as_record_keeps_prefetch_for_workers():
    """Verifies prefetch and persistence survive when workers are configured."""
    spec   = LoaderSpec(batch_size=32, num_workers=2, prefetch_factor=6, pin_memory=True, persistent_workers=True)
    record = spec.as_record()

    assert record["prefetch_factor"]    == 6
    assert record["persistent_workers"] is True


def test_sweep_runs_every_spec_in_order():
    """Verifies the sweep measures every spec in declaration order and marks them ok."""
    specs   = [LoaderSpec(64, w) for w in (1, 2, 4)]
    timings = {s.label: 100.0 + i for i, s in enumerate(specs)}

    sweep   = DataLoaderSweep(FakeBenchmark(timings), specs)
    results = sweep.run()

    assert [s.label for s in sweep.benchmark.seen] == [s.label for s in specs]
    assert len(results) == 3
    assert all(r["status"] == "ok" for r in results)


def test_sweep_on_result_callback_invoked_per_record():
    """Verifies the result callback fires once per measured spec."""
    specs     = [LoaderSpec(64, 1), LoaderSpec(64, 2)]
    timings   = {s.label: 100.0 for s in specs}
    collected = []

    sweep = DataLoaderSweep(FakeBenchmark(timings), specs, on_result=collected.append)
    sweep.run()

    assert len(collected) == 2


def test_sweep_marks_oom_specs_and_continues():
    """Verifies an out-of-memory spec is recorded as 'oom' without aborting the sweep."""
    specs   = [LoaderSpec(64, 1), LoaderSpec(512, 1)]
    timings = {specs[0].label: 100.0}

    bench   = FakeBenchmark(timings, oom_specs={specs[1].label})
    results = DataLoaderSweep(bench, specs).run()

    by_label = {(r["batch_size"], r["num_workers"]): r for r in results}
    assert by_label[(64, 1)]["status"]  == "ok"
    assert by_label[(512, 1)]["status"] == "oom"


def test_sweep_reraises_non_oom_runtime_error():
    """Verifies a non-OOM runtime error propagates out of the sweep."""
    specs = [LoaderSpec(64, 1)]
    bench = FakeBenchmark({}, fail_specs={specs[0].label})

    with pytest.raises(RuntimeError, match="kernel exploded"):
        DataLoaderSweep(bench, specs).run()


def test_recommendation_picks_highest_throughput_among_saturated():
    """Verifies the recommendation is the fastest GPU-saturated configuration."""
    s_lo = LoaderSpec(64, 2)
    s_hi = LoaderSpec(128, 4)

    results = [
        _ok_record(s_lo, throughput=100.0, wait=0.0),
        _ok_record(s_hi, throughput=300.0, wait=0.0),
    ]

    rec = SweepReport(results).recommendation
    assert rec["found"]       is True
    assert rec["cpu_bound"]   is False
    assert rec["batch_size"]  == 128
    assert rec["num_workers"] == 4
    assert rec["end_to_end_samples_per_s"] == 300.0


def test_recommendation_saturation_by_feed_ratio_when_wait_high():
    """Verifies a feed ratio above one counts as saturated despite a high data wait."""
    spec   = LoaderSpec(64, 8)
    record = _ok_record(spec, throughput=200.0, wait=0.5, feed=1.5)
    rec    = SweepReport([record], wait_threshold=0.05).recommendation

    assert rec["found"]     is True
    assert rec["cpu_bound"] is False


def test_recommendation_cpu_bound_when_nothing_saturated():
    """Verifies the report flags CPU-bound and still names the fastest configuration."""
    s_a = LoaderSpec(64, 1)
    s_b = LoaderSpec(64, 2)

    results = [
        _ok_record(s_a, throughput=100.0, wait=0.5, feed=0.5),
        _ok_record(s_b, throughput=150.0, wait=0.4, feed=0.6),
    ]

    rec = SweepReport(results, wait_threshold=0.05).recommendation
    assert rec["cpu_bound"]   is True
    assert rec["num_workers"] == 2
    assert rec["end_to_end_samples_per_s"] == 150.0


def test_recommendation_tie_breaks_to_fewer_workers_then_smaller_batch():
    """Verifies equal throughput resolves towards fewer workers and the smaller batch."""
    s_many  = LoaderSpec(128, 8)
    s_few   = LoaderSpec(64, 2)

    results = [
        _ok_record(s_many, throughput=200.0, wait=0.0),
        _ok_record(s_few,  throughput=200.0, wait=0.0),
    ]

    rec = SweepReport(results).recommendation
    assert rec["num_workers"] == 2
    assert rec["batch_size"]  == 64


def test_recommendation_not_found_when_all_failed():
    """Verifies a sweep of only failed rows yields no recommendation and a reason."""
    spec   = LoaderSpec(64, 1)
    failed = {**spec.as_record(), "status": "oom"}

    rec = SweepReport([failed]).recommendation
    assert rec["found"] is False
    assert "reason" in rec


def test_recommendation_ignores_oom_rows():
    """Verifies out-of-memory rows are excluded from the recommendation."""
    good   = LoaderSpec(64, 2)
    dead   = LoaderSpec(512, 4)

    results = [
        _ok_record(good, throughput=120.0, wait=0.0),
        {**dead.as_record(), "status": "oom"},
    ]

    rec = SweepReport(results).recommendation
    assert rec["found"]      is True
    assert rec["batch_size"] == 64


def test_frame_exposes_all_rows():
    """Verifies the full frame keeps failed rows while the ok frame drops them."""
    good = LoaderSpec(64, 2)
    bad  = LoaderSpec(512, 4)

    results = [
        _ok_record(good, throughput=120.0),
        {**bad.as_record(), "status": "oom"},
    ]

    report = SweepReport(results)
    assert len(report.frame)     == 2
    assert len(report.ok_frame)  == 1


class _CollectingLogger:
    """Logger stub collecting warning messages.

    Attributes:
        warnings : Warning messages received.
    """
    def __init__(self):
        """Creates the empty warning buffer."""
        self.warnings = []

    def warning(self, message):
        """Stores a warning message."""
        self.warnings.append(message)


def test_sampler_warns_when_nvml_init_fails(monkeypatch):
    """Verifies a failed NVML init warns once and reports the GPU as unavailable."""
    logger = _CollectingLogger()

    monkeypatch.setattr(tuning_module.pynvml, "nvmlInit", lambda: (_ for _ in ()).throw(RuntimeError("no driver")))

    sampler = GpuUtilizationSampler(logger=logger, gpu_index=0)
    sampler.start()
    summary = sampler.stop()

    assert summary["gpu_available"] is False
    assert len(logger.warnings)     == 1
    assert "NVML unavailable"       in logger.warnings[0]


def test_sampler_warns_when_every_read_fails(monkeypatch):
    """Verifies repeated NVML read failures warn and leave the GPU unavailable."""
    logger = _CollectingLogger()

    monkeypatch.setattr(tuning_module.pynvml, "nvmlInit",                  lambda: None)
    monkeypatch.setattr(tuning_module.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: object())
    monkeypatch.setattr(tuning_module.pynvml, "nvmlShutdown",              lambda: None)
    monkeypatch.setattr(tuning_module.pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: (_ for _ in ()).throw(RuntimeError("gpu lost")))

    sampler = GpuUtilizationSampler(logger=logger, gpu_index=0, interval_s=0.01)
    sampler.start()
    time.sleep(0.05)
    summary = sampler.stop()

    assert summary["gpu_available"] is False
    assert any("NVML reads failed"  in message for message in logger.warnings)
    assert any("No NVML read succeeded" in message for message in logger.warnings)


def test_sampler_without_an_index_samples_the_device_torch_computes_on(monkeypatch):
    """Verifies a sampler given no NVML index finds the device by the UUID of torch's CUDA device rather than by the requested position."""
    logger = _CollectingLogger()
    found  = {}

    monkeypatch.setattr(tuning_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(GpuUtilizationSampler, "torch_uuid", staticmethod(lambda: "GPU-abc"))
    monkeypatch.setattr(tuning_module.pynvml, "nvmlInit",                      lambda: None)
    monkeypatch.setattr(tuning_module.pynvml, "nvmlShutdown",                  lambda: None)
    monkeypatch.setattr(tuning_module.pynvml, "nvmlDeviceGetHandleByUUID",     lambda uuid: found.setdefault("uuid", uuid))
    monkeypatch.setattr(tuning_module.pynvml, "nvmlDeviceGetIndex",            lambda handle: 3)
    monkeypatch.setattr(tuning_module.pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: type("Util", (), {"gpu": 50.0})())
    monkeypatch.setattr(tuning_module.pynvml, "nvmlDeviceGetMemoryInfo",       lambda handle: type("Mem", (), {"used": 2 * 1024 ** 3})())

    sampler = GpuUtilizationSampler(logger=logger, interval_s=0.01)
    sampler.start()
    time.sleep(0.05)
    summary = sampler.stop()

    assert found["uuid"]            == "GPU-abc"
    assert sampler.gpu_index        == 3
    assert summary["gpu_available"] is True
    assert logger.warnings          == []


def test_sampler_without_a_cuda_device_samples_nothing_and_says_so(monkeypatch):
    """Verifies a sampler with no index and no CUDA device does not read a physical GPU that the run does not use."""
    logger = _CollectingLogger()
    monkeypatch.setattr(tuning_module.torch.cuda, "is_available", lambda: False)

    sampler = GpuUtilizationSampler(logger=logger)
    sampler.start()
    summary = sampler.stop()

    assert summary["gpu_available"] is False
    assert logger.warnings == ["[GpuUtilizationSampler] No CUDA device is visible and no NVML index was given; this sweep entry reports no GPU utilisation"]
