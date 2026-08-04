from __future__ import annotations

import pytest

from tools.benchmarking.dataloader_tuning import DataLoaderSweep, LoaderSpec
from tools.benchmarking.sweep_report      import SweepReport


def _ok_record(spec: LoaderSpec, throughput: float, wait: float = 0.0, feed: float = 2.0, util: float = 90.0) -> dict:
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
    def __init__(self, timings: dict, fail_specs=None, oom_specs=None):
        self.timings    = timings
        self.fail_specs = set(fail_specs or [])
        self.oom_specs  = set(oom_specs or [])
        self.seen       = []

    def measure(self, spec: LoaderSpec) -> dict:
        self.seen.append(spec)

        if spec.label in self.oom_specs:
            raise RuntimeError("CUDA out of memory: tried to allocate")
        if spec.label in self.fail_specs:
            raise RuntimeError("kernel exploded")

        return _ok_record(spec, self.timings[spec.label])


def test_loaderspec_label_is_deterministic():
    spec = LoaderSpec(batch_size=64, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    assert spec.label == "bs64_w4_pf2_pin1_persist1"


def test_loaderspec_as_record_zeroes_prefetch_for_zero_workers():
    spec   = LoaderSpec(batch_size=32, num_workers=0, prefetch_factor=4, pin_memory=False, persistent_workers=True)
    record = spec.as_record()

    assert record["prefetch_factor"]    == 0
    assert record["persistent_workers"] is False
    assert record["num_workers"]        == 0


def test_loaderspec_as_record_keeps_prefetch_for_workers():
    spec   = LoaderSpec(batch_size=32, num_workers=2, prefetch_factor=6, pin_memory=True, persistent_workers=True)
    record = spec.as_record()

    assert record["prefetch_factor"]    == 6
    assert record["persistent_workers"] is True


def test_sweep_runs_every_spec_in_order():
    specs   = [LoaderSpec(64, w) for w in (1, 2, 4)]
    timings = {s.label: 100.0 + i for i, s in enumerate(specs)}

    sweep   = DataLoaderSweep(FakeBenchmark(timings), specs)
    results = sweep.run()

    assert [s.label for s in sweep.benchmark.seen] == [s.label for s in specs]
    assert len(results) == 3
    assert all(r["status"] == "ok" for r in results)


def test_sweep_on_result_callback_invoked_per_record():
    specs     = [LoaderSpec(64, 1), LoaderSpec(64, 2)]
    timings   = {s.label: 100.0 for s in specs}
    collected = []

    sweep = DataLoaderSweep(FakeBenchmark(timings), specs, on_result=collected.append)
    sweep.run()

    assert len(collected) == 2


def test_sweep_marks_oom_specs_and_continues():
    specs   = [LoaderSpec(64, 1), LoaderSpec(512, 1)]
    timings = {specs[0].label: 100.0}

    bench   = FakeBenchmark(timings, oom_specs={specs[1].label})
    results = DataLoaderSweep(bench, specs).run()

    by_label = {(r["batch_size"], r["num_workers"]): r for r in results}
    assert by_label[(64, 1)]["status"]  == "ok"
    assert by_label[(512, 1)]["status"] == "oom"


def test_sweep_reraises_non_oom_runtime_error():
    specs = [LoaderSpec(64, 1)]
    bench = FakeBenchmark({}, fail_specs={specs[0].label})

    with pytest.raises(RuntimeError, match="kernel exploded"):
        DataLoaderSweep(bench, specs).run()


def test_recommendation_picks_highest_throughput_among_saturated():
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
    spec   = LoaderSpec(64, 8)
    record = _ok_record(spec, throughput=200.0, wait=0.5, feed=1.5)
    rec    = SweepReport([record], wait_threshold=0.05).recommendation

    assert rec["found"]     is True
    assert rec["cpu_bound"] is False


def test_recommendation_cpu_bound_when_nothing_saturated():
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
    spec   = LoaderSpec(64, 1)
    failed = {**spec.as_record(), "status": "oom"}

    rec = SweepReport([failed]).recommendation
    assert rec["found"] is False
    assert "reason" in rec


def test_recommendation_ignores_oom_rows():
    good   = LoaderSpec(64, 2)
    dead   = LoaderSpec(512, 4)

    results = [
        _ok_record(good, throughput=120.0, wait=0.0),
        {**dead.as_record(), "status": "oom"},
    ]

    rec = SweepReport(results).recommendation
    assert rec["found"]      is True
    assert rec["batch_size"] == 64


def test_dataframe_exposes_all_rows():
    good = LoaderSpec(64, 2)
    bad  = LoaderSpec(512, 4)

    results = [
        _ok_record(good, throughput=120.0),
        {**bad.as_record(), "status": "oom"},
    ]

    report = SweepReport(results)
    assert len(report.dataframe) == 2
    assert len(report.ok_frame)  == 1
