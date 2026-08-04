from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pipelines.dataloader_tuning.pipeline      import DataLoaderTuningPipeline
from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from tools.benchmarking                        import LoaderSpec, SweepReport


def _config(tmp_path, **kwargs):
    config            = DataLoaderTuningEntryConfig(mode="synthetic", **kwargs)
    config.output_dir = tmp_path / "tuning"
    return config


def _ok_record(batch_size, num_workers, prefetch_factor, pin_memory, samples_per_s):
    return {
        "status"                   : "ok",
        "batch_size"               : batch_size,
        "num_workers"              : num_workers,
        "prefetch_factor"          : prefetch_factor,
        "pin_memory"               : pin_memory,
        "persistent_workers"       : True,
        "end_to_end_samples_per_s" : samples_per_s,
        "data_wait_fraction"       : 0.01,
        "gpu_util_mean"            : 90.0,
        "feed_ratio"               : 1.2,
    }


def test_worker_counts_drops_counts_above_cores(tmp_path, monkeypatch):
    config   = _config(tmp_path, worker_counts=[0, 2, 1000000])
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 4)

    assert pipeline._worker_counts() == [0, 2]


def test_worker_counts_falls_back_to_min_when_all_dropped(tmp_path, monkeypatch):
    config   = _config(tmp_path, worker_counts=[64, 128])
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 4)

    assert pipeline._worker_counts() == [64]


def test_main_specs_is_cartesian_product(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256, 512], worker_counts=[0, 2])
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 8)

    specs = pipeline._main_specs()

    assert len(specs) == 4
    assert all(isinstance(spec, LoaderSpec) for spec in specs)
    assert {spec.batch_size for spec in specs} == {256, 512}


def test_main_specs_uses_reference_prefetch_and_pins(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256], worker_counts=[2], reference_prefetch=4)
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 8)

    spec = pipeline._main_specs()[0]

    assert spec.prefetch_factor == 4
    assert spec.pin_memory is True
    assert spec.persistent_workers is True


def test_refine_specs_sweeps_prefetch_and_pin(tmp_path):
    config   = _config(tmp_path, prefetch_factors=[2, 4])
    pipeline = DataLoaderTuningPipeline(config)

    specs = pipeline._refine_specs({"batch_size": 512, "num_workers": 4})

    assert len(specs) == 4
    assert {spec.prefetch_factor for spec in specs} == {2, 4}
    assert {spec.pin_memory for spec in specs}      == {True, False}
    assert all(spec.batch_size == 512 for spec in specs)


def test_refine_specs_forces_at_least_one_worker(tmp_path):
    config   = _config(tmp_path, prefetch_factors=[2])
    pipeline = DataLoaderTuningPipeline(config)

    specs = pipeline._refine_specs({"batch_size": 256, "num_workers": 0})

    assert all(spec.num_workers == 1 for spec in specs)


def test_final_config_uses_recommendation_defaults(tmp_path):
    config   = _config(tmp_path, reference_prefetch=4)
    pipeline = DataLoaderTuningPipeline(config)

    recommendation = {"batch_size": 512, "num_workers": 4, "cpu_bound": False}
    final          = pipeline._final_config(recommendation, refine_report=None)

    assert final["batch_size"]      == 512
    assert final["num_workers"]     == 4
    assert final["prefetch_factor"] == 4
    assert final["pin_memory"] is True


def test_final_config_overrides_from_refine_report(tmp_path):
    config   = _config(tmp_path)
    pipeline = DataLoaderTuningPipeline(config)

    refine_records = [
        _ok_record(512, 4, 8, False, 5000.0),
        _ok_record(512, 4, 2, True,  3000.0),
    ]
    refine_report = SweepReport(refine_records, wait_threshold=0.05)

    recommendation = {"batch_size": 512, "num_workers": 4, "cpu_bound": False}
    final          = pipeline._final_config(recommendation, refine_report)

    assert final["prefetch_factor"] == 8
    assert final["pin_memory"] is False


@pytest.mark.slow
def test_run_exits_with_reason_when_no_configuration_succeeds(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256], worker_counts=[2], refine=False, save_figures=False)
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 8)

    class FakeTarget:
        dataset        = [torch.zeros(2)]
        model          = torch.nn.Linear(2, 2)
        to_model_input = staticmethod(lambda batch, device: batch)
        forward_loss   = staticmethod(lambda model, x: model(x).mean())
        model_name     = "mlp_ae"
        sample_text    = "synthetic"
        item_source    = "synthetic"
        config_hint    = "synthetic"

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.build_feed_target", lambda *a, **k: FakeTarget())
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.GpuFeedBenchmark", lambda **kwargs: object())

    oom_records = [{"status": "oom", "batch_size": 256, "num_workers": 2, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True}]

    class FakeSweep:
        def __init__(self, benchmark, specs, on_result):
            self.on_result = on_result
        def run(self):
            for record in oom_records:
                self.on_result(dict(record))
            return oom_records

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.DataLoaderSweep", FakeSweep)

    with pytest.raises(SystemExit, match="no working configuration"):
        pipeline.run()


@pytest.mark.slow
def test_run_selects_highest_throughput_config(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256, 512], worker_counts=[2], refine=False, save_figures=False)
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 8)

    class FakeTarget:
        dataset        = [torch.zeros(2)]
        model          = torch.nn.Linear(2, 2)
        to_model_input = staticmethod(lambda batch, device: batch)
        forward_loss   = staticmethod(lambda model, x: model(x).mean())
        model_name     = "mlp_ae"
        sample_text    = "synthetic"
        item_source    = "synthetic"
        config_hint    = "synthetic"

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.build_feed_target", lambda *a, **k: FakeTarget())
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.GpuFeedBenchmark", lambda **kwargs: object())

    main_records = [
        _ok_record(256, 2, 4, True, 1000.0),
        _ok_record(512, 2, 4, True, 9000.0),
    ]

    class FakeSweep:
        def __init__(self, benchmark, specs, on_result):
            self.on_result = on_result
        def run(self):
            for record in main_records:
                self.on_result(dict(record))
            return main_records

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.DataLoaderSweep", FakeSweep)

    final = pipeline.run()

    assert final["batch_size"] == 512

    payload = json.loads((config.output_dir / config.mode / "results.json").read_text())
    assert payload["final"]["batch_size"] == 512
