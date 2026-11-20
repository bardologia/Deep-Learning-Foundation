from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
import torch

from pipelines.dataloader_tuning.pipeline      import DataLoaderTuningPipeline
from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from tools.benchmarking                        import LoaderSpec, SweepReport


def _config(tmp_path, **kwargs):
    config            = DataLoaderTuningEntryConfig(**kwargs)
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


def test_the_measured_device_names_the_nvml_device_the_utilisation_comes_from(tmp_path, monkeypatch):
    from tools.benchmarking import GpuUtilizationSampler

    pipeline = DataLoaderTuningPipeline(_config(tmp_path, gpu=1))
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.torch.cuda.get_device_name", lambda index: "Fake GPU")
    monkeypatch.setattr(GpuUtilizationSampler, "nvml_device", classmethod(lambda cls: {"index": 5, "name": "Fake GPU", "uuid": "GPU-abc"}))

    assert pipeline._gpu() == {"index": 1, "name": "Fake GPU", "nvml_index": 5, "nvml_name": "Fake GPU", "uuid": "GPU-abc"}


def test_a_cpu_sweep_records_no_nvml_device(tmp_path, monkeypatch):
    pipeline = DataLoaderTuningPipeline(_config(tmp_path, gpu=2))
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.torch.cuda.is_available", lambda: False)

    assert pipeline._gpu() == {"index": 2, "name": None, "nvml_index": None, "nvml_name": None, "uuid": None}


def test_each_sweep_writes_into_its_own_directory_named_after_the_model(tmp_path, monkeypatch):
    tags = iter(["20261002_120000", "20261002_120500"])
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.RunTag.now", lambda: next(tags))

    first  = DataLoaderTuningPipeline(_config(tmp_path))
    second = DataLoaderTuningPipeline(_config(tmp_path, model_name="unet"))

    assert first.output_dir  == tmp_path / "tuning" / "resunet_20261002_120000"
    assert second.output_dir == tmp_path / "tuning" / "unet_20261002_120500"
    assert first.work_dir    == second.work_dir == tmp_path / "tuning" / "work"
    assert (first.output_dir / "logs" / "dataloader_tuning.log").is_file()


def test_a_sweep_refuses_an_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.RunTag.now", lambda: "20261002_120000")

    DataLoaderTuningPipeline(_config(tmp_path))

    with pytest.raises(FileExistsError):
        DataLoaderTuningPipeline(_config(tmp_path))


def test_feed_target_table_names_the_default_model_device_and_threads(tmp_path):
    pipeline     = DataLoaderTuningPipeline(_config(tmp_path, warmup_batches=4, timed_batches=20))
    pipeline.gpu = {"index": 0, "name": None, "nvml_index": None, "nvml_name": None, "uuid": None}
    tables       = []
    warnings     = []
    pipeline.logger.kv_table = lambda data, title=None: tables.append((title, dict(data)))
    pipeline.logger.warning  = warnings.append

    class Target:
        model       = torch.nn.Linear(2, 2)
        model_name  = "resunet"
        sample_text = "96 samples"
        item_source = "PatchDataset"

    assert pipeline._log_feed_target(Target(), torch.device("cpu")) == 6

    title, rows = tables[0]
    assert title                 == "Feed target"
    assert rows["Model"]         == "resunet (default)"
    assert rows["GPU"]           == "none (CPU)"
    assert rows["Torch threads"] == torch.get_num_threads()
    assert rows["Warm-up/timed"] == "4/20 batches"
    assert warnings[0].startswith("No CUDA device is visible")


def test_worker_counts_drops_counts_above_cores(tmp_path, monkeypatch):
    config   = _config(tmp_path, worker_counts=[0, 2, 1000000])
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 4)

    assert pipeline._worker_counts() == [0, 2]


def test_worker_counts_clamps_to_the_core_count_when_all_dropped(tmp_path, monkeypatch):
    config   = _config(tmp_path, worker_counts=[64, 128])
    pipeline = DataLoaderTuningPipeline(config)

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 4)

    assert pipeline._worker_counts() == [4]


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


def test_refinement_is_skipped_with_a_warning_for_a_zero_worker_recommendation(tmp_path):
    config   = _config(tmp_path, prefetch_factors=[2])
    pipeline = DataLoaderTuningPipeline(config)
    warnings = []
    pipeline.logger.warning = warnings.append

    assert pipeline._refine(object(), {"batch_size": 256, "num_workers": 0}) == ([], None)
    assert warnings and warnings[0].startswith("Refinement skipped: the main sweep recommends 0 loader workers")


def test_final_config_of_a_zero_worker_loader_has_no_prefetch_or_persistent_workers(tmp_path):
    pipeline = DataLoaderTuningPipeline(_config(tmp_path, reference_prefetch=4))

    final = pipeline._final_config({"batch_size": 256, "num_workers": 0, "cpu_bound": False}, refine_report=None)

    assert final["prefetch_factor"]    == 0
    assert final["persistent_workers"] is False


def test_final_config_warns_when_no_refinement_spec_succeeded(tmp_path):
    pipeline = DataLoaderTuningPipeline(_config(tmp_path, reference_prefetch=4))
    warnings = []
    pipeline.logger.warning = warnings.append
    report   = SweepReport([{"status": "oom", "batch_size": 512, "num_workers": 4, "prefetch_factor": 8, "pin_memory": True, "persistent_workers": True}])

    final = pipeline._final_config({"batch_size": 512, "num_workers": 4, "cpu_bound": False, "end_to_end_samples_per_s": 3000.0}, report)

    assert final["prefetch_factor"] == 4
    assert warnings == ["No refinement spec succeeded; keeping prefetch 4 and pin_memory True as the main sweep measured them"]


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

    recommendation = {"batch_size": 512, "num_workers": 4, "cpu_bound": False, "end_to_end_samples_per_s": 4000.0}
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
        model_name     = "resunet"
        sample_text    = "patches"
        item_source    = "PatchDataset"
        config_hint    = "DatasetConfig"

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.FeedAdapter.build", lambda self: FakeTarget())
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.GpuFeedBenchmark", lambda **kwargs: object())

    oom_records = [{"status": "oom", "batch_size": 256, "num_workers": 2, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True}]

    class FakeSweep:
        def __init__(self, benchmark, specs, on_result, on_start):
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
        model_name     = "resunet"
        sample_text    = "patches"
        item_source    = "PatchDataset"
        config_hint    = "DatasetConfig"

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.FeedAdapter.build", lambda self: FakeTarget())
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.GpuFeedBenchmark", lambda **kwargs: object())

    main_records = [
        _ok_record(256, 2, 4, True, 1000.0),
        _ok_record(512, 2, 4, True, 9000.0),
    ]

    class FakeSweep:
        def __init__(self, benchmark, specs, on_result, on_start):
            self.on_result = on_result
        def run(self):
            for record in main_records:
                self.on_result(dict(record))
            return main_records

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.DataLoaderSweep", FakeSweep)

    final = pipeline.run()

    assert final["batch_size"] == 512

    payload = json.loads((pipeline.output_dir / "results.json").read_text())
    assert payload["final"]["batch_size"] == 512
    assert payload["status"]                     == "complete"
    assert payload["config"]["data_wait_target"] == config.data_wait_target
    assert payload["config"]["timed_batches"]    == config.timed_batches
    assert payload["gpu"]["index"]               == config.gpu
    assert set(payload["provenance"]) >= {"git_commit", "host", "torch"}
    assert payload["started_at"] <= payload["finished_at"]
    assert json.loads((pipeline.output_dir / "resolved_config.json").read_text())["batch_sizes"] == [256, 512]


class _FakeTarget:
    dataset        = [torch.zeros(2)]
    model          = torch.nn.Linear(2, 2)
    to_model_input = staticmethod(lambda batch, device: batch)
    forward_loss   = staticmethod(lambda model, x: model(x).mean())
    model_name     = "resunet"
    sample_text    = "patches"
    item_source    = "PatchDataset"
    config_hint    = "DatasetConfig"


def _replay_sweeps(monkeypatch, phases: list):
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.os.cpu_count", lambda: 8)
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.FeedAdapter.build", lambda self: _FakeTarget())
    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.GpuFeedBenchmark", lambda **kwargs: object())

    pending = list(phases)

    class ReplaySweep:
        def __init__(self, benchmark, specs, on_result, on_start):
            self.on_result = on_result

        def run(self):
            phase = pending.pop(0)
            if isinstance(phase, Exception):
                Mock(side_effect=phase)()
            for record in phase:
                self.on_result(dict(record))
            return phase

    monkeypatch.setattr("pipelines.dataloader_tuning.pipeline.DataLoaderSweep", ReplaySweep)


def test_sweep_lines_carry_their_position_and_the_run_ends_with_a_summary(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256, 512], worker_counts=[2], refine=False, save_figures=False)
    pipeline = DataLoaderTuningPipeline(config)
    starving = {**_ok_record(256, 2, 4, True, 1000.0), "data_wait_fraction": 0.4, "feed_ratio": 0.5}
    _replay_sweeps(monkeypatch, [[starving, {"status": "oom", "batch_size": 512, "num_workers": 2, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True}]])

    subsections = []
    tables      = []
    warnings    = []
    pipeline.logger.subsection = subsections.append
    pipeline.logger.kv_table   = lambda data, title=None: tables.append((title, dict(data)))
    pipeline.logger.warning    = warnings.append

    pipeline.run()

    assert subsections[0].startswith("[main 1/2] bs=  256 workers=2")
    assert subsections[1] == "[main 2/2] bs=512 workers=2 -> oom"
    assert ("Sweep summary", {"Main sweep": "1 ok, 1 out of memory", "Refinement": "not run", "Saturated": "0 of 1 working specs"}) in tables
    assert any(message.startswith("No configuration reached the 5% data-wait target") for message in warnings)


def test_a_sweep_with_no_working_configuration_records_its_measurements(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256], worker_counts=[2], refine=False, save_figures=False)
    pipeline = DataLoaderTuningPipeline(config)
    _replay_sweeps(monkeypatch, [[{"status": "oom", "batch_size": 256, "num_workers": 2, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True}]])

    with pytest.raises(SystemExit, match="no working configuration"):
        pipeline.run()

    payload = json.loads((pipeline.output_dir / "results.json").read_text())
    assert payload["status"]                  == "failed"
    assert payload["recommendation"]["found"] is False
    assert payload["main"][0]["status"]       == "oom"


def test_a_crash_during_refinement_leaves_the_main_sweep_as_partial_results(tmp_path, monkeypatch):
    config   = _config(tmp_path, batch_sizes=[256], worker_counts=[2], prefetch_factors=[2], refine=True, save_figures=False)
    pipeline = DataLoaderTuningPipeline(config)
    _replay_sweeps(monkeypatch, [[_ok_record(256, 2, 4, True, 1000.0)], RuntimeError("loader worker hung")])

    with pytest.raises(RuntimeError, match="loader worker hung"):
        pipeline.run()

    log = (pipeline.output_dir / "logs" / "dataloader_tuning.log").read_text()
    assert ">>> ABORTED dataloader_tuning" in log
    assert "RuntimeError: loader worker hung" in log
    assert ">>> END dataloader_tuning" not in log

    payload = json.loads((pipeline.output_dir / "results.json").read_text())
    assert payload["status"]                       == "partial"
    assert payload["finished_at"]                  is None
    assert payload["final"]                        is None
    assert payload["recommendation"]["batch_size"] == 256
    assert len(payload["main"])                    == 1
