
from __future__ import annotations

import time
import types

import pytest

import tools.monitoring.resource_monitor as rm
from tools.monitoring.resource_monitor import ResourceMonitor
from unittest.mock import Mock


class FakeConfig:
    def __init__(self, **kwargs):
        self.enabled            = kwargs.get("enabled", True)
        self.poll_interval_sec  = kwargs.get("poll_interval_sec", 0.05)
        self.log_to_tensorboard = kwargs.get("log_to_tensorboard", True)
        self.warn_ram_pct       = kwargs.get("warn_ram_pct", 90.0)
        self.warn_vram_pct      = kwargs.get("warn_vram_pct", 90.0)
        self.warn_shm_pct       = kwargs.get("warn_shm_pct", 80.0)
        self.warn_cooldown_sec  = kwargs.get("warn_cooldown_sec", 30.0)


class CollectingLogger:
    def __init__(self):
        self.warnings    = []
        self.sections    = []
        self.subsections = []
        self.kv_tables   = []

    def warning(self, msg):
        self.warnings.append(msg)

    def section(self, msg):
        self.sections.append(msg)

    def subsection(self, msg):
        self.subsections.append(msg)

    def kv_table(self, data, *args, **kwargs):
        self.kv_tables.append(dict(data))


class RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_metrics(self, prefix, metrics, step):
        self.calls.append((prefix, dict(metrics), step))


@pytest.fixture
def cfg():
    return FakeConfig()


@pytest.fixture
def monitor(cfg):
    return ResourceMonitor(cfg, logger=CollectingLogger())


def test_load_config_reads_fields(cfg):
    m = ResourceMonitor(cfg, logger=None)

    assert m.enabled       is True
    assert m.interval      == 0.05
    assert m.warn_ram_pct == 90.0
    assert m.warn_shm_pct == 80.0


def test_missing_config_field_raises():
    with pytest.raises(AttributeError):
        ResourceMonitor(types.SimpleNamespace(), logger=None)


def test_renamed_config_field_raises():
    cfg = FakeConfig()
    cfg.poll_interval_s = cfg.poll_interval_sec
    del cfg.poll_interval_sec

    with pytest.raises(AttributeError, match="poll_interval_sec"):
        ResourceMonitor(cfg, logger=None)


def test_step_getter_default_is_zero(cfg):
    m = ResourceMonitor(cfg, logger=None)

    assert m.step_getter() == 0


def test_bytes_to_gb():
    assert ResourceMonitor._bytes_to_gb(1024 ** 3) == 1.0
    assert ResourceMonitor._bytes_to_gb(0)         == 0.0


def test_peak_initialised_to_zero(monitor):
    assert set(monitor.peak) >= {"ram_used_gb", "ram_pct", "vram_used_gb"}
    assert all(v == 0.0 for v in monitor.peak.values())


def test_sample_returns_cpu_mem_metrics(monitor):
    metrics = monitor.sample()

    assert metrics["ram_used_gb"]      > 0.0
    assert metrics["ram_total_gb"]     > 0.0
    assert metrics["ram_available_gb"] > 0.0
    assert 0.0 <= metrics["ram_pct"] <= 100.0
    assert "swap_used_gb" in metrics
    assert "proc_rss_gb"  in metrics
    assert "cpu_pct"      in metrics


def test_sample_includes_vram_keys(monitor):
    metrics = monitor.sample()

    assert "vram_used_gb" in metrics
    assert "vram_pct"     in metrics
    assert metrics["vram_used_gb"] >= 0.0


def test_sample_updates_peak(monitor):
    monitor.sample()

    assert monitor.peak["ram_used_gb"] > 0.0
    assert monitor.peak["ram_pct"]     > 0.0


def test_peak_is_monotonic_non_decreasing(monitor):
    monitor.sample()
    first = monitor.peak["ram_used_gb"]
    monitor.sample()
    second = monitor.peak["ram_used_gb"]

    assert second >= first


def test_get_shm_usage_returns_tuple(monitor):
    used, pct = monitor._get_shm_usage()

    assert used >= 0.0
    assert 0.0 <= pct <= 100.0


def test_maybe_warn_respects_cooldown():
    cfg    = FakeConfig(warn_cooldown_sec=1000.0)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)

    m._maybe_warn("ram", "first")
    m._maybe_warn("ram", "second")

    assert logger.warnings == ["[ResourceMonitor] first"]


def test_maybe_warn_distinct_keys_both_fire():
    cfg    = FakeConfig(warn_cooldown_sec=1000.0)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)

    m._maybe_warn("ram", "a")
    m._maybe_warn("swap", "b")

    assert len(logger.warnings) == 2


def test_maybe_warn_after_cooldown_fires_again():
    cfg    = FakeConfig(warn_cooldown_sec=0.0)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)

    m._maybe_warn("ram", "a")
    m._maybe_warn("ram", "b")

    assert len(logger.warnings) == 2


def test_check_warnings_triggers_on_low_thresholds():
    cfg    = FakeConfig(warn_ram_pct=0.0, warn_shm_pct=0.0, warn_cooldown_sec=1000.0)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)

    m.sample()

    assert any("RAM usage" in w for w in logger.warnings)


def test_check_warnings_silent_with_high_thresholds():
    cfg    = FakeConfig(warn_ram_pct=999.0, warn_shm_pct=999.0, warn_vram_pct=999.0)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)

    m.sample()

    assert logger.warnings == []


def test_disabled_start_does_not_spawn_thread():
    cfg    = FakeConfig(enabled=False)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)
    m.start()

    assert m._thread is None
    assert any("disabled" in s for s in logger.subsections)

    m.stop()


def test_start_stop_runs_sampling_thread():
    cfg     = FakeConfig(poll_interval_sec=0.02)
    tracker = RecordingTracker()
    logger  = CollectingLogger()
    m       = ResourceMonitor(cfg, logger=logger, tracker=tracker, step_getter=lambda: 7)

    m.start()

    assert m._thread is not None
    assert m._thread.is_alive()

    deadline = time.time() + 3.0
    while m._sample_idx == 0 and time.time() < deadline:
        time.sleep(0.02)

    m.stop()

    assert m._thread is None
    assert m._sample_idx > 0
    assert len(tracker.calls) > 0


def test_publish_logs_to_tracker_with_step():
    cfg     = FakeConfig()
    tracker = RecordingTracker()
    m       = ResourceMonitor(cfg, logger=None, tracker=tracker, step_getter=lambda: 42)

    m._publish({"ram_pct": 12.0})

    assert tracker.calls == [("system", {"ram_pct": 12.0}, 42)]


def test_publish_filters_to_tb_whitelist():
    cfg     = FakeConfig()
    tracker = RecordingTracker()
    m       = ResourceMonitor(cfg, logger=None, tracker=tracker, step_getter=lambda: 3)

    m._publish({"ram_pct": 12.0, "ram_total_gb": 64.0, "vram_pct": 50.0, "proc_num_threads": 8.0, "loadavg_1m": 1.0})

    assert tracker.calls == [("system", {"ram_pct": 12.0}, 3)]


def test_publish_skipped_when_tb_disabled():
    cfg     = FakeConfig(log_to_tensorboard=False)
    tracker = RecordingTracker()
    m       = ResourceMonitor(cfg, logger=None, tracker=tracker)

    m._publish({"ram_pct": 1.0})

    assert tracker.calls == []


def test_publish_skipped_without_tracker():
    cfg = FakeConfig()
    m   = ResourceMonitor(cfg, logger=None, tracker=None)

    m._publish({"ram_pct": 1.0})


def test_start_logs_startup_info():
    logger = CollectingLogger()
    m      = ResourceMonitor(FakeConfig(poll_interval_sec=0.02), logger=logger)
    m.start()
    m.stop()

    assert any("Resource Monitor" in s for s in logger.sections)
    assert any("NVML available" in table for table in logger.kv_tables)


def test_stop_logs_peak_metrics():
    logger = CollectingLogger()
    m      = ResourceMonitor(FakeConfig(poll_interval_sec=0.02), logger=logger)
    m.sample()
    m.start()
    m.stop()

    assert any("Peaks" in s for s in logger.sections)
    assert any("Total samples" in table for table in logger.kv_tables)


def test_stop_without_start_is_safe():
    m = ResourceMonitor(FakeConfig(), logger=CollectingLogger())
    m.stop()

    assert m._thread is None


def test_context_manager_starts_and_stops():
    cfg     = FakeConfig(poll_interval_sec=0.02)
    tracker = RecordingTracker()

    with ResourceMonitor(cfg, logger=CollectingLogger(), tracker=tracker) as m:
        assert m._thread is not None
        deadline = time.time() + 3.0
        while m._sample_idx == 0 and time.time() < deadline:
            time.sleep(0.02)

    assert m._thread is None
    assert m._sample_idx > 0


def test_run_survives_sample_exception(monkeypatch):
    cfg    = FakeConfig(poll_interval_sec=0.02)
    logger = CollectingLogger()
    m      = ResourceMonitor(cfg, logger=logger)

    def boom():
        Mock(side_effect=ValueError("sample blew up"))()

    monkeypatch.setattr(m, "sample", boom)

    m.start()
    deadline = time.time() + 2.0
    while not logger.warnings and time.time() < deadline:
        time.sleep(0.02)
    m.stop()

    assert any("sample failed" in w for w in logger.warnings)


def test_nvml_unavailable_path_has_no_gpu_handles(monkeypatch):
    def fail_init():
        Mock(side_effect=RuntimeError("no nvml"))()

    monkeypatch.setattr(rm.pynvml, "nvmlInit", fail_init)

    m = ResourceMonitor(FakeConfig(), logger=CollectingLogger())

    assert m._nvml_ok     is False
    assert m._gpu_handles == []
    assert m._nvml_error  == "RuntimeError: no nvml"

    metrics = m.sample()

    assert metrics["vram_used_gb"] == 0.0
    assert metrics["vram_pct"]     == 0.0


def test_nvml_failure_is_named_in_the_startup_table_and_warned_on_a_cuda_process(monkeypatch):
    def fail_init():
        Mock(side_effect=RuntimeError("driver/library version mismatch"))()

    monkeypatch.setattr(rm.pynvml, "nvmlInit", fail_init)
    monkeypatch.setattr(ResourceMonitor, "_cuda_available", staticmethod(lambda: False))

    logger = CollectingLogger()
    m      = ResourceMonitor(FakeConfig(), logger=logger)
    m._log_startup_info()

    assert logger.kv_tables[0]["NVML available"] == "False (RuntimeError: driver/library version mismatch)"
    assert logger.warnings == []

    monkeypatch.setattr(ResourceMonitor, "_cuda_available", staticmethod(lambda: True))
    m._log_startup_info()

    assert len(logger.warnings) == 1
    assert "driver/library version mismatch" in logger.warnings[0]
    assert "system/gpu*" in logger.warnings[0]


def test_a_device_whose_queries_fail_is_warned_about_once(monkeypatch):
    def broken_memory(handle):
        Mock(side_effect=RuntimeError("GPU is lost"))()

    monkeypatch.setattr(rm.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: f"handle{index}")
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetMemoryInfo", broken_memory)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    logger = CollectingLogger()
    m      = ResourceMonitor(FakeConfig(), logger=logger)

    m.sample()
    m.sample()

    assert len(logger.warnings) == 1
    assert "gpu0" in logger.warnings[0] and "GPU is lost" in logger.warnings[0]
    assert m._nvml_failed == {0}


def test_handles_cover_only_the_visible_devices(monkeypatch):
    requested = []

    monkeypatch.setattr(rm.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetCount", lambda: 4)
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: requested.append(index) or f"handle{index}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    m = ResourceMonitor(FakeConfig(), logger=CollectingLogger())

    assert m._nvml_ok     is True
    assert requested      == [2]
    assert m._gpu_handles == ["handle2"]


def test_handles_cover_every_device_when_no_selection_is_set(monkeypatch):
    monkeypatch.setattr(rm.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetCount", lambda: 3)
    monkeypatch.setattr(rm.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: f"handle{index}")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    m = ResourceMonitor(FakeConfig(), logger=CollectingLogger())

    assert m._gpu_handles == ["handle0", "handle1", "handle2"]

