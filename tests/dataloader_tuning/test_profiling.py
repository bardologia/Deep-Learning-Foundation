"""Tests for the training-step profiler over the synthetic feed."""

from __future__ import annotations

import json

import pytest
import torch

from configuration.benchmark.dataloader_tuning import TrainingProfileEntryConfig
from pipelines.dataloader_tuning.profiling     import TrainingProfilePipeline
from tools.benchmarking.step_profiler          import StepProfiler

from tests.conftest import SilentLogger

pytestmark = pytest.mark.usefixtures("force_cpu")


def _entry(tmp_path) -> TrainingProfileEntryConfig:
    """Returns a tiny synthetic profiling entry writing under tmp_path."""
    entry = TrainingProfileEntryConfig(
        mode              = "synthetic",
        batch_size        = 16,
        num_workers       = 0,
        warmup_steps      = 1,
        active_steps      = 2,
        with_stack        = False,
        synthetic_samples = 64,
        synthetic_length  = 32,
    )
    entry.output_dir = tmp_path / "profiles"

    return entry


def test_pipeline_writes_trace_table_and_summary(tmp_path):
    """A synthetic CPU profile produces the chrome trace, operator table and summary."""
    summary = TrainingProfilePipeline(_entry(tmp_path), SilentLogger()).run()

    stamps = list((tmp_path / "profiles").iterdir())
    stamp  = next(d for d in stamps if (d / "summary.json").is_file())

    payload = json.loads((stamp / "summary.json").read_text())

    assert (stamp / "trace.json").is_file()
    assert (stamp / "op_table.txt").stat().st_size > 0
    assert summary["active_steps"] == 2
    assert payload["top_operators"]
    assert all(entry["count"] > 0 for entry in payload["top_operators"])


def test_profiler_rejects_an_empty_loader(tmp_path):
    """A batch size larger than the dataset fails loudly instead of recording nothing."""
    entry                   = _entry(tmp_path)
    entry.batch_size        = 4096
    entry.synthetic_samples = 8

    with pytest.raises(ValueError, match="zero batches"):
        TrainingProfilePipeline(entry, SilentLogger()).run()


def test_profiler_cpu_activities_only_without_cuda(tmp_path):
    """On a CPU device the profiler requests CPU activity only."""
    profiler = StepProfiler(
        feed            = None,
        device          = torch.device("cpu"),
        batch_size      = 1,
        num_workers     = 0,
        prefetch_factor = 2,
        use_amp         = False,
        warmup_steps    = 0,
        active_steps    = 1,
        record_shapes   = False,
        profile_memory  = False,
        with_stack      = False,
        logger          = SilentLogger(),
    )

    assert [activity.name for activity in profiler._activities()] == ["CPU"]
