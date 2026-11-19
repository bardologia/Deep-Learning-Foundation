from __future__ import annotations

import json

import pytest
import torch

from torch.utils.data import TensorDataset

from configuration.benchmark.dataloader_tuning import TrainingProfileEntryConfig
from pipelines.dataloader_tuning.adapters      import FeedAdapter, FeedTarget
from pipelines.dataloader_tuning.profiling     import TrainingProfilePipeline
from tools.benchmarking.step_profiler          import StepProfiler

from tests.conftest import SilentLogger

pytestmark = pytest.mark.usefixtures("force_cpu")


def _entry(tmp_path) -> TrainingProfileEntryConfig:
    entry            = TrainingProfileEntryConfig(batch_size=16, num_workers=0, warmup_steps=1, active_steps=2, with_stack=False)
    entry.output_dir = tmp_path / "profiles"

    return entry


def _feed() -> FeedTarget:
    return FeedTarget(
        dataset        = TensorDataset(torch.randn(64, 2, 4, 4)),
        model          = torch.nn.Conv2d(2, 3, kernel_size=1),
        to_model_input = FeedAdapter.to_model_input,
        forward_loss   = FeedAdapter.forward_loss,
        model_name     = "conv",
        sample_text    = "64 samples",
        item_source    = "TensorDataset",
        config_hint    = "none",
    )


def test_pipeline_writes_trace_table_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(FeedAdapter, "build", lambda self: _feed())

    summary = TrainingProfilePipeline(_entry(tmp_path), SilentLogger()).run()

    stamps = list((tmp_path / "profiles").iterdir())
    stamp  = next(d for d in stamps if (d / "summary.json").is_file())

    payload = json.loads((stamp / "summary.json").read_text())

    assert (stamp / "trace.json").is_file()
    assert (stamp / "op_table.txt").stat().st_size > 0
    assert summary["active_steps"] == 2
    assert payload["top_operators"]
    assert all(entry["count"] > 0 for entry in payload["top_operators"])


def test_profiler_cpu_activities_only_without_cuda(tmp_path):
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
