from __future__ import annotations

import json

import pytest
import torch

from torch.utils.data import TensorDataset

from configuration.diagnostics              import TrainingProfileEntryConfig
from pipelines.backbone.training.profiling import FeedAdapter, FeedTarget, TrainingProfilePipeline
from tools.diagnostics.step_profiler       import StepProfiler

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
    )


def test_the_adapter_defaults_to_its_model_when_none_is_named():
    adapter = FeedAdapter(TrainingProfileEntryConfig(), ".", SilentLogger())

    assert adapter.model_name == FeedAdapter.DEFAULT_MODEL


def test_an_explicit_model_name_overrides_the_default():
    adapter = FeedAdapter(TrainingProfileEntryConfig(model_name="unet"), ".", SilentLogger())

    assert adapter.model_name == "unet"


def test_the_model_input_is_the_first_batch_element():
    inputs = torch.ones(2, 3, 4, 4)

    model_input = FeedAdapter.to_model_input((inputs, torch.zeros(2)), torch.device("cpu"))

    assert torch.equal(model_input, inputs)


def test_the_feed_loss_is_an_mse_surrogate_against_zeros():
    class Offset(torch.nn.Module):
        def forward(self, x):
            return x + 3.0

    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert float(FeedAdapter.forward_loss(Offset(), inputs)) == pytest.approx(float(((inputs + 3.0) ** 2).mean()))


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
