from __future__ import annotations

import pytest
import torch

from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from pipelines.dataloader_tuning.adapters      import FeedAdapter

from tests.conftest import SilentLogger


def test_the_adapter_defaults_to_its_model_when_none_is_named():
    adapter = FeedAdapter(DataLoaderTuningEntryConfig(), ".", SilentLogger())

    assert adapter.model_name == FeedAdapter.DEFAULT_MODEL


def test_an_explicit_model_name_overrides_the_default():
    adapter = FeedAdapter(DataLoaderTuningEntryConfig(model_name="unet"), ".", SilentLogger())

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
