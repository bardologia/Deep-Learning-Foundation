from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.conftest import SilentLogger


class RecordingTracker:
    def __init__(self, debug: bool = False):
        self.debug      = debug
        self.scalars    = []
        self.metrics    = []
        self.histograms = []

    def log_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def log_metrics(self, tag, mapping, step):
        self.metrics.append((tag, dict(mapping), step))

    def log_histogram(self, tag, values, step):
        self.histograms.append((tag, values, step))


class TinyModel(torch.nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def logger():
    return SilentLogger()


@pytest.fixture
def tracker():
    return RecordingTracker()


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyModel()


@pytest.fixture
def ns():
    return SimpleNamespace
