"""Shared fixtures and stubs for the training runtime tests.

Provides a silent logger, a recording experiment tracker, a tiny linear model,
and a SimpleNamespace factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.conftest import SilentLogger


class RecordingTracker:
    """Experiment tracker stub that stores everything it is asked to log.

    Attributes:
        debug      : Whether debug-level logging is enabled.
        scalars    : Recorded (tag, value, step) scalar entries.
        metrics    : Recorded (tag, mapping, step) metric groups.
        histograms : Recorded (tag, values, step) histogram entries.
    """
    def __init__(self, debug: bool = False):
        """Initialises empty scalar, metric, and histogram records."""
        self.debug      = debug
        self.scalars    = []
        self.metrics    = []
        self.histograms = []

    def log_scalar(self, tag, value, step):
        """Records a scalar under its tag and step."""
        self.scalars.append((tag, value, step))

    def log_metrics(self, tag, mapping, step):
        """Records a copy of a metric mapping under its tag and step."""
        self.metrics.append((tag, dict(mapping), step))

    def log_histogram(self, tag, values, step):
        """Records a histogram value set under its tag and step."""
        self.histograms.append((tag, values, step))


class TinyModel(torch.nn.Module):
    """Single linear layer standing in for a trainable model.

    Attributes:
        linear : The wrapped linear layer.
    """
    def __init__(self, in_features: int = 4, out_features: int = 3):
        """Builds the linear layer mapping in_features to out_features."""
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        """Applies the linear layer.

        Args:
            x : Input of shape (batch, in_features).

        Returns:
            Output of shape (batch, out_features).
        """
        return self.linear(x)


@pytest.fixture
def logger():
    """Returns a logger that swallows every message."""
    return SilentLogger()


@pytest.fixture
def tracker():
    """Returns a fresh recording tracker."""
    return RecordingTracker()


@pytest.fixture
def tiny_model():
    """Returns a seeded TinyModel so weights are reproducible."""
    torch.manual_seed(0)
    return TinyModel()


@pytest.fixture
def ns():
    """Returns the SimpleNamespace type for building ad-hoc config objects."""
    return SimpleNamespace
