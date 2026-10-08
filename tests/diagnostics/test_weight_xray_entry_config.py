"""Tests that the weight x-ray entry config forwards thresholds without duplicating them."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from configuration.diagnostics import WeightXrayEntryConfig, WeightXrayThresholds


def test_entry_config_forwards_every_threshold():
    """The threshold object is passed through to the resolved config unchanged."""
    entry = WeightXrayEntryConfig()

    for index, field in enumerate(dataclasses.fields(WeightXrayThresholds)):
        setattr(entry.thresholds, field.name, float(index + 1))

    config = entry.to_config(Path("/runs/unet"))

    assert config.thresholds is entry.thresholds
    assert dataclasses.asdict(config.thresholds) == dataclasses.asdict(entry.thresholds)


def test_entry_config_declares_no_threshold_of_its_own():
    """The entry config declares no field that shadows a threshold field."""
    entry_fields     = {field.name for field in dataclasses.fields(WeightXrayEntryConfig)}
    threshold_fields = {field.name for field in dataclasses.fields(WeightXrayThresholds)}

    assert entry_fields & threshold_fields == set()
