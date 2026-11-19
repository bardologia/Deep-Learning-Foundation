
from __future__ import annotations

import dataclasses
from pathlib import Path

from configuration.diagnostics import WeightXrayEntryConfig, WeightXrayThresholds


def test_entry_config_forwards_every_threshold():
    entry = WeightXrayEntryConfig()

    for index, field in enumerate(dataclasses.fields(WeightXrayThresholds)):
        setattr(entry.thresholds, field.name, float(index + 1))

    config = entry.to_config(Path("/runs/unet"))

    assert config.thresholds is entry.thresholds
    assert dataclasses.asdict(config.thresholds) == dataclasses.asdict(entry.thresholds)


def test_entry_config_declares_no_threshold_of_its_own():
    entry_fields     = {field.name for field in dataclasses.fields(WeightXrayEntryConfig)}
    threshold_fields = {field.name for field in dataclasses.fields(WeightXrayThresholds)}

    assert entry_fields & threshold_fields == set()
