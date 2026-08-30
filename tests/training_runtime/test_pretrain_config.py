"""Tests that the pre-training flags are exposed on every training entry config.

Covers presence of the pretrain leaf paths and their disabled-by-default state.
"""

from __future__ import annotations

import pytest

from configuration.training   import BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig
from tools.runtime.config_cli import ConfigCli


_ENTRY_CONFIGS = [BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig]

_EXPECTED_FLAGS = (
    "pretrain.find_batch_size",
    "pretrain.tune_loader",
    "pretrain.find_lr",
    "pretrain.vram_budget_gb",
    "pretrain.max_batch",
    "pretrain.lr_scale_min",
    "pretrain.lr_scale_max",
)


@pytest.mark.parametrize("entry_config", _ENTRY_CONFIGS)
def test_pretrain_flags_exposed_on_every_entry_config(entry_config):
    """Verifies that every pretrain flag is reachable as a CLI leaf path."""
    paths = dict(ConfigCli._leaves(entry_config()))

    for flag in _EXPECTED_FLAGS:
        assert flag in paths


@pytest.mark.parametrize("entry_config", _ENTRY_CONFIGS)
def test_pretrain_defaults_are_disabled(entry_config):
    """Verifies that batch finding, loader tuning and the LR probe default to off."""
    config = entry_config()

    assert config.pretrain.find_batch_size is False
    assert config.pretrain.tune_loader     is False
    assert config.pretrain.find_lr         is False
