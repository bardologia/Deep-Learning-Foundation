from __future__ import annotations

import pytest

from configuration.training   import BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig
from tools.runtime.config_cli import ConfigCli


_ENTRY_CONFIGS = [BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig]

_EXPECTED_FLAGS = (
    "pretrain.find_batch_size",
    "pretrain.tune_loader",
    "pretrain.vram_budget_gb",
    "pretrain.max_batch",
)


@pytest.mark.parametrize("entry_config", _ENTRY_CONFIGS)
def test_pretrain_flags_exposed_on_every_entry_config(entry_config):
    paths = dict(ConfigCli._leaves(entry_config()))

    for flag in _EXPECTED_FLAGS:
        assert flag in paths


@pytest.mark.parametrize("entry_config", _ENTRY_CONFIGS)
def test_pretrain_defaults_are_disabled(entry_config):
    config = entry_config()

    assert config.pretrain.find_batch_size is False
    assert config.pretrain.tune_loader     is False
