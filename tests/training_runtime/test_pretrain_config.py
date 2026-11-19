
from __future__ import annotations

from configuration.training   import BackboneEntryConfig
from tools.runtime.config_cli import ConfigCli


_EXPECTED_FLAGS = (
    "pretrain.find_batch_size",
    "pretrain.tune_loader",
    "pretrain.find_lr",
    "pretrain.vram_budget_gb",
    "pretrain.max_batch",
    "pretrain.lr_scale_min",
    "pretrain.lr_scale_max",
)


def test_pretrain_flags_exposed_on_every_BackboneEntryConfig():
    paths = dict(ConfigCli._leaves(BackboneEntryConfig()))

    for flag in _EXPECTED_FLAGS:
        assert flag in paths


def test_pretrain_defaults_are_disabled():
    config = BackboneEntryConfig()

    assert config.pretrain.find_batch_size is False
    assert config.pretrain.tune_loader     is False
    assert config.pretrain.find_lr         is False
