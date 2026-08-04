from __future__ import annotations

import pytest
import torch

from models.backbone                      import BACKBONE_CONFIG_REGISTRY, BACKBONE_MODEL_REGISTRY
from pipelines.shared.model.model_builder import ModelBuilder


def test_config_from_registry_applies_known_overrides():
    config = ModelBuilder.config_from_registry("resunet", {"dropout": 0.25})

    assert config.dropout == 0.25


def test_config_from_registry_rejects_unknown_override():
    with pytest.raises(AttributeError, match="encodr_lr"):
        ModelBuilder.config_from_registry("resunet", {"encodr_lr": 1e-4})


def test_all_groups_lr_fans_out_to_every_lr_field():
    config = ModelBuilder.config_from_registry("resunet", {"all_groups_lr": 1e-5})

    assert config.encoder_lr == 1e-5
    assert config.bottleneck_lr == 1e-5
    assert config.decoder_lr == 1e-5
    assert config.output_head_lr == 1e-5


def test_all_groups_lr_covers_architecture_specific_groups():
    deeplab = ModelBuilder.config_from_registry("deeplabv3plus", {"all_groups_lr": 1e-5})
    mlp     = ModelBuilder.config_from_registry("pixel_mlp", {"all_groups_lr": 1e-5})

    assert deeplab.aspp_lr == 1e-5
    assert mlp.trunk_lr == 1e-5
    assert mlp.output_head_lr == 1e-5


def test_all_groups_lr_accepted_by_every_registry_backbone():
    for name in BACKBONE_CONFIG_REGISTRY:
        config = ModelBuilder.config_from_registry(name, {"all_groups_lr": 1e-5})
        assert config.output_head_lr == 1e-5


def test_explicit_group_lr_wins_over_the_fanout():
    config = ModelBuilder.config_from_registry("resunet", {"all_groups_lr": 1e-5, "output_head_lr": 1e-3})

    assert config.encoder_lr == 1e-5
    assert config.output_head_lr == 1e-3


def test_all_groups_wd_fans_out_to_every_wd_field():
    config = ModelBuilder.config_from_registry("resunet", {"all_groups_wd": 0.0})

    assert config.encoder_wd == 0.0
    assert config.bottleneck_wd == 0.0
    assert config.decoder_wd == 0.0
    assert config.output_head_wd == 0.0


def test_all_groups_wd_accepted_by_every_registry_backbone():
    for name in BACKBONE_CONFIG_REGISTRY:
        config = ModelBuilder.config_from_registry(name, {"all_groups_wd": 0.0})
        assert config.output_head_wd == 0.0


def test_legacy_unet_overrides_build_a_valid_config():
    overrides = {"all_groups_lr": 1e-5, "all_groups_wd": 0.0, "features": [64, 128, 256, 512], "bottleneck_factor": 2, "dropout": 0.0, "normalization": "none", "conv_bias": True}
    config    = ModelBuilder.config_from_registry("unet", dict(overrides), head="conv")

    assert config.head == "conv"
    assert config.encoder_lr == 1e-5
    assert config.encoder_wd == 0.0
    assert config.normalization == "none"
    assert config.conv_bias is True
    assert config.dropout == 0.0


def test_legacy_unet_runs_forward_on_the_legacy_shape():
    overrides = {"all_groups_lr": 1e-5, "all_groups_wd": 0.0, "features": [64, 128, 256, 512], "bottleneck_factor": 2, "dropout": 0.0, "normalization": "none", "conv_bias": True, "in_channels": 8, "out_channels": 6}
    config    = ModelBuilder.config_from_registry("unet", overrides, head="conv")
    model     = BACKBONE_MODEL_REGISTRY["unet"](config)

    with torch.no_grad():
        out = model(torch.rand(1, 8, 32, 32))

    assert out.shape == (1, 6, 32, 32)


def test_image_size_override_empty_for_convolutional_backbone():
    assert ModelBuilder.image_size_override("resunet", (64, 128)) == {}


def test_image_size_override_square_patch_yields_scalar():
    assert ModelBuilder.image_size_override("swin_unet", (64, 64)) == {"image_size": 64}


def test_image_size_override_rejects_rectangular_patch():
    with pytest.raises(ValueError, match="rectangular"):
        ModelBuilder.image_size_override("unetr", (64, 128))
