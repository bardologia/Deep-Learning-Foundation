
from __future__ import annotations

from pipelines.shared.model.model_builder import ModelBuilder


def test_config_from_registry_applies_known_overrides():
    config = ModelBuilder.config_from_registry("resunet", {"dropout": 0.25})

    assert config.dropout == 0.25


def test_image_size_override_empty_for_convolutional_backbone():
    assert ModelBuilder.image_size_override("resunet", (64, 128)) == {}


def test_image_size_override_square_patch_yields_scalar():
    assert ModelBuilder.image_size_override("swin_unet", (64, 64)) == {"image_size": 64}


def test_image_size_override_rectangular_patch_takes_the_bigger_dimension():
    assert ModelBuilder.image_size_override("unetr", (64, 128)) == {"image_size": 128}


def test_square_patch_squares_size_and_stride_for_square_only_backbones():
    assert ModelBuilder.square_patch("swin_unet", (64, 32), (32, 16)) == ((64, 64), (32, 32))


def test_square_patch_keeps_rectangles_for_convolutional_backbones():
    assert ModelBuilder.square_patch("resunet", (64, 32), (32, 16)) == ((64, 32), (32, 16))
