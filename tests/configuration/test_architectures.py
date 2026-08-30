"""Tests for the architecture configuration dataclasses.

Covers every backbone, image-autoencoder and profile-autoencoder config: that it
instantiates with defaults, that its channel, dropout, learning-rate and width
fields are sane, and that its field values round trip through the constructor.
"""

from __future__ import annotations

import dataclasses

import pytest

from configuration   import architectures as arch
from models.backbone import BACKBONE_CONFIG_REGISTRY


BACKBONE_CONFIGS = [BACKBONE_CONFIG_REGISTRY[name] for name in sorted(BACKBONE_CONFIG_REGISTRY)]

IMAGE_AE_CONFIGS = [
    arch.ImageAutoencoderBaseConfig,
    arch.Conv2dImageAutoencoderConfig,
    arch.ResNet2dImageAutoencoderConfig,
    arch.ConvNeXt2dImageAutoencoderConfig,
    arch.DilatedConv2dImageAutoencoderConfig,
    arch.ViTImageAutoencoderConfig,
]

PROFILE_AE_CONFIGS = [
    arch.ProfileAutoencoderBaseConfig,
    arch.MlpAutoencoderConfig,
    arch.Conv1dAutoencoderConfig,
    arch.Transformer1dAutoencoderConfig,
    arch.ResMlpAutoencoderConfig,
    arch.TcnAutoencoderConfig,
    arch.GruAutoencoderConfig,
    arch.CnnAttnAutoencoderConfig,
]

ALL_ARCH_CONFIGS = BACKBONE_CONFIGS + IMAGE_AE_CONFIGS + PROFILE_AE_CONFIGS

ARCH_IDS = [c.__name__ for c in ALL_ARCH_CONFIGS]


@pytest.mark.parametrize("config_cls", ALL_ARCH_CONFIGS, ids=ARCH_IDS)
def test_arch_config_instantiates_with_defaults(config_cls):
    """Verifies every architecture config builds from defaults as a dataclass."""
    instance = config_cls()
    assert dataclasses.is_dataclass(instance)


@pytest.mark.parametrize("config_cls", BACKBONE_CONFIGS, ids=[c.__name__ for c in BACKBONE_CONFIGS])
def test_backbone_default_channels_are_positive(config_cls):
    """Verifies backbone configs default to positive channel counts, three parameters per Gaussian and the conv head."""
    instance = config_cls()
    assert instance.in_channels         > 0
    assert instance.out_channels        > 0
    assert instance.params_per_gaussian == 3
    assert instance.head                == "conv"


@pytest.mark.parametrize("config_cls", BACKBONE_CONFIGS, ids=[c.__name__ for c in BACKBONE_CONFIGS])
def test_backbone_dropout_in_unit_range(config_cls):
    """Verifies a backbone dropout rate, where present, lies in the unit interval."""
    instance = config_cls()
    if not hasattr(instance, "dropout"):
        pytest.skip("config has no dropout field")
    assert 0.0 <= instance.dropout <= 1.0


@pytest.mark.parametrize("config_cls", BACKBONE_CONFIGS, ids=[c.__name__ for c in BACKBONE_CONFIGS])
def test_backbone_learning_rates_positive(config_cls):
    """Verifies every backbone learning-rate field exists and is positive."""
    instance  = config_cls()
    lr_fields = [f.name for f in dataclasses.fields(instance) if f.name.endswith("_lr")]
    assert lr_fields
    for name in lr_fields:
        assert getattr(instance, name) > 0


@pytest.mark.parametrize("config_cls", BACKBONE_CONFIGS, ids=[c.__name__ for c in BACKBONE_CONFIGS])
def test_backbone_features_or_dims_present(config_cls):
    """Verifies every backbone config carries some width or embedding-size field."""
    instance = config_cls()
    has_features = any(
        hasattr(instance, attr)
        for attr in ("features", "cnn_features", "decoder_features", "embedding_dim",
                     "embedding_dims", "base_channels", "growth_rate", "width")
    )
    assert has_features


@pytest.mark.parametrize("config_cls", PROFILE_AE_CONFIGS, ids=[c.__name__ for c in PROFILE_AE_CONFIGS])
def test_profile_ae_embedding_and_length_positive(config_cls):
    """Verifies profile autoencoders default to a positive embedding size and profile length."""
    instance = config_cls()
    assert instance.embedding_dim  > 0
    assert instance.profile_length > 0


@pytest.mark.parametrize("config_cls", IMAGE_AE_CONFIGS, ids=[c.__name__ for c in IMAGE_AE_CONFIGS])
def test_image_ae_embedding_and_channels_positive(config_cls):
    """Verifies image autoencoders default to a positive embedding size and input channel count."""
    instance = config_cls()
    assert instance.embedding_dim > 0
    assert instance.in_channels   > 0


@pytest.mark.parametrize("config_cls", ALL_ARCH_CONFIGS, ids=ARCH_IDS)
def test_arch_config_asdict_round_trips(config_cls):
    """Verifies an architecture config rebuilt from its own field values compares equal field by field."""
    instance = config_cls()
    payload  = {
        f.name: getattr(instance, f.name)
        for f in dataclasses.fields(instance)
        if f.name != "shape_logger_types"
    }
    rebuilt  = config_cls(**payload)

    assert {f.name: getattr(rebuilt, f.name) for f in dataclasses.fields(rebuilt)} == {f.name: getattr(instance, f.name) for f in dataclasses.fields(instance)}


@pytest.mark.parametrize("config_cls", BACKBONE_CONFIGS, ids=[c.__name__ for c in BACKBONE_CONFIGS])
def test_backbone_shape_logger_types_is_tuple(config_cls):
    """Verifies the shape-logger types of a backbone config are a non-empty tuple."""
    instance = config_cls()
    assert isinstance(instance.shape_logger_types, tuple)
    assert len(instance.shape_logger_types) > 0


def test_ae_subclasses_inherit_base_lr_fields():
    """Verifies every autoencoder config carries the fields of its base config."""
    base_fields = {f.name for f in dataclasses.fields(arch.ProfileAutoencoderBaseConfig)}
    for cls in PROFILE_AE_CONFIGS:
        assert base_fields.issubset({f.name for f in dataclasses.fields(cls)})

    base_image_fields = {f.name for f in dataclasses.fields(arch.ImageAutoencoderBaseConfig)}
    for cls in IMAGE_AE_CONFIGS:
        assert base_image_fields.issubset({f.name for f in dataclasses.fields(cls)})
