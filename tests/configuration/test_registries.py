from __future__ import annotations

import dataclasses

import pytest

from models.backbone            import BACKBONE_CONFIG_REGISTRY, BACKBONE_MODEL_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS
from models.profile_autoencoder import PROFILE_AE_CONFIG_REGISTRY, PROFILE_AE_MODEL_REGISTRY
from models.image_autoencoder   import IMAGE_AE_CONFIG_REGISTRY, IMAGE_AE_MODEL_REGISTRY
from models.unrolled            import UNROLLED_CONFIG_REGISTRY, UNROLLED_MODEL_REGISTRY
from models.dual                import DUAL_CONFIG_REGISTRY, DUAL_MODEL_REGISTRY
from models                     import config_registry


ALL_REGISTRIES = {
    "backbone"   : BACKBONE_CONFIG_REGISTRY,
    "profile_ae" : PROFILE_AE_CONFIG_REGISTRY,
    "image_ae"   : IMAGE_AE_CONFIG_REGISTRY,
    "unrolled"   : UNROLLED_CONFIG_REGISTRY,
    "dual"       : DUAL_CONFIG_REGISTRY,
}

REGISTRY_ENTRIES = [
    pytest.param(reg_name, key, factory, id=f"{reg_name}-{key}")
    for reg_name, registry in ALL_REGISTRIES.items()
    for key, factory in registry.items()
]


@pytest.mark.parametrize("reg_name, key, factory", REGISTRY_ENTRIES)
def test_registry_entry_is_dataclass_type(reg_name, key, factory):
    assert isinstance(factory, type)
    assert dataclasses.is_dataclass(factory)


@pytest.mark.parametrize("reg_name, key, factory", REGISTRY_ENTRIES)
def test_registry_entry_constructs_with_defaults(reg_name, key, factory):
    instance = factory()
    assert dataclasses.is_dataclass(instance)


@pytest.mark.parametrize("reg_name, key, factory", REGISTRY_ENTRIES)
def test_registry_entry_asdict_round_trips(reg_name, key, factory):
    instance = factory()
    payload  = dataclasses.asdict(instance)
    rebuilt  = factory(**{f.name: payload[f.name] for f in dataclasses.fields(instance) if f.name != "shape_logger_types"})

    assert isinstance(rebuilt, factory)


@pytest.mark.parametrize("reg_name, key, factory", REGISTRY_ENTRIES)
def test_registry_entry_tunable_params_are_dicts(reg_name, key, factory):
    assert isinstance(factory.tunable_lr_params(),   dict)
    assert isinstance(factory.tunable_arch_params(), dict)


def test_backbone_config_and_model_registries_share_keys():
    assert set(BACKBONE_CONFIG_REGISTRY) == set(BACKBONE_MODEL_REGISTRY)


def test_profile_ae_config_and_model_registries_share_keys():
    assert set(PROFILE_AE_CONFIG_REGISTRY) == set(PROFILE_AE_MODEL_REGISTRY)


def test_image_ae_config_and_model_registries_share_keys():
    assert set(IMAGE_AE_CONFIG_REGISTRY) == set(IMAGE_AE_MODEL_REGISTRY)


def test_unrolled_config_and_model_registries_share_keys():
    assert set(UNROLLED_CONFIG_REGISTRY) == set(UNROLLED_MODEL_REGISTRY)


def test_dual_config_and_model_registries_share_keys():
    assert set(DUAL_CONFIG_REGISTRY) == set(DUAL_MODEL_REGISTRY)


def test_image_size_models_are_known_backbones():
    assert BACKBONE_IMAGE_SIZE_MODELS.issubset(set(BACKBONE_CONFIG_REGISTRY))


def test_config_registry_dispatch_matches_named_registries():
    assert config_registry("backbone")            is BACKBONE_CONFIG_REGISTRY
    assert config_registry("profile_autoencoder") is PROFILE_AE_CONFIG_REGISTRY
    assert config_registry("image_autoencoder")   is IMAGE_AE_CONFIG_REGISTRY
    assert config_registry("unrolled")            is UNROLLED_CONFIG_REGISTRY
    assert config_registry("dual")                is DUAL_CONFIG_REGISTRY


def test_config_registry_unknown_falls_back_to_backbone():
    assert config_registry("something_else") is BACKBONE_CONFIG_REGISTRY


@pytest.mark.parametrize("reg_name, key, factory", REGISTRY_ENTRIES)
def test_tunable_arch_param_keys_are_config_fields(reg_name, key, factory):
    field_names = {f.name for f in dataclasses.fields(factory)}
    for param_name in factory.tunable_arch_params():
        assert param_name in field_names


@pytest.mark.parametrize("reg_name, key, factory", REGISTRY_ENTRIES)
def test_tunable_lr_param_keys_are_config_fields(reg_name, key, factory):
    field_names = {f.name for f in dataclasses.fields(factory)}
    for param_name in factory.tunable_lr_params():
        assert param_name in field_names
