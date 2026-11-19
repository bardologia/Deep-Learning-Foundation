
from __future__ import annotations

import dataclasses

import pytest

from models.backbone import BACKBONE_CONFIG_REGISTRY, BACKBONE_MODEL_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS


REGISTRY_ENTRIES = [pytest.param(key, factory, id=key) for key, factory in BACKBONE_CONFIG_REGISTRY.items()]


@pytest.mark.parametrize("key, factory", REGISTRY_ENTRIES)
def test_registry_entry_is_dataclass_type(key, factory):
    assert isinstance(factory, type)
    assert dataclasses.is_dataclass(factory)


@pytest.mark.parametrize("key, factory", REGISTRY_ENTRIES)
def test_registry_entry_constructs_with_defaults(key, factory):
    instance = factory()
    assert dataclasses.is_dataclass(instance)


@pytest.mark.parametrize("key, factory", REGISTRY_ENTRIES)
def test_registry_entry_asdict_round_trips(key, factory):
    instance = factory()
    payload  = dataclasses.asdict(instance)
    rebuilt  = factory(**{f.name: payload[f.name] for f in dataclasses.fields(instance) if f.name != "shape_logger_types"})

    assert isinstance(rebuilt, factory)


@pytest.mark.parametrize("key, factory", REGISTRY_ENTRIES)
def test_registry_entry_tunable_params_are_dicts(key, factory):
    assert isinstance(factory.tunable_lr_params(),   dict)
    assert isinstance(factory.tunable_arch_params(), dict)


def test_backbone_config_and_model_registries_share_keys():
    assert set(BACKBONE_CONFIG_REGISTRY) == set(BACKBONE_MODEL_REGISTRY)


def test_image_size_models_are_known_backbones():
    assert BACKBONE_IMAGE_SIZE_MODELS.issubset(set(BACKBONE_CONFIG_REGISTRY))


@pytest.mark.parametrize("key, factory", REGISTRY_ENTRIES)
def test_tunable_arch_param_keys_are_config_fields(key, factory):
    field_names = {f.name for f in dataclasses.fields(factory)}
    for param_name in factory.tunable_arch_params():
        assert param_name in field_names


@pytest.mark.parametrize("key, factory", REGISTRY_ENTRIES)
def test_tunable_lr_param_keys_are_config_fields(key, factory):
    field_names = {f.name for f in dataclasses.fields(factory)}
    for param_name in factory.tunable_lr_params():
        assert param_name in field_names
