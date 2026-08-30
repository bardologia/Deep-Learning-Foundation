"""Tests that the webui backbone model library stays in step with the model registry.

Checks the config classes, notes, families and heads exposed to the frontend
cover BACKBONE_MODEL_REGISTRY and BACKBONE_HEADS exactly once, and that the
architecture panels hide pipeline-owned fields while surfacing list ladders,
kernel presets, activation options and the per-page learning-rate fields.
"""

from __future__ import annotations

import pytest

from backbone_model_library      import BackboneModelLibrary
from configuration.architectures import backbone
from models                      import BACKBONE_HEADS, BACKBONE_MODEL_REGISTRY


@pytest.fixture(scope="module")
def library():
    """Returns a module-scoped backbone model library."""
    return BackboneModelLibrary()


@pytest.fixture(scope="module")
def collected(library):
    """Returns the library's collected family listing, computed once per module."""
    return library.collect()


def _model_keys(collected):
    """Returns the flat list of model keys across all collected families."""
    return [model["key"] for family in collected for model in family["models"]]


def test_config_classes_cover_the_backbone_registry():
    """Checks the config class map matches the backbone registry key for key."""
    assert set(BackboneModelLibrary.CONFIG_CLASSES) == set(BACKBONE_MODEL_REGISTRY)


def test_note_files_cover_the_backbone_registry():
    """Checks every registered backbone has a note file entry."""
    assert set(BackboneModelLibrary.NOTE_FILES) == set(BACKBONE_MODEL_REGISTRY)


def test_families_cover_the_backbone_registry_exactly_once(collected):
    """Checks the family grouping lists each registered backbone once and none extra."""
    keys = _model_keys(collected)

    assert sorted(keys) == sorted(set(keys))
    assert set(keys) == set(BACKBONE_MODEL_REGISTRY)


def test_every_model_resolves_a_note(library):
    """Checks each declared model note file resolves to a note on disk."""
    missing = [key for key in BackboneModelLibrary.NOTE_FILES if library.note(key) is None]

    assert not missing, f"models without a resolvable note under notes/models: {missing}"


def test_collect_resolves_display_defaults(collected):
    """Checks every collected model carries non-empty activation and normalization labels."""
    for family in collected:
        for model in family["models"]:
            assert model["activation"], f"{model['key']} has no activation label"
            assert model["normalization"], f"{model['key']} has no normalization label"


def test_heads_cover_the_backbone_heads_exactly_once(library):
    """Checks the head listing matches BACKBONE_HEADS without duplicates."""
    keys = [head["key"] for head in library.heads()]

    assert sorted(keys) == sorted(set(keys))
    assert set(keys) == set(BACKBONE_HEADS)


def test_head_note_files_cover_the_backbone_heads():
    """Checks every backbone head has a note file entry."""
    assert set(BackboneModelLibrary.HEAD_NOTE_FILES) == set(BACKBONE_HEADS)


def test_every_head_resolves_a_note(library):
    """Checks each declared head note file resolves to a note on disk."""
    missing = [key for key in BackboneModelLibrary.HEAD_NOTE_FILES if library.note(key) is None]

    assert not missing, f"heads without a resolvable note under notes/models: {missing}"


def _arch(collected):
    """Returns the architecture entries per model key, indexed by field name."""
    return {model["key"]: {entry["name"]: entry for entry in model["arch"]} for family in collected for model in family["models"]}


def test_arch_hides_the_pipeline_owned_fields(collected):
    """Checks channel counts, head, image size and logger types are never exposed as arch fields."""
    for key, entries in _arch(collected).items():
        for name in ("in_channels", "out_channels", "params_per_gaussian", "head", "image_size", "shape_logger_types"):
            assert name not in entries, f"{key} exposes pipeline-owned field {name}"


def test_arch_exposes_the_feature_ladders_as_lists(collected):
    """Checks feature and block ladders are typed as lists with defaults and presets from the configs."""
    arch = _arch(collected)

    assert arch["unet"]["features"]["kind"]    == "list"
    assert arch["unet"]["features"]["default"] == backbone.UNetConfig().features
    assert arch["unet"]["features"]["presets"] == [list(choice) for choice in backbone.UNetConfig.tunable_arch_params()["features"]["choices"]]

    assert arch["nafnet"]["enc_blocks"]["kind"]  == "list"
    assert arch["u2net"]["rsu_heights"]["kind"]  == "list"
    assert arch["u2net"]["rsu_heights"]["default"] == list(backbone.U2NetLiteConfig().rsu_heights)


def test_arch_exposes_the_kernel_choices(collected):
    """Checks kernel size presets and the optional per-block kernel list are exposed."""
    arch = _arch(collected)

    assert arch["linknet"]["initial_kernel_size"]["presets"] == [3, 5, 7]
    assert arch["local_cnn"]["block_kernels"]["kind"]        == "list"
    assert arch["local_cnn"]["block_kernels"]["default"]     is None


def test_arch_exposes_activation_options_with_the_default_included(collected):
    """Checks activation and normalization options are listed and the default is one of them."""
    arch = _arch(collected)

    assert arch["unet"]["activation"]["options"]    == ["relu", "leaky_relu", "gelu", "silu"]
    assert arch["unet"]["normalization"]["options"] == ["batch", "instance", "group"]
    assert str(arch["unet"]["activation"]["default"]) in arch["unet"]["activation"]["options"]


def test_arch_keeps_the_learning_rate_fields_for_the_backbone_page(collected):
    """Checks encoder and trunk learning rates remain visible in the architecture panels."""
    arch = _arch(collected)

    assert "encoder_lr" in arch["unet"]
    assert "trunk_lr"   in arch["local_cnn"]
