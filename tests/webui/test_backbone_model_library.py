
from __future__ import annotations

from pathlib import Path

import pytest

from backbone_model_library      import BackboneModelLibrary
from configuration.architectures import backbone
from models                      import BACKBONE_MODEL_REGISTRY


@pytest.fixture(scope="module")
def library():
    return BackboneModelLibrary()


@pytest.fixture(scope="module")
def collected(library):
    return library.collect()


def _model_keys(collected):
    return [model["key"] for family in collected for model in family["models"]]


def test_config_classes_cover_the_backbone_registry():
    assert set(BackboneModelLibrary.CONFIG_CLASSES) == set(BACKBONE_MODEL_REGISTRY)


def test_note_files_cover_the_backbone_registry():
    assert set(BackboneModelLibrary.NOTE_FILES) == set(BACKBONE_MODEL_REGISTRY)


def test_families_cover_the_backbone_registry_exactly_once(collected):
    keys = _model_keys(collected)

    assert sorted(keys) == sorted(set(keys))
    assert set(keys) == set(BACKBONE_MODEL_REGISTRY)


def test_every_model_resolves_a_note(library):
    missing = [key for key in BackboneModelLibrary.NOTE_FILES if library.note(key) is None]

    assert not missing, f"models without a resolvable note under notes/models: {missing}"


def test_collect_resolves_display_defaults(collected):
    for family in collected:
        for model in family["models"]:
            assert model["activation"], f"{model['key']} has no activation label"
            assert model["normalization"], f"{model['key']} has no normalization label"


def _arch(collected):
    return {model["key"]: {entry["name"]: entry for entry in model["arch"]} for family in collected for model in family["models"]}


def test_arch_hides_the_pipeline_owned_fields(collected):
    for key, entries in _arch(collected).items():
        for name in ("in_channels", "out_channels", "image_size", "shape_logger_types"):
            assert name not in entries, f"{key} exposes pipeline-owned field {name}"


def test_arch_exposes_the_feature_ladders_as_lists(collected):
    arch = _arch(collected)

    assert arch["unet"]["features"]["kind"]    == "list"
    assert arch["unet"]["features"]["default"] == backbone.UNetConfig().features
    assert arch["unet"]["features"]["presets"] == [list(choice) for choice in backbone.UNetConfig.tunable_arch_params()["features"]["choices"]]

    assert arch["nafnet"]["enc_blocks"]["kind"]  == "list"
    assert arch["u2net"]["rsu_heights"]["kind"]  == "list"
    assert arch["u2net"]["rsu_heights"]["default"] == list(backbone.U2NetLiteConfig().rsu_heights)


def test_arch_exposes_the_kernel_choices(collected):
    arch = _arch(collected)

    assert arch["linknet"]["initial_kernel_size"]["presets"] == [3, 5, 7]
    assert arch["local_cnn"]["block_kernels"]["kind"]        == "list"
    assert arch["local_cnn"]["block_kernels"]["default"]     is None


def test_arch_exposes_activation_options_with_the_default_included(collected):
    arch = _arch(collected)

    assert arch["unet"]["activation"]["options"]    == ["relu", "leaky_relu", "gelu", "silu"]
    assert arch["unet"]["normalization"]["options"] == ["batch", "instance", "group"]
    assert str(arch["unet"]["activation"]["default"]) in arch["unet"]["activation"]["options"]


def test_arch_keeps_the_learning_rate_fields_for_the_backbone_page(collected):
    arch = _arch(collected)

    assert "encoder_lr" in arch["unet"]
    assert "trunk_lr"   in arch["local_cnn"]


class TempNoteLibrary(BackboneModelLibrary):

    NOTE_FILES = {"demo": "Demo Model.md", "other": "Other Model.md"}

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _notes_dir(self) -> Path:
        return self.directory


NOTE = """# Demo Model

An encoder-decoder with skip connections.

## Correction

The stride was wrong and got fixed on 2026-07-03.

### Still inside the correction

Dropped as well.

## Architecture

Four levels (reviewed 2026-07-03) of downsampling.
The receptive field is 41 pixels. This claim was flagged in the audit.
"""


@pytest.fixture
def temp_notes(tmp_path):
    (tmp_path / "Demo Model.md").write_text(NOTE)
    return TempNoteLibrary(tmp_path)


def test_a_missing_note_file_yields_nothing(temp_notes):
    assert temp_notes.note("other")  is None
    assert temp_notes.note("absent") is None


def test_a_note_links_every_sibling_model(temp_notes):
    note = temp_notes.note("demo")

    assert note["note"]  == "Demo Model"
    assert note["links"] == {"Demo Model": "demo", "Other Model": "other"}


def test_an_operational_heading_takes_its_whole_subtree_with_it(temp_notes):
    markdown = temp_notes.note("demo")["markdown"]

    assert "## Correction"     not in markdown
    assert "The stride"        not in markdown
    assert "Still inside"      not in markdown
    assert "Dropped as well"   not in markdown
    assert "## Architecture"       in markdown


def test_dated_parentheticals_and_flagged_sentences_are_removed(temp_notes):
    markdown = temp_notes.note("demo")["markdown"]

    assert "(reviewed 2026-07-03)"  not in markdown
    assert "flagged in the audit"   not in markdown
    assert "Four levels of downsampling." in markdown
    assert "The receptive field is 41 pixels." in markdown


def test_stripping_never_leaves_a_run_of_blank_lines(temp_notes):
    assert "\n\n\n" not in temp_notes.note("demo")["markdown"]
