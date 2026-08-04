from __future__ import annotations

from pathlib import Path

import pytest

from configuration.architectures       import image_autoencoder, profile_autoencoder
from image_autoencoder_model_library   import ImageAutoencoderModelLibrary
from jepa_model_library                import JepaModelLibrary
from model_library_base                import ModelNoteLibrary
from models                            import IMAGE_AE_MODEL_REGISTRY, PROFILE_AE_MODEL_REGISTRY
from profile_autoencoder_model_library import ProfileAutoencoderModelLibrary


REGISTERED = [
    (ProfileAutoencoderModelLibrary, PROFILE_AE_MODEL_REGISTRY),
    (ImageAutoencoderModelLibrary,   IMAGE_AE_MODEL_REGISTRY),
]

LIBRARIES = [ProfileAutoencoderModelLibrary, ImageAutoencoderModelLibrary, JepaModelLibrary]


def _model_keys(collected: list) -> list:
    return [model["key"] for family in collected for model in family["models"]]


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_config_classes_cover_the_registry(library_class, registry):
    assert set(library_class.CONFIG_CLASSES) == set(registry)


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_note_files_cover_the_registry(library_class, registry):
    assert set(library_class.NOTE_FILES) == set(registry)


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_families_cover_the_registry_exactly_once(library_class, registry):
    keys = _model_keys(library_class().collect())

    assert sorted(keys) == sorted(set(keys))
    assert set(keys)    == set(registry)


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_every_config_class_exists_on_its_module(library_class, registry):
    missing = [name for name in library_class.CONFIG_CLASSES.values() if not hasattr(library_class.CONFIG_MODULE, name)]

    assert not missing, f"config classes named by the library but absent from the module: {missing}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_every_model_resolves_a_note(library_class):
    library = library_class()
    missing = [key for key in library_class.NOTE_FILES if library.note(key) is None]

    assert not missing, f"models without a resolvable note under notes/models: {missing}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_the_families_name_exactly_one_recommended_model(library_class):
    for family in library_class().collect():
        recommended = [model["key"] for model in family["models"] if model["recommended"]]

        assert len(recommended) == 1, f"{family['family']} recommends {recommended}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_every_model_card_is_filled_in(library_class):
    for family in library_class().collect():
        assert family["family"] and family["blurb"]

        for model in family["models"]:
            for field in ("key", "name", "skip", "head", "params", "when"):
                assert model[field], f"{model['key']} has an empty {field}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_an_unknown_key_has_no_note(library_class):
    assert library_class().note("no_such_model") is None


def test_a_note_links_every_sibling_model():
    library = ProfileAutoencoderModelLibrary()
    note    = library.note("mlp_ae")

    assert note["key"]  == "mlp_ae"
    assert note["note"] == "MLP Autoencoder"
    assert note["markdown"].strip()
    assert note["links"] == {name[:-3]: key for key, name in ProfileAutoencoderModelLibrary.NOTE_FILES.items()}


def test_the_profile_library_reads_the_embedding_norm_attribute():
    labels = {model["key"]: model for family in ProfileAutoencoderModelLibrary().collect() for model in family["models"]}
    config = profile_autoencoder.MlpAutoencoderConfig()

    assert ProfileAutoencoderModelLibrary.NORMALIZATION_ATTR == "embedding_norm"
    assert labels["mlp_ae"]["normalization"] == str(config.embedding_norm).lower()
    assert labels["mlp_ae"]["activation"]    == str(config.activation).lower()


def test_the_image_library_reads_the_plain_normalization_attribute():
    labels = {model["key"]: model for family in ImageAutoencoderModelLibrary().collect() for model in family["models"]}
    config = image_autoencoder.Conv2dImageAutoencoderConfig()

    assert ImageAutoencoderModelLibrary.NORMALIZATION_ATTR == "normalization"
    assert labels["conv2d_ae"]["normalization"] == str(config.normalization).lower()


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_every_label_is_resolved_from_a_real_config(library_class, registry):
    for family in library_class().collect():
        for model in family["models"]:
            assert model["activation"],    f"{model['key']} has no activation label"
            assert model["normalization"], f"{model['key']} has no normalization label"


class TempNoteLibrary(ModelNoteLibrary):

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
