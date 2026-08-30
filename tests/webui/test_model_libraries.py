"""Tests that the webui model libraries stay in step with the model registries.

Covers registry coverage of config classes and note files, the completeness of every
model card, the architecture field lists derived from the config dataclasses, and the
markdown stripping applied to the model notes.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib     import Path

import pytest

from configuration.architectures       import image_autoencoder, profile_autoencoder
from image_autoencoder_model_library   import ImageAutoencoderModelLibrary
from jepa_model_library                import JepaModelLibrary
from model_library_base                import ModelDefaultsLibrary, ModelNoteLibrary
from models                            import IMAGE_AE_MODEL_REGISTRY, PROFILE_AE_MODEL_REGISTRY
from profile_autoencoder_model_library import ProfileAutoencoderModelLibrary


REGISTERED = [
    (ProfileAutoencoderModelLibrary, PROFILE_AE_MODEL_REGISTRY),
    (ImageAutoencoderModelLibrary,   IMAGE_AE_MODEL_REGISTRY),
]

LIBRARIES = [ProfileAutoencoderModelLibrary, ImageAutoencoderModelLibrary, JepaModelLibrary]


def _model_keys(collected: list) -> list:
    """Returns the model keys across every collected family."""
    return [model["key"] for family in collected for model in family["models"]]


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_config_classes_cover_the_registry(library_class, registry):
    """The library names a config class for exactly the registered models."""
    assert set(library_class.CONFIG_CLASSES) == set(registry)


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_note_files_cover_the_registry(library_class, registry):
    """The library names a note file for exactly the registered models."""
    assert set(library_class.NOTE_FILES) == set(registry)


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_families_cover_the_registry_exactly_once(library_class, registry):
    """The collected families list every registered model once and nothing else."""
    keys = _model_keys(library_class().collect())

    assert sorted(keys) == sorted(set(keys))
    assert set(keys)    == set(registry)


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_every_config_class_exists_on_its_module(library_class, registry):
    """Every config class the library names is present on its configuration module."""
    missing = [name for name in library_class.CONFIG_CLASSES.values() if not hasattr(library_class.CONFIG_MODULE, name)]

    assert not missing, f"config classes named by the library but absent from the module: {missing}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_every_model_resolves_a_note(library_class):
    """Every model with a declared note file resolves to a readable note."""
    library = library_class()
    missing = [key for key in library_class.NOTE_FILES if library.note(key) is None]

    assert not missing, f"models without a resolvable note under notes/models: {missing}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_the_families_name_exactly_one_recommended_model(library_class):
    """Each family marks exactly one model as recommended."""
    for family in library_class().collect():
        recommended = [model["key"] for model in family["models"] if model["recommended"]]

        assert len(recommended) == 1, f"{family['family']} recommends {recommended}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_every_model_card_is_filled_in(library_class):
    """Every family carries a name and blurb and every model card field is non-empty."""
    for family in library_class().collect():
        assert family["family"] and family["blurb"]

        for model in family["models"]:
            for field in ("key", "name", "skip", "head", "params", "when"):
                assert model[field], f"{model['key']} has an empty {field}"


@pytest.mark.parametrize("library_class", LIBRARIES)
def test_an_unknown_key_has_no_note(library_class):
    """An unregistered model key resolves to no note."""
    assert library_class().note("no_such_model") is None


def test_a_note_links_every_sibling_model():
    """A resolved note carries its key, title, markdown body and links to every sibling note."""
    library = ProfileAutoencoderModelLibrary()
    note    = library.note("mlp_ae")

    assert note["key"]  == "mlp_ae"
    assert note["note"] == "MLP Autoencoder"
    assert note["markdown"].strip()
    assert note["links"] == {name[:-3]: key for key, name in ProfileAutoencoderModelLibrary.NOTE_FILES.items()}


def test_the_profile_library_reads_the_embedding_norm_attribute():
    """The profile library labels normalization from the embedding_norm field and activation from the config activation."""
    labels = {model["key"]: model for family in ProfileAutoencoderModelLibrary().collect() for model in family["models"]}
    config = profile_autoencoder.MlpAutoencoderConfig()

    assert ProfileAutoencoderModelLibrary.NORMALIZATION_ATTR == "embedding_norm"
    assert labels["mlp_ae"]["normalization"] == str(config.embedding_norm).lower()
    assert labels["mlp_ae"]["activation"]    == str(config.activation).lower()


def test_the_image_library_reads_the_plain_normalization_attribute():
    """The image library labels normalization from the plain normalization field."""
    labels = {model["key"]: model for family in ImageAutoencoderModelLibrary().collect() for model in family["models"]}
    config = image_autoencoder.Conv2dImageAutoencoderConfig()

    assert ImageAutoencoderModelLibrary.NORMALIZATION_ATTR == "normalization"
    assert labels["conv2d_ae"]["normalization"] == str(config.normalization).lower()


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_every_label_is_resolved_from_a_real_config(library_class, registry):
    """Every model card carries non-empty activation and normalization labels."""
    for family in library_class().collect():
        for model in family["models"]:
            assert model["activation"],    f"{model['key']} has no activation label"
            assert model["normalization"], f"{model['key']} has no normalization label"


def _expected_arch_names(library, config_class, key) -> list:
    """Returns the config field names an architecture panel should expose for one model.

    Args:
        library      : Model library supplying the exclusion lists.
        config_class : Architecture config dataclass of the model.
        key          : Registry key of the model, used for per-model exclusions.

    Returns:
        Names of the scalar and scalar-sequence fields left after exclusion, in
        dataclass declaration order.
    """
    config   = config_class()
    excluded = set(library.ARCH_EXCLUDED) | set(library.ARCH_EXCLUDED_BY_MODEL.get(key, ()))
    names    = []

    for spec in dataclass_fields(config_class):
        if spec.name in excluded:
            continue

        value    = getattr(config, spec.name)
        scalar   = isinstance(value, (bool, int, float, str))
        sequence = ModelDefaultsLibrary._is_scalar_sequence(value) or (value is None and "list" in str(spec.type))

        if scalar or sequence:
            names.append(spec.name)

    return names


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_arch_fields_cover_every_scalar_and_sequence_config_field(library_class, registry):
    """The exposed architecture entries match the scalar and sequence config fields in declaration order."""
    library = library_class()

    for family in library.collect():
        for model in family["models"]:
            config_class = getattr(library.CONFIG_MODULE, library.CONFIG_CLASSES[model["key"]])

            assert [entry["name"] for entry in model["arch"]] == _expected_arch_names(library, config_class, model["key"])


@pytest.mark.parametrize("library_class,registry", REGISTERED)
def test_every_string_arch_option_list_contains_its_default(library_class, registry):
    """Every architecture entry offering options includes its own default among them."""
    for family in library_class().collect():
        for model in family["models"]:
            for entry in model["arch"]:
                if "options" in entry:
                    assert str(entry["default"]) in entry["options"], f"{model['key']}.{entry['name']}"


def test_the_profile_arch_exposes_the_embedding_and_hides_derived_and_lr_fields():
    """The profile architecture exposes the embedding size, its presets, norm options and hint while hiding dataset-derived and per-group optimiser fields."""
    arch = {model["key"]: {entry["name"]: entry for entry in model["arch"]} for family in ProfileAutoencoderModelLibrary().collect() for model in family["models"]}
    mlp  = profile_autoencoder.MlpAutoencoderConfig()

    assert arch["mlp_ae"]["embedding_dim"]["default"] == mlp.embedding_dim
    assert arch["mlp_ae"]["embedding_dim"]["presets"] == [16, 24, 32, 48]
    assert arch["mlp_ae"]["embedding_norm"]["options"] == ["none", "l2", "layernorm"]
    assert arch["mlp_ae"]["embedding_dim"]["hint"]

    for key in arch:
        assert "profile_length" not in arch[key]
        assert "encoder_lr"     not in arch[key]
        assert "decoder_wd"     not in arch[key]

    assert "num_heads" not in arch["transformer1d_ae"]
    assert "num_heads"     in arch["cnn_attn_ae"]


def test_the_image_arch_hides_the_dataset_derived_channels_and_the_unread_fields():
    """The image architecture hides the dataset-derived channel count and every field a given model does not read."""
    arch = {model["key"]: {entry["name"]: entry for entry in model["arch"]} for family in ImageAutoencoderModelLibrary().collect() for model in family["models"]}

    for key in arch:
        assert "in_channels" not in arch[key]
        assert "encoder_lr"      in arch[key]
        assert "embedding_dim"   in arch[key]

    assert "downsample_factor" not in arch["vit_ae"]
    assert "base_channels"     not in arch["vit_ae"]
    assert "downsample_factor" not in arch["dilated2d_ae"]
    assert "depth"             not in arch["dilated2d_ae"]
    assert "downsample_factor"     in arch["conv2d_ae"]
    assert "base_channels"         in arch["conv2d_ae"]
    assert "depth"                 in arch["conv2d_ae"]


class TempNoteLibrary(ModelNoteLibrary):
    """Note library reading from a temporary directory.

    Attributes:
        directory : Directory holding the markdown note files.
    """

    NOTE_FILES = {"demo": "Demo Model.md", "other": "Other Model.md"}

    def __init__(self, directory: Path) -> None:
        """Stores the directory the notes are read from."""
        self.directory = directory

    def _notes_dir(self) -> Path:
        """Returns the temporary directory holding the notes."""
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
    """Returns a note library over a temporary directory holding the demo note."""
    (tmp_path / "Demo Model.md").write_text(NOTE)
    return TempNoteLibrary(tmp_path)


def test_a_missing_note_file_yields_nothing(temp_notes):
    """A declared note whose file is absent and an unknown key both resolve to None."""
    assert temp_notes.note("other")  is None
    assert temp_notes.note("absent") is None


def test_an_operational_heading_takes_its_whole_subtree_with_it(temp_notes):
    """Stripping an operational section removes its nested headings and body while keeping later sections."""
    markdown = temp_notes.note("demo")["markdown"]

    assert "## Correction"     not in markdown
    assert "The stride"        not in markdown
    assert "Still inside"      not in markdown
    assert "Dropped as well"   not in markdown
    assert "## Architecture"       in markdown


def test_dated_parentheticals_and_flagged_sentences_are_removed(temp_notes):
    """Dated parentheticals and audit-flagged sentences are dropped while the surrounding prose survives."""
    markdown = temp_notes.note("demo")["markdown"]

    assert "(reviewed 2026-07-03)"  not in markdown
    assert "flagged in the audit"   not in markdown
    assert "Four levels of downsampling." in markdown
    assert "The receptive field is 41 pixels." in markdown


def test_stripping_never_leaves_a_run_of_blank_lines(temp_notes):
    """The stripped markdown never contains consecutive blank lines."""
    assert "\n\n\n" not in temp_notes.note("demo")["markdown"]
