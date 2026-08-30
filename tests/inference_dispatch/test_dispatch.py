"""Tests that RunClassifier identifies a run's type from the config files in its meta directory.

Covers the backbone, profile-autoencoder, image-autoencoder, unrolled and dual run types, the
routing of JEPA runs to the backbone type, the refusal to classify an empty run, the is_type
predicate, and agreement between RunArtifacts filenames and the config IO classes."""

from __future__ import annotations

import pytest

from pipelines.shared.config.config_persistence import BackboneModelConfigIO, DualModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO
from pipelines.shared.inference.run_classifier  import RunArtifacts, RunClassifier, RunType


def _make_run(tmp_path, *filenames):
    """Creates a run directory whose meta/ holds the named placeholder config files."""
    meta = tmp_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (meta / name).write_text("{}")
    return tmp_path


def test_classify_backbone(tmp_path):
    """A run holding only a backbone model config classifies as BACKBONE."""
    run = _make_run(tmp_path, "model_config.json")
    assert RunClassifier.classify(run) == RunType.BACKBONE


def test_classify_profile_autoencoder(tmp_path):
    """A run holding only a profile-autoencoder config classifies as PROFILE_AE."""
    run = _make_run(tmp_path, "profile_autoencoder_config.json")
    assert RunClassifier.classify(run) == RunType.PROFILE_AE


def test_classify_image_autoencoder(tmp_path):
    """A run holding only an image-autoencoder config classifies as IMAGE_AE."""
    run = _make_run(tmp_path, "image_autoencoder_config.json")
    assert RunClassifier.classify(run) == RunType.IMAGE_AE


def test_classify_unrolled(tmp_path):
    """A run holding an unrolled model config classifies as UNROLLED."""
    run = _make_run(tmp_path, "unrolled_model_config.json")
    assert RunClassifier.classify(run) == RunType.UNROLLED


def test_classify_dual(tmp_path):
    """A run holding a dual model config classifies as DUAL."""
    run = _make_run(tmp_path, "dual_model_config.json")
    assert RunClassifier.classify(run) == RunType.DUAL


def test_classify_jepa_run_routes_to_backbone(tmp_path):
    """A backbone paired with a profile autoencoder still classifies as BACKBONE."""
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json")
    assert RunClassifier.classify(run) == RunType.BACKBONE


def test_classify_unrecognized_run_raises(tmp_path):
    """A run with no recognised config file raises ValueError."""
    run = _make_run(tmp_path)
    with pytest.raises(ValueError):
        RunClassifier.classify(run)


def test_is_type_matches_only_its_type(tmp_path):
    """is_type answers true only for the run's own type."""
    run = _make_run(tmp_path, "profile_autoencoder_config.json")
    assert RunClassifier.is_type(run, RunType.PROFILE_AE) is True
    assert RunClassifier.is_type(run, RunType.BACKBONE)   is False


def test_is_type_returns_false_for_unrecognized_run(tmp_path):
    """is_type answers false rather than raising for an unrecognised run."""
    run = _make_run(tmp_path)
    assert RunClassifier.is_type(run, RunType.BACKBONE) is False


def test_run_classifier_artifacts_match_config_io_filenames():
    """The RunArtifacts filenames match the FILENAME of every corresponding config IO class."""
    assert BackboneModelConfigIO.FILENAME      == RunArtifacts.BACKBONE_CONFIG
    assert ProfileAutoencoderConfigIO.FILENAME == RunArtifacts.PROFILE_AE_CONFIG
    assert ImageAutoencoderConfigIO.FILENAME   == RunArtifacts.IMAGE_AE_CONFIG
    assert DualModelConfigIO.FILENAME          == RunArtifacts.DUAL_CONFIG
