from __future__ import annotations

import pytest

from pipelines.shared.config.config_persistence import BackboneModelConfigIO, DualModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO, UnrolledModelConfigIO
from pipelines.shared.inference.run_classifier  import RunArtifacts, RunClassifier, RunType


def _make_run(tmp_path, *filenames):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (meta / name).write_text("{}")
    return tmp_path


def test_classify_backbone(tmp_path):
    run = _make_run(tmp_path, "model_config.json")
    assert RunClassifier.classify(run) == RunType.BACKBONE


def test_classify_profile_autoencoder(tmp_path):
    run = _make_run(tmp_path, "profile_autoencoder_config.json")
    assert RunClassifier.classify(run) == RunType.PROFILE_AE


def test_classify_image_autoencoder(tmp_path):
    run = _make_run(tmp_path, "image_autoencoder_config.json")
    assert RunClassifier.classify(run) == RunType.IMAGE_AE


def test_classify_unrolled(tmp_path):
    run = _make_run(tmp_path, "unrolled_model_config.json")
    assert RunClassifier.classify(run) == RunType.UNROLLED


def test_classify_dual(tmp_path):
    run = _make_run(tmp_path, "dual_model_config.json")
    assert RunClassifier.classify(run) == RunType.DUAL


def test_classify_jepa_run_routes_to_backbone(tmp_path):
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json")
    assert RunClassifier.classify(run) == RunType.BACKBONE


def test_classify_unrecognized_run_raises(tmp_path):
    run = _make_run(tmp_path)
    with pytest.raises(ValueError):
        RunClassifier.classify(run)


def test_is_type_matches_only_its_type(tmp_path):
    run = _make_run(tmp_path, "profile_autoencoder_config.json")
    assert RunClassifier.is_type(run, RunType.PROFILE_AE) is True
    assert RunClassifier.is_type(run, RunType.BACKBONE)   is False


def test_is_type_returns_false_for_unrecognized_run(tmp_path):
    run = _make_run(tmp_path)
    assert RunClassifier.is_type(run, RunType.BACKBONE) is False


def test_run_classifier_artifacts_match_config_io_filenames():
    assert BackboneModelConfigIO.FILENAME      == RunArtifacts.BACKBONE_CONFIG
    assert ProfileAutoencoderConfigIO.FILENAME == RunArtifacts.PROFILE_AE_CONFIG
    assert ImageAutoencoderConfigIO.FILENAME   == RunArtifacts.IMAGE_AE_CONFIG
    assert UnrolledModelConfigIO.FILENAME      == RunArtifacts.UNROLLED_CONFIG
    assert DualModelConfigIO.FILENAME          == RunArtifacts.DUAL_CONFIG
