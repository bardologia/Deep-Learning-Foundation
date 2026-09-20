"""Tests that RunClassifier identifies a run's type from the config files in its meta directory.

Covers the backbone, profile-autoencoder, image-autoencoder, unrolled and dual run types, the
routing of JEPA runs to the backbone type, the refusal to classify an empty run, the is_type
predicate, and agreement between RunArtifacts filenames and the config IO classes."""

from __future__ import annotations

from pathlib import Path

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


def test_spatial_run_config_scores_a_fold_unit_over_its_recorded_block(tmp_path):
    """A fold unit's manifest supplies the region override of the split it records."""
    from configuration.inference                       import InferenceConfig
    from pipelines.cross_validation.folds              import FoldUnitManifest
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "fold_unit"
    FoldUnitManifest.write(run, run_tag="cv", unit="fold_unit", fold_index=2, seed=None, training_type="backbone", inference_regions={"test": (1212, 1312, 0, 256), "val": (1312, 1412, 0, 256)})

    config = SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="val"), run)

    assert config.run_directory   == run
    assert config.region_override == [1312, 1412, 0, 256]
    assert config.output_subdir   is None


def test_spatial_run_config_keeps_an_explicit_region_override(tmp_path):
    """An explicit region override on the entry configuration wins over the fold's recorded block."""
    from configuration.inference                       import InferenceConfig
    from pipelines.cross_validation.folds              import FoldUnitManifest
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "fold_unit"
    FoldUnitManifest.write(run, run_tag="cv", unit="fold_unit", fold_index=2, seed=None, training_type="backbone", inference_regions={"test": (1212, 1312, 0, 256)})

    config = SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="test", region_override=[0, 10, 0, 10]), run)

    assert config.region_override == [0, 10, 0, 10]


def test_spatial_run_config_refuses_a_split_the_fold_does_not_score(tmp_path):
    """Asking a fold unit for a split outside its recorded blocks raises instead of scoring the whole scene."""
    from configuration.inference                       import InferenceConfig
    from pipelines.cross_validation.folds              import FoldUnitManifest
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "fold_unit"
    FoldUnitManifest.write(run, run_tag="cv", unit="fold_unit", fold_index=2, seed=None, training_type="backbone", inference_regions={"test": (1212, 1312, 0, 256)})

    with pytest.raises(ValueError, match="scores the splits"):
        SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="train"), run)


def test_spatial_run_config_leaves_a_plain_run_alone(tmp_path):
    """A run without a manifest gets no region override."""
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "plain"
    (run / "meta").mkdir(parents=True)

    config = SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="test"), run)

    assert config.region_override is None


def test_inference_layout_places_nested_families_under_their_subdirectory(tmp_path):
    """The spatial families stamp directly under inference/ and the others under their family subdirectory."""
    from pipelines.shared.inference.run_classifier import InferenceLayout

    assert InferenceLayout.base(tmp_path, RunType.BACKBONE)   == tmp_path / "inference"
    assert InferenceLayout.base(tmp_path, RunType.DUAL)       == tmp_path / "inference"
    assert InferenceLayout.base(tmp_path, RunType.PROFILE_AE) == tmp_path / "inference" / "profile_ae"
    assert InferenceLayout.base(tmp_path, RunType.IMAGE_AE)   == tmp_path / "inference" / "image_ae"
