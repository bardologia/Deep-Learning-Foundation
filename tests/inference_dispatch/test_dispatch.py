"""Tests that RunClassifier identifies a run's type from the config files in its meta directory.

Covers the backbone, profile-autoencoder, image-autoencoder and dual run types, the routing of
JEPA runs to the backbone type, the refusal to classify an empty run, and agreement between
RunArtifacts filenames and the config IO classes. Also covers the per-family runners, the
per-region scoring of a split stored as several regions and the single pass of a run scored
on a target dataset."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from pipelines.shared.config.config_persistence import BackboneModelConfigIO, DualModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO
from pipelines.shared.inference.metadata        import RegionPass
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


class _RecordingPipeline:
    """Inference pipeline stand-in that records every configuration it is built with, and the region pass of each pass.

    Attributes:
        passes        : Configuration and component set of every pass, in construction order.
        region_passes : Region pass handed to every pass, in construction order.
    """

    passes        = []
    region_passes = []

    def __init__(self, config, components=None, region_pass=None) -> None:
        """Records the configuration, component set and region pass of one pass."""
        _RecordingPipeline.passes.append((config, components))
        _RecordingPipeline.region_passes.append(region_pass)

    def run(self) -> None:
        """Does nothing; the pass is already recorded."""
        pass


def _write_split_regions(run: Path, test_regions) -> Path:
    """Writes a dataset creation config whose test split holds the given region dict or list of them."""
    (run / "meta").mkdir(parents=True)

    split_regions = {
        "train" : {"row_start": 0,   "row_end": 100, "col_start": 0, "col_end": 50},
        "val"   : {"row_start": 100, "row_end": 150, "col_start": 0, "col_end": 50},
        "test"  : test_regions,
    }
    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({"split_regions": split_regions}))

    return run


@pytest.mark.parametrize("runner_name, entry_name, pipeline_module, pipeline_name", [
    ("ProfileAeInferenceRunner", "ProfileAeInferenceEntryConfig", "pipelines.profile_autoencoder.inference.pipeline", "ProfileAeInferencePipeline"),
    ("ImageAeInferenceRunner",   "ImageAeInferenceEntryConfig",   "pipelines.image_autoencoder.inference.pipeline",   "ImageAeInferencePipeline"),
])
def test_autoencoder_runners_drive_their_pipeline_once(tmp_path, monkeypatch, runner_name, entry_name, pipeline_module, pipeline_name):
    """The profile-AE and image-AE runners build their pipeline once on the run, with no region handling."""
    import importlib

    import configuration.inference                       as inference_configs
    import pipelines.shared.inference.inference_dispatch as dispatch

    monkeypatch.setattr(importlib.import_module(pipeline_module), pipeline_name, _RecordingPipeline)
    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    getattr(dispatch, runner_name)(getattr(inference_configs, entry_name)()).run(tmp_path)

    assert len(_RecordingPipeline.passes) == 1

    config, components = _RecordingPipeline.passes[0]

    assert config.run_directory == tmp_path
    assert config.output_subdir is None
    assert components           is None


def test_dual_runner_scores_each_stored_region_with_the_dual_components(tmp_path, monkeypatch):
    """The dual runner goes through the split-region loop: one pass per disjoint test region, each handed the dual component set."""
    import configuration.inference                       as inference_configs
    import pipelines.backbone.inference.pipeline         as backbone_pipeline
    import pipelines.shared.inference.inference_dispatch as dispatch
    from pipelines.dual.inference.pipeline               import DUAL_INFERENCE_COMPONENTS

    monkeypatch.setattr(backbone_pipeline, "InferencePipeline", _RecordingPipeline)
    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = _write_split_regions(tmp_path / "run", [
        {"row_start": 150, "row_end": 200, "col_start": 0, "col_end": 50},
        {"row_start": 250, "row_end": 300, "col_start": 0, "col_end": 50},
    ])

    dispatch.DualInferenceRunner(inference_configs.DualInferenceEntryConfig()).run(run)

    overrides  = [config.region_override for config, _ in _RecordingPipeline.passes]
    components = [components for _, components in _RecordingPipeline.passes]

    assert overrides == [[150, 200, 0, 50], [250, 300, 0, 50]]
    assert all(config.run_directory == run for config, _ in _RecordingPipeline.passes)
    assert components == [DUAL_INFERENCE_COMPONENTS] * 2
    assert [region_pass.index for region_pass in _RecordingPipeline.region_passes] == [0, 1]


def test_dual_runner_scores_an_enabled_target_in_one_pass_without_the_stored_regions(tmp_path, monkeypatch):
    """With a target dataset enabled the dual runner skips the split-region loop: one pass, no region override, no region pass."""
    import configuration.inference                       as inference_configs
    import pipelines.backbone.inference.pipeline         as backbone_pipeline
    import pipelines.shared.inference.inference_dispatch as dispatch
    from pipelines.dual.inference.pipeline               import DUAL_INFERENCE_COMPONENTS

    monkeypatch.setattr(backbone_pipeline, "InferencePipeline", _RecordingPipeline)
    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = _write_split_regions(tmp_path / "run", [
        {"row_start": 150, "row_end": 200, "col_start": 0, "col_end": 50},
        {"row_start": 250, "row_end": 300, "col_start": 0, "col_end": 50},
    ])

    entry                          = inference_configs.DualInferenceEntryConfig()
    entry.inference.target.enabled = True

    dispatch.DualInferenceRunner(entry).run(run)

    assert len(_RecordingPipeline.passes) == 1

    config, components = _RecordingPipeline.passes[0]

    assert config.run_directory             == run
    assert config.output_subdir             is None
    assert config.region_override           is None
    assert components                       == DUAL_INFERENCE_COMPONENTS
    assert _RecordingPipeline.region_passes == [None]


def test_split_region_loop_scores_each_disjoint_region_in_its_own_pass(tmp_path, monkeypatch):
    """A test split stored as two disjoint regions is scored twice, each region as override under its own indexed stamp."""
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = _write_split_regions(tmp_path / "run", [
        {"row_start": 150, "row_end": 200, "col_start": 0, "col_end": 50},
        {"row_start": 250, "row_end": 300, "col_start": 0, "col_end": 50},
    ])

    SplitRegionLoop(InferenceConfig(run_directory=run, split="test"), _RecordingPipeline, "components").run()

    overrides = [config.region_override for config, _ in _RecordingPipeline.passes]
    stamps    = [config.output_subdir   for config, _ in _RecordingPipeline.passes]
    group     = stamps[0][:-len("_region0")]

    assert overrides == [[150, 200, 0, 50], [250, 300, 0, 50]]
    assert stamps[0].endswith("_region0") and stamps[1].endswith("_region1")
    assert stamps[1][:-len("_region1")] == group
    assert all(components == "components" for _, components in _RecordingPipeline.passes)
    assert _RecordingPipeline.region_passes == [RegionPass(index=0, count=2, group=group), RegionPass(index=1, count=2, group=group)]


def test_split_region_loop_scores_a_single_region_split_once(tmp_path, monkeypatch):
    """A split stored as one region is scored in a single pass with the configuration unchanged."""
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run    = _write_split_regions(tmp_path / "run", {"row_start": 150, "row_end": 200, "col_start": 0, "col_end": 50})
    config = InferenceConfig(run_directory=run, split="test")

    SplitRegionLoop(config, _RecordingPipeline, "components").run()

    assert _RecordingPipeline.passes        == [(config, "components")]
    assert _RecordingPipeline.region_passes == [None]


def test_split_region_loop_keeps_an_explicit_region_override(tmp_path, monkeypatch):
    """A configured region override is scored in a single pass without reading the stored regions."""
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    config = InferenceConfig(run_directory=tmp_path / "no_meta", split="test", region_override=[150, 200, 0, 50])

    SplitRegionLoop(config, _RecordingPipeline, "components").run()

    assert _RecordingPipeline.passes == [(config, "components")]


def test_split_region_loop_raises_when_the_run_stores_no_split_regions(tmp_path, monkeypatch):
    """A dataset creation config without split regions raises instead of scoring the run as one region."""
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = tmp_path / "run"
    (run / "meta").mkdir(parents=True)
    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({}))

    with pytest.raises(KeyError, match="split_regions"):
        SplitRegionLoop(InferenceConfig(run_directory=run, split="test"), _RecordingPipeline, "components").run()

    assert _RecordingPipeline.passes == []
