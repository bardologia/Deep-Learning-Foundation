
from __future__ import annotations

import json

from pathlib import Path

import pytest

from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.inference.metadata        import RegionPass
from pipelines.shared.inference.run_classifier  import RunArtifacts


def _make_run(tmp_path, *filenames):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (meta / name).write_text("{}")
    return tmp_path


def test_a_run_is_a_directory_holding_the_backbone_config(tmp_path):
    assert RunArtifacts.is_run(_make_run(tmp_path / "trained", "model_config.json"))
    assert not RunArtifacts.is_run(_make_run(tmp_path / "empty"))


def test_run_artifacts_match_the_config_io_filename():
    assert BackboneModelConfigIO.FILENAME == RunArtifacts.BACKBONE_CONFIG


def test_spatial_run_config_scores_a_fold_unit_over_its_recorded_block(tmp_path):
    from configuration.inference                       import InferenceConfig
    from pipelines.cross_validation.folds              import FoldUnitManifest
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "fold_unit"
    FoldUnitManifest.write(run, run_tag="cv", unit="fold_unit", fold_index=2, seed=None, inference_regions={"test": (1212, 1312, 0, 256), "val": (1312, 1412, 0, 256)})

    config = SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="val"), run)

    assert config.run_directory   == run
    assert config.region_override == [1312, 1412, 0, 256]
    assert config.output_subdir   is None


def test_spatial_run_config_keeps_an_explicit_region_override(tmp_path):
    from configuration.inference                       import InferenceConfig
    from pipelines.cross_validation.folds              import FoldUnitManifest
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "fold_unit"
    FoldUnitManifest.write(run, run_tag="cv", unit="fold_unit", fold_index=2, seed=None, inference_regions={"test": (1212, 1312, 0, 256)})

    config = SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="test", region_override=[0, 10, 0, 10]), run)

    assert config.region_override == [0, 10, 0, 10]


def test_spatial_run_config_refuses_a_split_the_fold_does_not_score(tmp_path):
    from configuration.inference                       import InferenceConfig
    from pipelines.cross_validation.folds              import FoldUnitManifest
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "fold_unit"
    FoldUnitManifest.write(run, run_tag="cv", unit="fold_unit", fold_index=2, seed=None, inference_regions={"test": (1212, 1312, 0, 256)})

    with pytest.raises(ValueError, match="scores the splits"):
        SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="train"), run)


def test_spatial_run_config_leaves_a_plain_run_alone(tmp_path):
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SpatialRunConfig

    run = tmp_path / "plain"
    (run / "meta").mkdir(parents=True)

    config = SpatialRunConfig.build(InferenceConfig(run_directory=Path("."), split="test"), run)

    assert config.region_override is None


def test_inference_layout_places_stamps_under_the_run(tmp_path):
    from pipelines.shared.inference.run_classifier import InferenceLayout

    assert InferenceLayout.base(tmp_path)                 == tmp_path / "inference"
    assert InferenceLayout.target_base(tmp_path, "scene") == tmp_path / "cross_dataset" / "scene"


class _RecordingPipeline:

    passes        = []
    region_passes = []

    def __init__(self, config, region_pass=None) -> None:
        _RecordingPipeline.passes.append(config)
        _RecordingPipeline.region_passes.append(region_pass)

    def run(self) -> None:
        pass


def _write_split_regions(run: Path, test_regions: list) -> Path:
    (run / "meta").mkdir(parents=True)

    split_regions = {"train": [[0, 100, 0, 50]], "val": [[100, 150, 0, 50]], "test": test_regions}
    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({"split_regions": split_regions}))

    return run


def test_backbone_runner_scores_each_stored_region_in_its_own_pass(tmp_path, monkeypatch):
    import configuration.inference                       as inference_configs
    import pipelines.backbone.inference.pipeline         as backbone_pipeline
    import pipelines.shared.inference.inference_dispatch as dispatch

    monkeypatch.setattr(backbone_pipeline, "InferencePipeline", _RecordingPipeline)
    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = _write_split_regions(tmp_path / "run", [[150, 200, 0, 50], [250, 300, 0, 50]])

    dispatch.BackboneInferenceRunner(inference_configs.BackboneInferenceEntryConfig()).run(run)

    overrides = [config.region_override for config in _RecordingPipeline.passes]

    assert overrides == [[150, 200, 0, 50], [250, 300, 0, 50]]
    assert all(config.run_directory == run for config in _RecordingPipeline.passes)
    assert [region_pass.index for region_pass in _RecordingPipeline.region_passes] == [0, 1]


def test_backbone_runner_scores_an_enabled_target_in_one_pass_without_the_stored_regions(tmp_path, monkeypatch):
    import configuration.inference                       as inference_configs
    import pipelines.backbone.inference.pipeline         as backbone_pipeline
    import pipelines.shared.inference.inference_dispatch as dispatch

    monkeypatch.setattr(backbone_pipeline, "InferencePipeline", _RecordingPipeline)
    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = _write_split_regions(tmp_path / "run", [[150, 200, 0, 50], [250, 300, 0, 50]])

    entry                          = inference_configs.BackboneInferenceEntryConfig()
    entry.inference.target.enabled = True

    dispatch.BackboneInferenceRunner(entry).run(run)

    assert len(_RecordingPipeline.passes) == 1

    config = _RecordingPipeline.passes[0]

    assert config.run_directory             == run
    assert config.output_subdir             is None
    assert config.region_override           is None
    assert _RecordingPipeline.region_passes == [None]


def test_split_region_loop_scores_each_disjoint_region_in_its_own_pass(tmp_path, monkeypatch):
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = _write_split_regions(tmp_path / "run", [[150, 200, 0, 50], [250, 300, 0, 50]])

    SplitRegionLoop(InferenceConfig(run_directory=run, split="test"), _RecordingPipeline).run()

    overrides = [config.region_override for config in _RecordingPipeline.passes]
    stamps    = [config.output_subdir   for config in _RecordingPipeline.passes]
    group     = stamps[0][:-len("_region0")]

    assert overrides == [[150, 200, 0, 50], [250, 300, 0, 50]]
    assert stamps[0].endswith("_region0") and stamps[1].endswith("_region1")
    assert stamps[1][:-len("_region1")] == group
    assert _RecordingPipeline.region_passes == [RegionPass(index=0, count=2, group=group), RegionPass(index=1, count=2, group=group)]


def test_split_region_loop_scores_a_single_region_split_once(tmp_path, monkeypatch):
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run    = _write_split_regions(tmp_path / "run", [[150, 200, 0, 50]])
    config = InferenceConfig(run_directory=run, split="test")

    SplitRegionLoop(config, _RecordingPipeline).run()

    assert _RecordingPipeline.passes        == [config]
    assert _RecordingPipeline.region_passes == [None]


def test_split_region_loop_keeps_an_explicit_region_override(tmp_path, monkeypatch):
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    config = InferenceConfig(run_directory=tmp_path / "no_meta", split="test", region_override=[150, 200, 0, 50])

    SplitRegionLoop(config, _RecordingPipeline).run()

    assert _RecordingPipeline.passes == [config]


def test_split_region_loop_raises_when_the_run_stores_no_split_regions(tmp_path, monkeypatch):
    from configuration.inference                       import InferenceConfig
    from pipelines.shared.inference.inference_dispatch import SplitRegionLoop

    monkeypatch.setattr(_RecordingPipeline, "passes", [])
    monkeypatch.setattr(_RecordingPipeline, "region_passes", [])

    run = tmp_path / "run"
    (run / "meta").mkdir(parents=True)
    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({}))

    with pytest.raises(KeyError, match="split_regions"):
        SplitRegionLoop(InferenceConfig(run_directory=run, split="test"), _RecordingPipeline).run()

    assert _RecordingPipeline.passes == []
