from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pytest

from configuration.dataset                           import DatasetConfig, PatchConfig
from configuration.inference                         import InferenceConfig, TargetDatasetConfig
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.shared.inference.inference_scheduler  import InferenceScheduler
from pipelines.shared.inference.metadata             import CompletedInference, TargetDataset
from pipelines.shared.inference.run_layout           import InferenceLayout, RunArtifacts
from tools.data.io                                   import FileIO
from tools.data.regions                              import CropRegion, SplitRegions
from tools.runtime.completion                        import CompletionMarker

from tests.conftest import SilentLogger


TARGET_SPLITS = {"train": [[0, 40, 0, 32]], "val": [[40, 48, 0, 32]], "test": [[48, 64, 0, 32]]}


def _target_dataset(root: Path, name: str = "scene_a") -> Path:
    dataset = root / name
    FileIO.save_json({"name": name, "input_channels": ["x0"], "target_channels": ["y"], "rows": 64, "cols": 32, "splits": TARGET_SPLITS, "generator": {}}, dataset / "dataset.json")

    return dataset


def _config(dataset: Path, **overrides) -> InferenceConfig:
    return InferenceConfig(run_directory=Path("."), target=TargetDatasetConfig(enabled=True, dataset_path=dataset), **overrides)


def test_a_disabled_target_scores_the_run_on_its_own_split():
    assert TargetDataset.of(InferenceConfig(run_directory=Path("."))) is None


def test_an_enabled_target_names_its_dataset(tmp_path):
    target = TargetDataset.of(_config(_target_dataset(tmp_path)))

    assert target.name    == "scene_a"
    assert target.payload() == {"dataset_path": str(tmp_path / "scene_a")}
    assert "test split" in target.describe()


def test_the_dataset_config_moves_onto_the_target_and_keeps_the_runs_geometry(tmp_path):
    dataset = _target_dataset(tmp_path)
    trained = DatasetConfig(dataset_directory=tmp_path / "trained", split_regions=SplitRegions(train=CropRegion(0, 8, 0, 8), val=CropRegion(8, 16, 0, 8), test=CropRegion(16, 24, 0, 8)), patch=PatchConfig(size=(16, 16), stride=(8, 8)), batch_size=7)

    moved = TargetDataset.of(_config(dataset)).dataset_config(trained)

    assert moved.dataset_directory      == dataset
    assert moved.split_regions.bounds() == TARGET_SPLITS
    assert moved.patch                  == trained.patch
    assert moved.batch_size             == 7


def test_target_stamps_are_written_under_the_runs_cross_dataset_tree(tmp_path):
    run_dir              = tmp_path / "run"
    config               = _config(_target_dataset(tmp_path), output_subdir="stamp")
    config.run_directory = run_dir

    meta = InferenceMetadata(config, TargetDataset.of(config))

    assert meta.output_dir == run_dir / InferenceLayout.TARGET_DIRNAME / "scene_a" / "stamp"
    assert InferenceMetadata(config).output_dir == run_dir / InferenceLayout.DIRNAME / "stamp"


def test_a_marker_without_a_target_is_refused_under_the_target_tree(tmp_path):
    stamp = tmp_path / "stamp"
    stamp.mkdir()
    CompletionMarker.stamp(stamp, {"split": "test"})

    with pytest.raises(KeyError, match="target_dataset"):
        CompletedInference.target(stamp)


def _run(directory: Path) -> Path:
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / RunArtifacts.BACKBONE_CONFIG).write_text("{}")
    return directory


def _target_stamp(run_dir: Path, target: TargetDataset, name: str, payload: dict | None = None) -> None:
    stamp = InferenceLayout.target_base(run_dir, target.name) / name
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"split": "test", CompletedInference.CHECKPOINT_KEY: {"name": "best_model.pt"}, TargetDataset.KEY: payload if payload is not None else target.payload()})


def _scheduler(runs: Path, block: InferenceConfig) -> InferenceScheduler:
    config = SimpleNamespace(runs_dir=str(runs), run_filter=[], gpus=[0], gpus_file="", poll_interval_s=1, skip_completed=True, inference=block)
    return InferenceScheduler(config, Path("main/x.py"))


def test_the_scheduler_skips_only_runs_scored_on_the_same_target(tmp_path):
    block  = _config(_target_dataset(tmp_path))
    target = TargetDataset.of(block)
    runs   = tmp_path / "runs"

    own_only    = _run(runs / "own_only")
    other_scene = _run(runs / "other_scene")
    scored      = _run(runs / "scored")

    own_stamp = InferenceLayout.base(own_only) / "20260101_000000"
    own_stamp.mkdir(parents=True)
    CompletionMarker.stamp(own_stamp, {"split": "test", CompletedInference.CHECKPOINT_KEY: {"name": "best_model.pt"}})

    _target_stamp(other_scene, target, "20260101_000000", payload={"dataset_path": str(tmp_path / "elsewhere")})
    _target_stamp(scored, target, "20260101_000000")

    scheduler = _scheduler(runs, block)
    remaining = sorted(directory.name for directory in scheduler._run_dirs(SilentLogger()))

    assert remaining                                           == ["other_scene", "own_only"]
    assert [directory.name for directory in scheduler.skipped] == ["scored"]
