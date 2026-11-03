"""Tests the target dataset a trained run is scored on instead of its own split.

Covers the checks applied when a target is enabled (split, region override,
dataset layout, label cube placement, crop bounds), the zero bounds that span
the target's whole global crop along an axis, the dataset configuration moved
onto the target scene, the marker payload, the stamp tree the target inference
writes to, and the scheduler rule that only a stamp recording the same target
counts as scored.
"""

from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from configuration.dataset                           import DatasetConfig, InputConfig, PatchConfig, SplitRegions
from configuration.inference                         import InferenceConfig, TargetDatasetConfig
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.shared.inference.inference_scheduler  import InferenceScheduler
from pipelines.shared.inference.metadata             import CompletedInference, TargetDataset
from pipelines.shared.inference.run_classifier       import InferenceLayout, RunArtifacts, RunType
from tools.data.io                                   import FileIO
from tools.data.regions                              import CropRegion
from tools.runtime.completion                        import CompletionMarker

from tests.conftest import SilentLogger


GLOBAL_CROP = [1000, 2000, 500, 1000]
CHANNELS    = ["CH05", "CH06"]


def _target_dataset(root: Path, name: str = "scene_a") -> tuple[Path, Path]:
    """Creates a dataset directory holding a layout manifest and one parameter run, returning the dataset and its label cube."""
    dataset    = root / name
    parameters = dataset / "params" / "params_k2" / "parameters.npy"

    FileIO.save_json({"global_crop": GLOBAL_CROP}, dataset / "data" / "dataset.json")
    parameters.parent.mkdir(parents=True)
    np.save(parameters, np.zeros((6, 4, 4), dtype=np.float32))

    return dataset, parameters


def _config(dataset: Path, parameters: Path, crop=(1200, 1296, 600, 696), channels=CHANNELS, **overrides) -> InferenceConfig:
    """Returns an inference configuration with the target enabled over the given dataset, labels, crop and secondary channels."""
    target = TargetDatasetConfig(
        enabled          = True,
        dataset_path     = dataset,
        parameters_path  = parameters,
        secondary_labels = list(channels),
        row_start        = crop[0],
        row_end          = crop[1],
        col_start        = crop[2],
        col_end          = crop[3],
    )
    return InferenceConfig(run_directory=Path("."), target=target, **overrides)


def test_a_disabled_target_scores_the_run_on_its_own_split():
    """With the target off, no target is resolved, so the run is scored on its own split."""
    assert TargetDataset.of(InferenceConfig(run_directory=Path("."))) is None


def test_an_enabled_target_resolves_its_scene_labels_channels_and_crop(tmp_path):
    """The resolved target names the dataset directory, keeps the channel order and builds the crop."""
    dataset, parameters = _target_dataset(tmp_path)

    target = TargetDataset.of(_config(dataset, parameters))

    assert target.name             == "scene_a"
    assert target.parameters_path  == parameters
    assert target.secondary_labels == tuple(CHANNELS)
    assert target.region           == CropRegion(1200, 1296, 600, 696)


def test_empty_target_channels_feed_the_full_target_stack(tmp_path):
    """An empty channel list resolves to None, which the cropper reads as every secondary channel."""
    dataset, parameters = _target_dataset(tmp_path)

    assert TargetDataset.of(_config(dataset, parameters, channels=[])).secondary_labels is None


@pytest.mark.parametrize("overrides, message", [
    ({"split": "val"},                            "leave inference.split at 'test'"),
    ({"region_override": [1200, 1296, 600, 696]}, "leave region_override unset"),
])
def test_a_target_refuses_settings_of_the_runs_own_split(tmp_path, overrides, message):
    """A target cannot be combined with another split or a region override, which only describe the training scene."""
    dataset, parameters = _target_dataset(tmp_path)

    with pytest.raises(ValueError, match=message):
        TargetDataset.of(_config(dataset, parameters, **overrides))


def test_a_target_without_a_dataset_layout_is_refused(tmp_path):
    """A directory without data/dataset.json is not a dataset."""
    with pytest.raises(FileNotFoundError, match="dataset.json"):
        TargetDataset.of(_config(tmp_path / "missing", tmp_path / "missing" / "params" / "p" / "parameters.npy"))


def test_target_labels_outside_the_target_dataset_are_refused(tmp_path):
    """Labels must come from a parameter run of the target scene itself."""
    dataset, _      = _target_dataset(tmp_path, "scene_a")
    _, other_labels = _target_dataset(tmp_path, "scene_b")

    with pytest.raises(ValueError, match="must live inside"):
        TargetDataset.of(_config(dataset, other_labels))


def test_a_missing_target_label_cube_is_refused(tmp_path):
    """A label path inside the dataset that holds no file is refused before any loading."""
    dataset, _ = _target_dataset(tmp_path)

    with pytest.raises(FileNotFoundError, match="parameters_path"):
        TargetDataset.of(_config(dataset, dataset / "params" / "absent" / "parameters.npy"))


@pytest.mark.parametrize("crop, expected", [
    ((0, 0, 0, 0),       (1000, 2000, 500, 1000)),
    ((1200, 1296, 0, 0), (1200, 1296, 500, 1000)),
    ((0, 0, 600, 696),   (1000, 2000, 600, 696)),
])
def test_zero_bounds_span_the_whole_target_crop_along_their_axis(tmp_path, crop, expected):
    """An axis left at 0 to 0 takes the target's global extent, so all four at 0 scores the full scene."""
    dataset, parameters = _target_dataset(tmp_path)

    assert TargetDataset.of(_config(dataset, parameters, crop=crop)).region == CropRegion(*expected)


@pytest.mark.parametrize("crop, message", [
    ((1300, 1200, 600, 696), "empty or inverted"),
    ((0, 1296, 600, 696),    "not contained in the global crop"),
    ((1200, 1296, 600, 0),   "empty or inverted"),
    ((1900, 2100, 600, 696), "not contained in the global crop"),
])
def test_an_invalid_target_crop_is_refused(tmp_path, crop, message):
    """The crop must be non-empty and lie inside the target's global crop."""
    dataset, parameters = _target_dataset(tmp_path)

    with pytest.raises(ValueError, match=message):
        TargetDataset.of(_config(dataset, parameters, crop=crop))


def test_the_dataset_config_moves_onto_the_target_and_keeps_the_runs_geometry(tmp_path):
    """Scene, labels, secondary channels and regions change; patch geometry, inputs and batch settings stay the run's."""
    dataset, parameters = _target_dataset(tmp_path)
    target              = TargetDataset.of(_config(dataset, parameters))

    trained = DatasetConfig(
        preprocessing_run_directory = tmp_path / "training_scene",
        parameters_path             = tmp_path / "training_scene" / "params" / "p" / "parameters.npy",
        split_regions               = SplitRegions(train=CropRegion(0, 10, 0, 10), val=CropRegion(10, 20, 0, 10), test=CropRegion(20, 30, 0, 10)),
        secondary_labels            = ("CH01", "CH02"),
        patch                       = PatchConfig(size=(64, 64), stride=(32, 32), use_symmetric_padding=True),
        input_config                = InputConfig(),
        batch_size                  = 4,
        n_gaussians                 = 2,
    )

    moved = target.dataset_config(trained)

    assert moved.preprocessing_run_directory    == dataset
    assert moved.parameters_path                == parameters
    assert moved.secondary_labels               == tuple(CHANNELS)
    assert moved.split_regions.regions("test")  == [CropRegion(1200, 1296, 600, 696)]
    assert moved.split_regions.regions("train") == []
    assert moved.split_regions.regions("val")   == []
    assert moved.patch                          == trained.patch
    assert moved.input_config                   == trained.input_config
    assert moved.batch_size                     == trained.batch_size
    assert moved.n_gaussians                    == trained.n_gaussians


def test_the_payload_names_the_scene_labels_channels_and_crop(tmp_path):
    """The marker entry carries every field that decides what was scored."""
    dataset, parameters = _target_dataset(tmp_path)

    payload = TargetDataset.of(_config(dataset, parameters)).payload()

    assert payload == {"dataset_path": str(dataset), "parameters_path": str(parameters), "secondary_labels": CHANNELS, "region": [1200, 1296, 600, 696]}


def test_target_stamps_are_written_under_the_runs_cross_dataset_tree(tmp_path):
    """A target pass stamps under <run>/cross_dataset/<dataset>, never under the run's inference directory."""
    dataset, parameters  = _target_dataset(tmp_path)
    run_dir              = tmp_path / "run"
    config               = _config(dataset, parameters, output_subdir="stamp")
    config.run_directory = run_dir

    meta = InferenceMetadata(config, TargetDataset.of(config))

    assert meta.output_dir == run_dir / InferenceLayout.TARGET_DIRNAME / "scene_a" / "stamp"
    assert InferenceMetadata(config).output_dir == run_dir / InferenceLayout.DIRNAME / "stamp"


def test_a_nested_family_keeps_its_subdirectory_under_the_target_tree(tmp_path):
    """Families that nest their stamps nest them the same way under a target."""
    assert InferenceLayout.target_base(tmp_path, RunType.PROFILE_AE, "scene_a") == tmp_path / "cross_dataset" / "scene_a" / "profile_ae"
    assert InferenceLayout.target_base(tmp_path, RunType.BACKBONE, "scene_a")   == tmp_path / "cross_dataset" / "scene_a"


def test_a_marker_without_a_target_is_refused_under_the_target_tree(tmp_path):
    """Every stamp of a cross-dataset tree must record its target."""
    stamp = tmp_path / "stamp"
    stamp.mkdir()
    CompletionMarker.stamp(stamp, {"split": "test"})

    with pytest.raises(KeyError, match="records no 'target_dataset'"):
        CompletedInference.target(stamp)


def _run(directory: Path) -> Path:
    """Creates a backbone run directory."""
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / RunArtifacts.BACKBONE_CONFIG).write_text("{}")
    return directory


def _target_stamp(run_dir: Path, target: TargetDataset, name: str, payload: dict | None = None) -> None:
    """Writes a completed target stamp under the run, recording the given target payload."""
    stamp = InferenceLayout.target_base(run_dir, RunType.BACKBONE, target.name) / name
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"split": "test", CompletedInference.CHECKPOINT_KEY: {"name": "best_model.pt"}, TargetDataset.KEY: payload if payload is not None else target.payload()})


def _scheduler(runs: Path, block: InferenceConfig) -> InferenceScheduler:
    """Builds a backbone scheduler over a runs directory with the given inference block."""
    config = SimpleNamespace(runs_dir=str(runs), run_filter=[], gpus=[0], gpus_file="", poll_interval_s=1, skip_completed=True)
    return InferenceScheduler(config, Path("main/x.py"), RunType.BACKBONE, block=block)


def test_the_scheduler_skips_only_runs_scored_on_the_same_target(tmp_path):
    """Own-split stamps and stamps on another crop of the target do not count; a stamp on the same target does."""
    dataset, parameters = _target_dataset(tmp_path)
    block               = _config(dataset, parameters)
    target              = TargetDataset.of(block)
    runs                = tmp_path / "runs"

    own_only   = _run(runs / "own_only")
    other_crop = _run(runs / "other_crop")
    scored     = _run(runs / "scored")

    own_stamp = InferenceLayout.base(own_only, RunType.BACKBONE) / "20260101_000000"
    own_stamp.mkdir(parents=True)
    CompletionMarker.stamp(own_stamp, {"split": "test", CompletedInference.CHECKPOINT_KEY: {"name": "best_model.pt"}})

    _target_stamp(other_crop, target, "20260101_000000", payload={**target.payload(), "region": [1300, 1396, 600, 696]})
    _target_stamp(scored, target, "20260101_000000")

    scheduler = _scheduler(runs, block)
    remaining = sorted(directory.name for directory in scheduler._run_dirs(SilentLogger()))

    assert remaining                                           == ["other_crop", "own_only"]
    assert [directory.name for directory in scheduler.skipped] == ["scored"]


def test_the_scheduler_refuses_an_invalid_target_before_dispatching(tmp_path):
    """An invalid target block fails when the scheduler is built, before any worker starts."""
    dataset, parameters = _target_dataset(tmp_path)

    with pytest.raises(ValueError, match="empty or inverted"):
        _scheduler(tmp_path, _config(dataset, parameters, crop=(1300, 1200, 600, 696)))
