from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configuration.dataset                  import AugmentationConfig, DatasetConfig, PatchConfig
from configuration.normalization            import ChannelStats, NormMethod
from pipelines.backbone.dataset.datasets    import MultiRegionDataset, PatchDataset
from pipelines.backbone.dataset.normalizer  import Normalizer
from pipelines.backbone.dataset.pipeline    import DatasetPipeline
from pipelines.backbone.dataset.spatial     import Patcher
from pipelines.backbone.dataset.stats       import Stats
from pipelines.shared.dataset.dataset_files import DatasetFiles

from tests.conftest import SilentLogger


def _identity(n: int) -> ChannelStats:
    return ChannelStats(names=[f"c{index}" for index in range(n)], method=NormMethod.ZSCORE, loc=[0.0] * n, scale=[1.0] * n)


def _config(test_data_dir, augmentation: AugmentationConfig | None = None) -> DatasetConfig:
    return DatasetConfig(
        dataset_directory = test_data_dir,
        split_regions     = DatasetFiles.read(test_data_dir).splits,
        patch             = PatchConfig(size=(16, 16), stride=(8, 8)),
        augmentation      = augmentation if augmentation is not None else AugmentationConfig(p_flip_h=0.0, p_flip_v=0.0),
        batch_size        = 4,
        num_workers       = 0,
    )


def test_patches_carry_inputs_targets_and_their_pixel_weights():
    inputs  = np.random.default_rng(0).normal(size=(3, 20, 24)).astype(np.float32)
    targets = np.random.default_rng(1).normal(size=(2, 20, 24)).astype(np.float32)
    patcher = Patcher.build(spatial_size=(20, 24), patch_size=(16, 16), stride=(8, 8))
    dataset = PatchDataset(inputs, targets, patcher, Normalizer(Stats(_identity(3), _identity(2))), count_once=True)

    x, y, w = dataset[0]

    assert (x.shape, y.shape, w.shape) == ((3, 16, 16), (2, 16, 16), (1, 16, 16))
    assert dataset.input_channels == 3 and dataset.target_channels == 2
    assert float(sum(dataset[index][2].sum() for index in range(len(dataset)))) == pytest.approx(20 * 24)


def test_train_patches_weigh_every_covering_patch_and_padding_nothing():
    patcher = Patcher.build(spatial_size=(20, 24), patch_size=(16, 16), stride=(8, 8))
    grid    = patcher.grid

    train = patcher.weight_map(count_once=False)
    once  = patcher.weight_map(count_once=True)

    assert train.shape == grid.padded_size
    assert train.sum() == 20 * 24
    assert float((once[grid.pad_top:grid.pad_top + 20, grid.pad_left:grid.pad_left + 24] * grid.coverage()).sum()) == pytest.approx(20 * 24)


def test_multi_region_dataset_concatenates_its_parts():
    normalizer = Normalizer(Stats(_identity(1), _identity(1)))
    parts      = [PatchDataset(np.full((1, 16, 16), float(index), np.float32), np.zeros((1, 16, 16), np.float32), Patcher.build((16, 16), (16, 16), (16, 16)), normalizer, count_once=False) for index in range(3)]
    dataset    = MultiRegionDataset(parts)

    assert len(dataset) == 3
    assert [float(dataset[index][0][0, 0, 0]) for index in range(3)] == [0.0, 1.0, 2.0]
    assert float(dataset[-1][0][0, 0, 0]) == 2.0


def test_the_pipeline_builds_all_splits_and_writes_its_metadata(test_data_dir, tmp_path):
    train_loader, val_loader, test_loader, datasets, normalizer = DatasetPipeline(_config(test_data_dir), tmp_path, SilentLogger(), seed=0).run()

    inputs, targets, weights = next(iter(train_loader))
    meta                     = tmp_path / "meta"

    assert set(datasets) == {"train", "val", "test"}
    assert inputs.shape  == (4, 4, 16, 16)
    assert targets.shape == (4, 2, 16, 16)
    assert weights.shape == (4, 1, 16, 16)
    assert abs(float(inputs.mean())) < 1.0
    assert json.loads((meta / "dataset_creation_config.json").read_text())["dataset_directory"] == str(test_data_dir)
    assert set(json.loads((meta / "patch.json").read_text())) == {"train", "val", "test"}
    assert Stats.load(meta, SilentLogger()) == normalizer.stats


def test_the_train_order_is_reproducible_for_a_seed(test_data_dir, tmp_path):
    def first_batch(run_dir):
        loader = DatasetPipeline(_config(test_data_dir, augmentation=AugmentationConfig()), run_dir, SilentLogger(), seed=3).run()[0]
        return next(iter(loader))[0]

    assert torch.equal(first_batch(tmp_path / "a"), first_batch(tmp_path / "b"))


def test_the_dataset_config_round_trips_its_payload(test_data_dir):
    config = _config(test_data_dir)

    assert DatasetConfig.from_dict(json.loads(json.dumps(config.as_dict()))) == config
