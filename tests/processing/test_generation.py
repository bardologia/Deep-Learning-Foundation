from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from configuration.generation               import GenerationConfig
from pipelines.processing.generation        import DatasetGenerator, TargetFunction
from pipelines.shared.dataset.dataset_files import DatasetFiles
from tools.data.regions                     import CropRegion

from tests.conftest import SilentLogger


def test_the_generator_writes_the_three_files_of_the_contract(test_data_dir, scene_config):
    files = DatasetFiles.read(test_data_dir)

    assert sorted(path.name for path in test_data_dir.iterdir() if path.is_file()) == ["dataset.json", "inputs.npy", "targets.npy"]
    assert files.input_channels  == ["x0", "x1", "x2", "x3"]
    assert files.target_channels == ["pointwise", "neighbourhood"]
    assert (files.rows, files.cols) == (scene_config.rows, scene_config.cols)
    assert files.inputs().shape  == (4, scene_config.rows, scene_config.cols)
    assert files.targets().shape == (2, scene_config.rows, scene_config.cols)
    assert files.inputs().dtype  == np.float32
    assert files.generator["config"]["seed"] == scene_config.seed


def test_the_splits_are_disjoint_row_bands_covering_the_grid(test_data_dir, scene_config):
    splits = DatasetFiles.read(test_data_dir).splits
    bands  = [region for name in ("train", "val", "test") for region in splits.regions(name)]

    assert bands[0].row_start == 0
    assert bands[-1].row_end  == scene_config.rows
    assert all(left.row_end == right.row_start for left, right in zip(bands, bands[1:]))
    assert all((region.col_start, region.col_end) == (0, scene_config.cols) for region in bands)


def test_the_targets_follow_the_known_function_of_the_inputs(tmp_path):
    config = dataclasses.replace(GenerationConfig(rows=48, cols=40, target_noise=0.0), output_directory=tmp_path / "clean")
    files  = DatasetGenerator(config, SilentLogger()).run()

    inputs  = np.asarray(files.inputs(), dtype=np.float64)
    targets = np.asarray(files.targets(), dtype=np.float64)

    pointwise     = np.tanh(inputs[0] * inputs[1]) + 0.5 * inputs[2]
    neighbourhood = uniform_filter(inputs[3] ** 2, size=config.neighbourhood, mode="reflect") - uniform_filter(inputs[0], size=config.neighbourhood, mode="reflect")

    assert TargetFunction.NAMES == files.target_channels
    assert np.allclose(targets[0], pointwise, atol=1e-5)
    assert np.allclose(targets[1], neighbourhood, atol=1e-5)


def test_the_same_seed_rebuilds_the_same_arrays(tmp_path):
    first  = DatasetGenerator(dataclasses.replace(GenerationConfig(rows=32, cols=32), output_directory=tmp_path / "a"), SilentLogger()).run()
    second = DatasetGenerator(dataclasses.replace(GenerationConfig(rows=32, cols=32), output_directory=tmp_path / "b"), SilentLogger()).run()

    assert np.array_equal(first.inputs(), second.inputs())
    assert np.array_equal(first.targets(), second.targets())


def test_the_generator_refuses_an_existing_directory(test_data_dir):
    with pytest.raises(FileExistsError):
        DatasetGenerator(dataclasses.replace(GenerationConfig(rows=32, cols=32), output_directory=test_data_dir), SilentLogger()).run()


def test_the_reader_rejects_a_layout_without_a_required_key(tmp_path):
    (tmp_path / "dataset.json").write_text(json.dumps({"name": "broken", "input_channels": ["x0"]}))

    with pytest.raises(KeyError):
        DatasetFiles.read(tmp_path)


def test_crop_cuts_a_region_as_contiguous_float32():
    array  = np.arange(2 * 6 * 5, dtype=np.float64).reshape(2, 6, 5)
    region = CropRegion(1, 4, 2, 5)

    crop = DatasetFiles.crop(array, region)

    assert crop.shape == (2, 3, 3)
    assert crop.dtype == np.float32
    assert crop.flags["C_CONTIGUOUS"]
    assert np.array_equal(crop, array[:, 1:4, 2:5])
