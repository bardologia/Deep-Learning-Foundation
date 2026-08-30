"""Tests covering ConfigFactory against real preprocessing data: the global crop, secondary-track labels, input representations, split regions, patching, and the inference and trainer configs it derives."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.benchmark.general        import BenchmarkConfig
from configuration.dataset                  import Representation
from pipelines.shared.config.config_factory import ConfigFactory
from tools.data.regions                     import CropRegion


def _factory(test_data_dir, params_dir, **overrides) -> ConfigFactory:
    """Builds a config factory over the real test dataset with fixed splits and patching.

    Args:
        test_data_dir : Preprocessing run directory.
        params_dir    : Directory holding parameters.npy.
        **overrides   : Attributes set on the benchmark config itself.

    Returns:
        Factory wrapping the assembled benchmark config.
    """
    cfg = BenchmarkConfig()
    cfg.paths.dataset_path     = test_data_dir
    cfg.paths.parameters_path  = params_dir / "parameters.npy"
    cfg.paths.secondary_labels = ("CH01", "CH02", "CH03", "CH04")

    cfg.training.train_rows   = (1000, 1400)
    cfg.training.val_rows     = (1400, 1700)
    cfg.training.test_rows    = (1700, 2000)
    cfg.training.patch_size   = (32, 32)
    cfg.training.patch_stride = (16, 16)
    cfg.training.batch_size   = 8

    for k, v in overrides.items():
        setattr(cfg, k, v)

    return ConfigFactory(cfg)


@pytest.mark.dataset
def test_global_crop_matches_dataset_json(test_data_dir, params_dir, dataset_json):
    """Verifies the derived global crop matches the crop recorded in the dataset metadata."""
    factory = _factory(test_data_dir, params_dir)
    crop    = factory.global_crop()

    assert isinstance(crop, CropRegion)
    assert list(crop.as_tuple()) == dataset_json["global_crop"]


@pytest.mark.dataset
def test_secondary_labels_propagate_as_tuple(test_data_dir, params_dir):
    """Verifies the configured secondary-track labels are passed through unchanged."""
    factory = _factory(test_data_dir, params_dir)

    assert factory._secondary_labels() == ("CH01", "CH02", "CH03", "CH04")


@pytest.mark.dataset
def test_empty_secondary_labels_become_none(test_data_dir, params_dir):
    """Verifies an empty secondary-label tuple is reported as None."""
    factory = _factory(test_data_dir, params_dir)
    factory.config.paths.secondary_labels = ()

    assert factory._secondary_labels() is None


@pytest.mark.dataset
def test_benchmark_input_config_representations(test_data_dir, params_dir):
    """Verifies the benchmark inputs use magnitudes for tracks and phase for derived channels."""
    factory = _factory(test_data_dir, params_dir)
    ic      = factory.benchmark_input_config()

    assert ic.use_primary        and ic.primary_representation        is Representation.MAG_ONLY
    assert ic.use_secondaries    and ic.secondaries_representation    is Representation.MAG_ONLY
    assert ic.use_derived and ic.derived_representation is Representation.ANGLE_ONLY


@pytest.mark.dataset
def test_training_dataset_config_split_regions_share_global_range(test_data_dir, params_dir):
    """Verifies the train, validation, and test regions share the global range extent and carry the configured row spans."""
    factory = _factory(test_data_dir, params_dir)
    crop    = factory.global_crop()
    cfg     = factory.training_dataset_config()

    for split in (cfg.split_regions.train, cfg.split_regions.val, cfg.split_regions.test):
        assert split.col_start == crop.col_start
        assert split.col_end   == crop.col_end

    assert cfg.split_regions.train.as_tuple() == (1000, 1400, crop.col_start, crop.col_end)
    assert cfg.split_regions.val.as_tuple()   == (1400, 1700, crop.col_start, crop.col_end)
    assert cfg.split_regions.test.as_tuple()  == (1700, 2000, crop.col_start, crop.col_end)


@pytest.mark.dataset
def test_training_dataset_config_paths_and_patch(test_data_dir, params_dir):
    """Verifies the dataset config carries the preprocessing paths, labels, patch geometry, batch size, and train shuffling."""
    factory = _factory(test_data_dir, params_dir)
    cfg     = factory.training_dataset_config()

    assert cfg.preprocessing_run_directory == test_data_dir
    assert cfg.parameters_path             == params_dir / "parameters.npy"
    assert cfg.secondary_labels            == ("CH01", "CH02", "CH03", "CH04")
    assert cfg.patch.size                  == (32, 32)
    assert cfg.patch.stride                == (16, 16)
    assert cfg.batch_size                  == 8
    assert cfg.shuffle_train is True


@pytest.mark.dataset
def test_inference_config_propagates_queue_settings(test_data_dir, params_dir):
    """Verifies the inference config carries the run directory, split, checkpoint name, and GIF axes."""
    factory = _factory(test_data_dir, params_dir)
    factory.config.inference.split           = "test"
    factory.config.inference.checkpoint_name = "best_model.pt"

    run_dir = Path("/tmp/some_run")
    cfg     = factory.inference_config(run_dir)

    assert cfg.run_directory   == run_dir
    assert cfg.split           == "test"
    assert cfg.checkpoint_name == "best_model.pt"
    assert cfg.gif_axes        == list(factory.config.inference.gif_axes)


@pytest.mark.dataset
def test_training_trainer_config_adopts_vram_reservation(test_data_dir, params_dir):
    """Verifies the trainer config adopts the configured VRAM reservation and headroom in gibibytes."""
    factory = _factory(test_data_dir, params_dir)
    factory.config.training.reserve_vram      = True
    factory.config.training.vram_keep_free_gb = 2.5

    cfg = factory.training_trainer_config(logdir=Path("/tmp/some_run"))

    assert cfg.memory.reserve_vram      is True
    assert cfg.memory.vram_keep_free_gb == 2.5
