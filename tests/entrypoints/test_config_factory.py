
from __future__ import annotations


import pytest

from configuration.benchmark                import BenchmarkConfig
from configuration.inference                import InferenceConfig
from configuration.training                 import BackboneTrainerConfig
from pipelines.shared.config.config_factory import ConfigFactory


@pytest.fixture
def factory(test_data_dir):
    config                    = BenchmarkConfig()
    config.paths.dataset_path = test_data_dir
    return ConfigFactory(config)


@pytest.fixture
def bare_factory():
    return ConfigFactory(BenchmarkConfig())


@pytest.mark.dataset
def test_training_dataset_config_takes_the_splits_from_the_dataset(factory, dataset_json, test_data_dir):
    dataset_config = factory.training_dataset_config()

    assert dataset_config.dataset_directory      == test_data_dir
    assert dataset_config.split_regions.bounds() == dataset_json["splits"]


@pytest.mark.dataset
def test_training_dataset_config_propagates_loader_settings(factory):
    training       = factory.config.training
    dataset_config = factory.training_dataset_config()

    assert dataset_config.batch_size      == training.batch_size
    assert dataset_config.num_workers     == training.num_workers
    assert dataset_config.prefetch_factor == training.prefetch_factor
    assert dataset_config.shuffle_train   is True
    assert dataset_config.pin_memory      is True


@pytest.mark.dataset
def test_training_dataset_config_propagates_patch_padding(factory):
    factory.config.training.use_symmetric_padding = False

    dataset_config = factory.training_dataset_config()

    assert dataset_config.patch.use_symmetric_padding is False


def test_inference_config_propagates_run_directory(bare_factory, tmp_path):
    run_directory = tmp_path / "run"

    config = bare_factory.inference_config(run_directory)

    assert isinstance(config, InferenceConfig)
    assert config.run_directory == run_directory
    assert config.output_subdir is None
    assert config.device        == "cuda"


def test_inference_config_propagates_inference_fields(bare_factory, tmp_path):
    inference = bare_factory.config.inference

    config = bare_factory.inference_config(tmp_path / "run")

    assert config.checkpoint_name == inference.checkpoint_name
    assert config.split           == inference.split
    assert config.num_workers     == inference.num_workers
    assert config.stitch_window   == inference.stitch_window
    assert config.save_maps       == inference.save_maps


@pytest.mark.dataset
def test_training_trainer_config_logdir(factory, tmp_path):
    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    assert isinstance(trainer_config, BackboneTrainerConfig)
    assert trainer_config.io.logdir == str(tmp_path / "logdir")


@pytest.mark.dataset
def test_training_trainer_config_epochs_propagate(factory, tmp_path):
    training       = factory.config.training
    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    assert trainer_config.training.epochs == training.epochs


@pytest.mark.dataset
def test_training_trainer_config_warmup_propagates(factory, tmp_path):
    training                = factory.config.training
    training.warmup_enabled = False
    training.warmup_fraction = 0.42

    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    assert trainer_config.warmup.warmup_enabled is False
    assert trainer_config.warmup.warmup_fraction == 0.42


@pytest.mark.dataset
def test_training_trainer_config_abort_flag_propagates(factory, tmp_path):
    factory.config.training.abort_on_nonfinite_loss = False

    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    assert trainer_config.training.abort_on_nonfinite_loss is False


@pytest.mark.dataset
def test_training_trainer_config_snapshot_cadence_propagates(factory, tmp_path):
    factory.config.training.snapshot_every_n_epochs = 5

    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    assert trainer_config.training.snapshot_every_n_epochs == 5


@pytest.mark.dataset
def test_training_trainer_config_lr_scale_with_batch(factory, tmp_path):
    training = factory.config.training
    training.scale_lr_with_batch = True

    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    expected = training.batch_size / training.lr_reference_batch_size
    assert trainer_config.optimizer.lr_scale == pytest.approx(expected)


@pytest.mark.dataset
def test_training_trainer_config_lr_scale_disabled(factory, tmp_path):
    factory.config.training.scale_lr_with_batch = False

    trainer_config = factory.training_trainer_config(tmp_path / "logdir")

    assert trainer_config.optimizer.lr_scale == pytest.approx(1.0)
