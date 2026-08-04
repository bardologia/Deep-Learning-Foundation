from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.benchmark                import BenchmarkConfig
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.config.run_metadata   import TrainingRunMetadata
from tools.monitoring.logger                import Logger


@pytest.fixture
def trainer_config(test_data_dir, tmp_path):
    config                       = BenchmarkConfig()
    config.paths.dataset_path    = str(test_data_dir)
    config.paths.parameters_path = test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    return ConfigFactory(config).training_trainer_config(tmp_path / "seed_logdir")


@pytest.fixture
def metadata(trainer_config, tmp_path):
    meta = TrainingRunMetadata(
        trainer_config = trainer_config,
        model_name     = "resunet",
        base_logdir    = tmp_path / "runs",
        run_name       = "unit_run",
    )
    yield meta
    meta.close()


@pytest.mark.real_data
def test_run_directory_uses_run_name(metadata, tmp_path):
    assert metadata.run_directory == tmp_path / "runs" / "unit_run"


@pytest.mark.real_data
def test_subdirectories_created(metadata):
    assert metadata.run_directory.is_dir()
    assert metadata.tensorboard_dir.is_dir()
    assert metadata.docs_directory.is_dir()
    assert metadata.logs_directory.is_dir()
    assert metadata.metadata_directory.is_dir()


@pytest.mark.real_data
def test_subdirectories_nested_under_run_directory(metadata):
    assert metadata.tensorboard_dir.parent    == metadata.run_directory
    assert metadata.docs_directory.parent     == metadata.run_directory
    assert metadata.metadata_directory.parent == metadata.run_directory


@pytest.mark.real_data
def test_writer_attached_to_trainer_io(metadata, trainer_config):
    assert trainer_config.io.writer is metadata.writer
    assert trainer_config.io.logdir == str(metadata.run_directory)


@pytest.mark.real_data
def test_default_run_name_includes_model(trainer_config, tmp_path):
    meta = TrainingRunMetadata(
        trainer_config = trainer_config,
        model_name     = "unet",
        base_logdir    = tmp_path / "runs",
    )
    try:
        assert meta.run_directory.name.startswith("run_unet_")
    finally:
        meta.close()


@pytest.mark.real_data
def test_save_trainer_config_serializes_without_writer(metadata):
    out_path = metadata.save_trainer_config()

    assert out_path == metadata.docs_directory / "trainer_config.json"
    payload = json.loads(out_path.read_text())
    assert payload["io"]["writer"] is None


@pytest.mark.real_data
def test_save_run_summary_payload(metadata):
    out_path = metadata.save_run_summary(
        model_name    = "resunet",
        in_channels   = 9,
        out_channels  = 15,
        x_axis_length = 256,
    )

    payload = json.loads(out_path.read_text())
    assert payload["model_name"]    == "resunet"
    assert payload["in_channels"]   == 9
    assert payload["out_channels"]  == 15
    assert payload["x_axis_length"] == 256
    assert payload["framework"]     == "pytorch"
    assert payload["run_directory"] == str(metadata.run_directory)


@pytest.mark.real_data
def test_context_manager_closes_writer(trainer_config, tmp_path):
    with TrainingRunMetadata(trainer_config, "resunet", tmp_path / "runs", run_name="ctx") as meta:
        meta.writer.add_scalar("probe/value", 1.0, 0)
        assert meta.writer.all_writers

    assert meta.writer.all_writers is None

    events = list(meta.tensorboard_dir.glob("events.out.tfevents.*"))
    assert len(events) == 1
    assert events[0].stat().st_size > 0


@pytest.mark.real_data
def test_owns_logger_when_none_passed(metadata):
    assert metadata._owns_logger is True


@pytest.mark.real_data
def test_does_not_own_external_logger(trainer_config, tmp_path):
    logger = Logger(log_dir=str(tmp_path / "ext_logs"), name="external")
    try:
        meta = TrainingRunMetadata(trainer_config, "resunet", tmp_path / "runs", run_name="ext", logger=logger)
        assert meta._owns_logger is False
        assert meta.logger is logger
        meta.close()
    finally:
        logger.close()
