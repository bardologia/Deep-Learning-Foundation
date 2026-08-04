from __future__ import annotations

import json

import pytest

from configuration.sar.gaussian_config    import GaussianConfig
from configuration.training               import BackboneTrainerConfig
from pipelines.shared.config.run_metadata import TrainingRunMetadata
from tools.monitoring.logger              import Logger


def _trainer_config() -> BackboneTrainerConfig:
    gaussian = GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0)

    return BackboneTrainerConfig(gaussian=gaussian)


def _logger(tmp_path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="rm_test", level="ERROR")


def test_directory_layout_created(tmp_path):
    meta = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_x", logger=_logger(tmp_path))

    try:
        assert meta.run_directory      == tmp_path / "run_x"
        assert meta.tensorboard_dir.is_dir()
        assert meta.docs_directory.is_dir()
        assert meta.logs_directory.is_dir()
        assert meta.metadata_directory.is_dir()
    finally:
        meta.close()


def test_io_logdir_and_writer_wired(tmp_path):
    tc   = _trainer_config()
    meta = TrainingRunMetadata(tc, "unet", tmp_path, run_name="run_io", logger=_logger(tmp_path))

    try:
        assert tc.io.logdir == str(meta.run_directory)
        assert tc.io.writer is meta.writer
    finally:
        meta.close()


def test_save_trainer_config_roundtrip(tmp_path):
    meta = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_cfg", logger=_logger(tmp_path))

    try:
        out_path = meta.save_trainer_config()
        payload  = json.loads(out_path.read_text())

        assert out_path == meta.docs_directory / "trainer_config.json"
        assert payload["io"]["writer"] is None
        assert payload["gaussian"]["n_default_gaussians"] == 5
    finally:
        meta.close()


def test_save_run_summary_payload(tmp_path):
    meta = TrainingRunMetadata(_trainer_config(), "swin", tmp_path, run_name="run_sum", logger=_logger(tmp_path))

    try:
        out_path = meta.save_run_summary("swin", in_channels=9, out_channels=15, x_axis_length=150)
        payload  = json.loads(out_path.read_text())

        assert payload["model_name"]    == "swin"
        assert payload["in_channels"]   == 9
        assert payload["out_channels"]  == 15
        assert payload["x_axis_length"] == 150
        assert payload["run_directory"] == str(meta.run_directory)
    finally:
        meta.close()


def test_context_manager_closes_writer(tmp_path):
    with TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_ctx", logger=_logger(tmp_path)) as meta:
        assert meta.run_directory.is_dir()
        assert meta.writer.all_writers is not None

    assert meta.run_directory.is_dir()
    assert meta.writer.all_writers is None
    assert meta.writer.file_writer is None
