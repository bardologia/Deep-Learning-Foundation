from __future__ import annotations

from pathlib import Path

import pytest

from configuration.patch_sweep              import PatchSweepConfig
from pipelines.patch_sweep.planner          import PatchSweepPlanner
from pipelines.patch_sweep.workers          import SweepTrainingWorker
from pipelines.shared.config.config_factory import ConfigFactory


def make_worker(test_data_dir, params_dir, tmp_path: Path, batch_size: int | None = None) -> SweepTrainingWorker:
    config                       = PatchSweepConfig()
    config.dataset_base_path     = test_data_dir.parent
    config.dataset_filter        = [test_data_dir.name]
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"
    config.paths.log_base_dir    = tmp_path

    if batch_size is not None:
        config.training.batch_size = batch_size

    return SweepTrainingWorker(config=config, run_tag="rt")


def apply_unit(worker: SweepTrainingWorker, unit_name: str) -> None:
    unit = PatchSweepPlanner(worker.config).unit(unit_name)
    worker._apply_unit(unit)


@pytest.mark.real_data
def test_unit_reroots_the_dataset_and_parameters(test_data_dir, params_dir, tmp_path):
    worker = make_worker(test_data_dir, params_dir, tmp_path)
    apply_unit(worker, f"{test_data_dir.name}-p016x016")

    assert worker.config.paths.dataset_path    == test_data_dir
    assert worker.config.paths.parameters_path == params_dir / "parameters.npy"


@pytest.mark.real_data
def test_small_patch_unit_keeps_the_lr_scale_at_one(test_data_dir, params_dir, tmp_path):
    worker    = make_worker(test_data_dir, params_dir, tmp_path)
    training  = worker.config.training
    reference = training.batch_size * training.patch_size[0] * training.patch_size[1]
    apply_unit(worker, f"{test_data_dir.name}-p016x016")

    assert worker.config.training.batch_size              == reference // (16 * 16)
    assert worker.config.training.lr_reference_batch_size == worker.config.training.batch_size

    trainer_config = ConfigFactory(worker.config).training_trainer_config(logdir=tmp_path / "run")

    assert trainer_config.optimizer.lr_scale == pytest.approx(1.0)


@pytest.mark.real_data
def test_unit_preserves_the_configured_lr_scale_ratio(test_data_dir, params_dir, tmp_path):
    worker = make_worker(test_data_dir, params_dir, tmp_path, batch_size=512)
    apply_unit(worker, f"{test_data_dir.name}-p016x016")

    trainer_config = ConfigFactory(worker.config).training_trainer_config(logdir=tmp_path / "run")

    assert trainer_config.optimizer.lr_scale == pytest.approx(2.0)
