"""Tests for the Gaussian slot configuration.

Covers construction and the parameters-per-slot count, the dataclass round
trip, GaussianConfig.from_dataset reading the slot count and the activity
threshold from the parameter extraction's metadata, which it requires, and the
guard refusing a parameter run that lives outside its dataset.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from configuration.dataset.gaussian import DatasetParameterPairing, GaussianConfig


def test_gaussian_config_instantiates():
    """Verifies the config holds three parameters per Gaussian and an increasing z range."""
    cfg = GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0, activity_threshold=1e-4)
    assert cfg.params_per_gaussian == 3
    assert cfg.x_max > cfg.x_min


def test_gaussian_config_asdict_round_trips():
    """Verifies the config survives a round trip through dataclasses.asdict."""
    cfg     = GaussianConfig(n_default_gaussians=3, x_min=0.0, x_max=1.0, activity_threshold=1e-3)
    payload = dataclasses.asdict(cfg)
    assert GaussianConfig(**payload) == cfg


def test_gaussian_from_dataset_reads_the_extraction_activity_threshold(test_data_dir, params_dir, param_extraction_meta):
    """Verifies the config read off a dataset takes its slot count and activity threshold from the extraction metadata."""
    gaussian = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")

    assert gaussian.n_default_gaussians == int(param_extraction_meta["k_max"])
    assert gaussian.activity_threshold  == float(param_extraction_meta["activity_threshold"])


def test_gaussian_from_dataset_requires_the_extraction_meta(tmp_path):
    """Verifies a dataset without the extraction metadata is refused with the missing activity threshold named."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "config_state.json").write_text(json.dumps({"cube_config": {"z_range": [-20.0, 80.0]}}))

    with pytest.raises(FileNotFoundError, match="activity threshold"):
        GaussianConfig.from_dataset(tmp_path, tmp_path / "params" / "parameters.npy")


def test_parameter_pairing_returns_the_relative_run_layout():
    """Verifies the parameter template is returned relative to the dataset root."""
    template = DatasetParameterPairing.relative_template(Path("/data/ds"), Path("/data/ds/params/run_a/parameters.npy"))
    assert template == Path("params/run_a/parameters.npy")


def test_parameter_pairing_rejects_a_foreign_parameter_run():
    """Verifies a parameter run outside the dataset root is rejected."""
    with pytest.raises(ValueError, match="must live inside"):
        DatasetParameterPairing.relative_template(Path("/data/ds_b"), Path("/data/ds_a/params/run_a/parameters.npy"))


def test_parameter_pairing_rejects_a_sibling_sharing_the_dataset_prefix():
    """Verifies a dataset whose name extends the paired dataset's name is not taken for it."""
    with pytest.raises(ValueError, match="must live inside"):
        DatasetParameterPairing.relative_template(Path("/data/ds"), Path("/data/ds_b/params/run_a/parameters.npy"))


def test_parameter_pairing_compares_resolved_paths(tmp_path, monkeypatch):
    """Verifies a checkout-relative dataset pairs with the rebased absolute parameter path and through a symlinked root."""
    dataset  = tmp_path / "datasets" / "ds"
    params   = dataset / "params" / "run_a" / "parameters.npy"
    linked   = tmp_path / "linked"
    dotted   = Path("datasets/ds/../ds/params/run_a/parameters.npy")
    expected = Path("params/run_a/parameters.npy")

    params.parent.mkdir(parents=True)
    os.symlink(tmp_path / "datasets", linked)
    monkeypatch.chdir(tmp_path)

    assert DatasetParameterPairing.relative_template(Path("datasets/ds"), params) == expected
    assert DatasetParameterPairing.relative_template(linked / "ds",       params) == expected
    assert DatasetParameterPairing.relative_template(dataset,             dotted) == expected


def test_gaussian_from_dataset_rejects_a_foreign_parameter_run(tmp_path):
    """Verifies GaussianConfig.from_dataset rejects parameters belonging to another dataset."""
    dataset_a = tmp_path / "ds_a"
    dataset_b = tmp_path / "ds_b"
    params    = dataset_a / "params" / "run_a"

    (dataset_b / "meta").mkdir(parents=True)
    params.mkdir(parents=True)

    (dataset_b / "meta" / "config_state.json").write_text(json.dumps({"cube_config": {"z_range": [-20.0, 80.0]}}))
    (params / "param_extraction_meta.json").write_text(json.dumps({"k_max": 2, "activity_threshold": 1e-3}))

    with pytest.raises(ValueError, match="must live inside"):
        GaussianConfig.from_dataset(dataset_b, params / "parameters.npy")
