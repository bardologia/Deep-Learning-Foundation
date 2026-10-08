"""Tests for the Gaussian slot configuration.

Covers construction and the parameters-per-slot count, the dataclass round
trip, and GaussianConfig.from_dataset reading the slot count and the activity
threshold from the parameter extraction's metadata, which it requires.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from configuration.dataset.gaussian import GaussianConfig


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
