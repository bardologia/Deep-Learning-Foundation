from __future__ import annotations

import dataclasses

from configuration.dataset.gaussian import GaussianConfig


def test_gaussian_config_instantiates():
    cfg = GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0)
    assert cfg.params_per_gaussian == 3
    assert cfg.x_max > cfg.x_min


def test_gaussian_config_asdict_round_trips():
    cfg     = GaussianConfig(n_default_gaussians=3, x_min=0.0, x_max=1.0)
    payload = dataclasses.asdict(cfg)
    assert GaussianConfig(**payload) == cfg
