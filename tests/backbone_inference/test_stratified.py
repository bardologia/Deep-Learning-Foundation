"""Tests covering error stratification by covariate, covariate discovery and rank correlation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.stratified import StratifiedErrors
from tools.metrics.scoring                  import RankCorrelation


H, W = 40, 30


def _run(params_gt=None, complex_inputs=None, aux=None, n_secondaries: int = 0, n_gaussians: int = 2) -> SimpleNamespace:
    """Returns a run stub carrying complex inputs, channel counts, slot count, activity threshold and the Aux."""
    return SimpleNamespace(
        complex_inputs     = complex_inputs,
        n_secondaries      = n_secondaries,
        n_gaussians        = n_gaussians,
        activity_threshold = 1e-4,
        dataset            = SimpleNamespace(aux=aux),
    )


def _result(pixel_mse, params_gt=None, label_r2=None) -> SimpleNamespace:
    """Returns an inference result stub with the pixel MSE map, ground-truth parameters and label R2 map."""
    return SimpleNamespace(pixel_mse=pixel_mse, params_gt=params_gt, label_r2=label_r2)


def test_monotone_relation_gives_high_spearman_and_rising_medians():
    """A covariate that drives the error gives a near-unit Spearman, rising bin medians and full pixel coverage."""
    rng = np.random.default_rng(0)
    cov = rng.uniform(0.0, 1.0, size=(H, W))
    err = cov * 10.0 + rng.uniform(0.0, 0.01, size=(H, W))

    strat = StratifiedErrors(_run(), _result(err, label_r2=cov))
    tables, scalars = strat.run()

    rows    = tables["label_r2"]["rows"]
    medians = [row["median"] for row in rows]

    assert scalars["strat_label_r2_spearman"] > 0.99
    assert medians == sorted(medians)
    assert sum(row["n"] for row in rows) == H * W


def test_discrete_covariate_bins_by_level():
    """The ground-truth active count is binned as a discrete covariate, one bin per level."""
    params = np.zeros((6, H, W), dtype=np.float64)

    params[0]           = 1.0
    params[3, :20]      = 1.0

    err = np.ones((H, W))

    strat = StratifiedErrors(_run(n_gaussians=2), _result(err, params_gt=params))
    tables, _scalars = strat.run()

    rows    = tables["gt_active_count"]["rows"]
    centers = [row["center"] for row in rows]

    assert tables["gt_active_count"]["discrete"] is True
    assert centers == [1.0, 2.0]
    assert rows[0]["n"] == 20 * W
    assert rows[1]["n"] == 20 * W


def test_constant_covariate_raises():
    """A covariate with no spread is rejected."""
    err = np.ones((H, W))
    cov = np.full((H, W), 0.5)

    with pytest.raises(ValueError):
        StratifiedErrors(_run(), _result(err, label_r2=cov)).run()


def test_no_covariates_raises():
    """A run offering no usable covariate is rejected."""
    with pytest.raises(ValueError):
        StratifiedErrors(_run(), _result(np.ones((H, W)))).covariates()


def test_amplitude_and_aux_covariates_appear():
    """Complex inputs and an aux channel yield the amplitude and gradient covariates."""
    rng    = np.random.default_rng(1)
    inputs = rng.normal(size=(3, H, W)) + 1j * rng.normal(size=(3, H, W))
    aux    = np.cumsum(rng.uniform(0.0, 1.0, size=(H, W)), axis=0)
    err    = rng.uniform(0.1, 1.0, size=(H, W))

    strat      = StratifiedErrors(_run(complex_inputs=inputs, aux=aux, n_secondaries=2), _result(err))
    covariates = strat.covariates()

    assert set(covariates) == {"primary_amplitude", "aux_gradient"}
    assert covariates["primary_amplitude"][0].shape == (H, W)


def test_spearman_of_anticorrelated_fields_is_negative():
    """Perfectly anticorrelated fields score a Spearman of minus one."""
    cov = np.arange(H * W, dtype=np.float64).reshape(H, W)
    err = -cov

    assert RankCorrelation.spearman(cov, err) == pytest.approx(-1.0)
