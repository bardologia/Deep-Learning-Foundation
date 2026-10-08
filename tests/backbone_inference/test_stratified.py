"""Tests covering error stratification by covariate, covariate discovery and rank correlation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.pipeline           import InferencePipeline
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.stratified         import StratifiedErrors
from tools.metrics.scoring                           import RankCorrelation

from tests.conftest import RecordingLogger


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


def test_sparse_discrete_level_is_dropped_but_counted():
    """A GT active-count level under MIN_BIN_COUNT pixels leaves the rows, yet its centre and pixel count are recorded with the dropped-bin scalars."""
    params = np.zeros((6, H, W), dtype=np.float64)

    params[0]         = 1.0
    params[3, 0, :10] = 1.0

    err = np.ones((H, W))

    tables, scalars = StratifiedErrors(_run(n_gaussians=2), _result(err, params_gt=params)).run()

    assert [row["center"] for row in tables["gt_active_count"]["rows"]] == [1.0]
    assert tables["gt_active_count"]["dropped"]              == [{"center": 2.0, "n": 10}]
    assert scalars["strat_gt_active_count_n_dropped_bins"]   == 1.0
    assert scalars["strat_gt_active_count_pixels_dropped"]   == 10.0
    assert scalars["strat_gt_active_count_pixels_nonfinite"] == 0.0


def test_non_finite_pairs_are_counted_and_every_other_pixel_lands_in_a_bin():
    """Pairs with a NaN error or an infinite covariate are counted as non-finite, and the kept and dropped bins account for every remaining pixel."""
    rng = np.random.default_rng(2)
    cov = rng.uniform(0.0, 1.0, size=(H, W))
    err = cov * 10.0 + rng.uniform(0.0, 0.01, size=(H, W))

    err[0, :5] = np.nan
    cov[1, :3] = np.inf

    tables, scalars = StratifiedErrors(_run(), _result(err, label_r2=cov)).run()

    kept = sum(row["n"] for row in tables["label_r2"]["rows"])

    assert scalars["strat_label_r2_pixels_nonfinite"] == 8.0
    assert kept + scalars["strat_label_r2_pixels_dropped"] == H * W - 8


def test_pipeline_stage_warns_naming_the_dropped_bins_and_non_finite_pixels(tmp_path):
    """The pipeline stage warns once per covariate that lost bins or pixels, naming the bin centres and counts."""
    params = np.zeros((6, H, W), dtype=np.float64)

    params[0]         = 1.0
    params[3, 0, :10] = 1.0

    err = np.random.default_rng(6).uniform(size=(H, W))
    err[1, :4] = np.nan

    config = InferenceConfig(run_directory=tmp_path, output_subdir="strat", device="cpu", compute_stratified=True)
    meta   = InferenceMetadata(config)
    meta.create_dirs()

    logger  = RecordingLogger()
    result  = _result(err, params_gt=params)
    metrics = {}

    InferencePipeline(config)._stratify_errors(config, meta, _run(n_gaussians=2), result, metrics, logger)

    warnings = [msg for level, msg in logger.messages if level == "warning"]

    assert metrics["strat_gt_active_count_n_dropped_bins"] == 1.0
    assert warnings == [
        "Stratified errors, gt_active_count: 1 bin(s) under 30 pixels left out of the table and figure, 10 pixels in all: 2 (n=10)",
        "Stratified errors, gt_active_count: 4 pixels with a non-finite covariate or error left out of every bin",
    ]


def test_pipeline_stage_names_the_strongest_finite_correlation_over_a_constant_covariate(tmp_path):
    """A covariate constant on the region has an undefined Spearman, is named as such, and never wins the strongest-correlation line."""
    rng    = np.random.default_rng(4)
    params = np.zeros((6, H, W), dtype=np.float64)
    cov    = rng.uniform(0.0, 1.0, size=(H, W))
    err    = cov * 10.0 + rng.uniform(0.0, 0.01, size=(H, W))

    params[0] = 1.0

    config = InferenceConfig(run_directory=tmp_path, output_subdir="strat", device="cpu", compute_stratified=True)
    meta   = InferenceMetadata(config)
    meta.create_dirs()

    logger = RecordingLogger()

    InferencePipeline(config)._stratify_errors(config, meta, _run(n_gaussians=2), _result(err, params_gt=params, label_r2=cov), {}, logger)

    lines = [msg for level, msg in logger.messages if level == "subsection"]

    assert "No rank correlation with error for ['gt_active_count']: the covariate or the error is constant on this region" in lines
    assert lines[-1].startswith("Stratified over ['gt_active_count', 'label_r2']; strongest rank correlation with error: label_r2 (1.000)")


def test_pipeline_stage_warns_when_every_correlation_is_undefined(tmp_path):
    """A region whose only covariate is constant yields a warning instead of a NaN strongest correlation."""
    params = np.zeros((6, H, W), dtype=np.float64)

    params[0] = 1.0

    config = InferenceConfig(run_directory=tmp_path, output_subdir="strat", device="cpu", compute_stratified=True)
    meta   = InferenceMetadata(config)
    meta.create_dirs()

    logger = RecordingLogger()

    InferencePipeline(config)._stratify_errors(config, meta, _run(n_gaussians=2), _result(np.random.default_rng(5).uniform(size=(H, W)), params_gt=params), {}, logger)

    assert ("warning", "Stratified over ['gt_active_count'], but every rank correlation with error is undefined; no covariate can be named as the strongest") in logger.messages
