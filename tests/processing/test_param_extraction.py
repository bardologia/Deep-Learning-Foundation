from __future__ import annotations

import importlib.util
import json
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pathlib import Path

from configuration.param_extraction import ExtractParamsEntryConfig
from pipelines.processing.param_extraction.metrics import (
    FittingMetricsCalculator,
    KSelectionDiagnostics,
    ContrastEstimator,
)
from pipelines.processing.param_extraction.pipeline  import ParameterExtractor
from pipelines.processing.param_extraction.inference import ParamRunInferencePipeline
from pipelines.processing.param_extraction.queue     import ExtractionPlanResolver
from pipelines.processing.param_extraction.plots     import FittingResultPlotter
from tools.data.gaussians                            import GaussianMixture
from tools.data.preprocessing                        import ProfilePreprocessor
from tools.monitoring.logger                         import Logger


K_MAX            = 5
LAMBDA_K         = 0.01
HEIGHT_RANGE     = (-20.0, 80.0)
THRESHOLD_FACTOR = 0.25
TRUNCATION_INDEX = 170
ACTIVITY_THRESH  = 0.001

_HAS_JAX = importlib.util.find_spec("jax") is not None


def test_plan_resolver_expands_dataset_k_groups():
    entry = ExtractParamsEntryConfig(
        fit_k_values      = [3, 5],
        fit_lambda_values = [1e-2, 1e-1],
        fit_modes         = ["sigma", "sigma_amp", "sigma_amp_mu"],
    )
    dataset_dirs = [Path("/data/a"), Path("/data/b")]

    groups = ExtractionPlanResolver(entry, dataset_dirs).resolve()

    assert len(groups) == 2 * 2
    assert all(len(group.configs) == 2 * 3 for group in groups)

    configs = [config for group in groups for config in group.configs.values()]
    assert len(configs) == 2 * 2 * 2 * 3

    subdirs = {config.output_subdir_name for config in configs}
    assert len(subdirs) == 2 * 2 * 3


def test_plan_resolver_group_carries_modes_and_lambdas():
    entry = ExtractParamsEntryConfig(fit_k_values=[4], fit_lambda_values=[1e-2, 1e-1], fit_modes=["sigma", "amp_mu"])

    group = ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()[0]

    assert group.k_max         == 4
    assert group.modes         == ["sigma", "amp_mu"]
    assert group.lambda_values == [1e-2, 1e-1]
    assert set(group.configs)  == {("sigma", 1e-2), ("amp_mu", 1e-2), ("sigma", 1e-1), ("amp_mu", 1e-1)}


def test_plan_resolver_maps_modes_to_free_flags():
    entry = ExtractParamsEntryConfig(fit_k_values=[4], fit_lambda_values=[1e-2], fit_modes=["sigma", "amp_mu", "sigma_amp_mu"])

    group = ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()[0]
    flags = [(c.fit_settings.fit_config.fit_sigma, c.fit_settings.fit_config.fit_amplitude, c.fit_settings.fit_config.fit_mean) for c in (group.configs[(mode, 1e-2)] for mode in group.modes)]

    assert flags == [(True, False, False), (False, True, True), (True, True, True)]


def test_plan_resolver_passes_fit_constants_and_adam_settings():
    entry = ExtractParamsEntryConfig(
        fit_k_values           = [3],
        fit_lambda_values      = [1e-2],
        fit_modes              = ["sigma"],
        fit_threshold_factor   = 0.4,
        fit_truncation_index   = 120,
        fit_prominence_frac    = 0.1,
        fit_activity_threshold = 5e-3,
        fit_sigma_init_divisor = 2.0,
        adam_steps             = 500,
        adam_lr                = 0.05,
        range_batch_size       = 100,
        gpu_pixel_batch_size   = 4096,
    )

    plan    = ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()[0].shared
    fit_cfg = plan.fit_settings.fit_config

    assert fit_cfg.threshold_factor   == 0.4
    assert fit_cfg.truncation_index   == 120
    assert fit_cfg.prominence_frac    == 0.1
    assert fit_cfg.activity_threshold == 5e-3
    assert fit_cfg.sigma_init_divisor == 2.0
    assert plan.adam_steps            == 500
    assert plan.adam_lr               == 0.05
    assert plan.range_batch_size      == 100
    assert plan.gpu_pixel_batch_size  == 4096


def test_plan_resolver_rejects_unknown_mode():
    entry = ExtractParamsEntryConfig(fit_k_values=[4], fit_lambda_values=[1e-2], fit_modes=["sigma", "quartic"])
    with pytest.raises(ValueError):
        ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()


def test_plan_resolver_rejects_empty_axis():
    entry = ExtractParamsEntryConfig(fit_k_values=[])
    with pytest.raises(ValueError):
        ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()


def test_plan_resolver_rejects_fixed_suffix_for_multi_permutation():
    entry = ExtractParamsEntryConfig(fit_k_values=[3, 5], output_suffix="fixed")
    with pytest.raises(ValueError):
        ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()


def test_plan_resolver_allows_fixed_suffix_for_single_permutation():
    entry  = ExtractParamsEntryConfig(fit_k_values=[5], fit_lambda_values=[1e-2], fit_modes=["sigma"], output_suffix="fixed")
    groups = ExtractionPlanResolver(entry, [Path("/data/a")]).resolve()
    assert len(groups) == 1
    assert len(groups[0].configs) == 1
    assert groups[0].shared.output_suffix_value == "fixed"


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_kernel_masks_freeze_parameter_groups():
    import jax.numpy as jnp
    from pipelines.processing.param_extraction.sigma.kernels import SigmaAdamKernel

    H      = 60
    height = np.linspace(-10.0, 30.0, H, dtype=np.float32)
    target = np.exp(-((height - 5.0) ** 2) / (2.0 * 4.0 ** 2)).astype(np.float32)
    prof   = np.tile(target[None, :], (4, 1))

    amps = np.full((4, 1), 0.6, dtype=np.float32)
    mus  = np.full((4, 1), 2.0, dtype=np.float32)
    sigs = np.full((4, 1), 6.0, dtype=np.float32)

    kernel = SigmaAdamKernel()

    def run(amp_on, mu_on, sigma_on):
        out = kernel(
            jnp.array(amps), jnp.array(mus), jnp.array(sigs),
            jnp.array(height), jnp.array(prof),
            jnp.float32(amp_on), jnp.float32(mu_on), jnp.float32(sigma_on),
            jnp.float32(height[0]), jnp.float32(height[-1]),
            jnp.float32(0.5), jnp.float32(20.0),
            50, 0.05, 0.9, 0.999,
        )
        return [np.array(o) for o in out]

    a_f, m_f, s_f, e_f = run(1.0, 1.0, 0.0)
    assert np.allclose(s_f, sigs)
    assert not np.allclose(a_f, amps)
    assert not np.allclose(m_f, mus)
    assert e_f.shape == (4,)
    assert np.all(np.isfinite(e_f))

    a_f, m_f, s_f, e_f = run(0.0, 0.0, 1.0)
    assert np.allclose(a_f, amps)
    assert np.allclose(m_f, mus)
    assert not np.allclose(s_f, sigs)

    a_f, m_f, s_f, e_f = run(0.0, 1.0, 0.0)
    assert np.allclose(a_f, amps)
    assert np.allclose(s_f, sigs)
    assert not np.allclose(m_f, mus)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_group_extractor_shares_fits_across_modes_and_lambdas(logger, tmp_path):
    rng      = np.random.default_rng(0)
    H, Az, R = 40, 6, 4
    height   = np.linspace(-10.0, 30.0, H, dtype=np.float32)
    layer    = np.exp(-((height - 8.0) ** 2) / (2.0 * 3.0 ** 2)).astype(np.float32)
    amps     = rng.uniform(0.5, 1.0, size=(Az, R)).astype(np.float32)
    tomo     = layer[:, None, None] * amps[None, :, :]

    tomo_path = tmp_path / "tomo.npy"
    np.save(tomo_path, tomo)

    extractor = ParameterExtractor(
        logger               = logger,
        modes                = ["sigma", "sigma_amp"],
        lambda_values        = [1e-2, 1e-1],
        k_max                = 2,
        threshold_factor     = 0.0,
        truncation_index     = H,
        prominence_frac      = 0.05,
        sigma_init_divisor   = 4.0,
        activity_threshold   = ACTIVITY_THRESH,
        range_batch_size     = 2,
        adam_steps           = 40,
        adam_lr              = 0.1,
        gpu_pixel_batch_size = 64,
        init_workers         = 1,
    )
    results = extractor.run(tomo_path, (-10.0, 30.0))

    assert set(results) == {("sigma", 1e-2), ("sigma", 1e-1), ("sigma_amp", 1e-2), ("sigma_amp", 1e-1)}
    assert all(params.shape == (6, Az, R) for params, _ in results.values())

    for mode in ("sigma", "sigma_amp"):
        lo = results[(mode, 1e-2)][1]
        hi = results[(mode, 1e-1)][1]

        assert np.allclose(lo["mse_per_k"], hi["mse_per_k"], equal_nan=True)
        assert not np.allclose(lo["penalised_per_k"], hi["penalised_per_k"], equal_nan=True)
        assert float(lo["lambda_k"]) == pytest.approx(1e-2)
        assert float(hi["lambda_k"]) == pytest.approx(1e-1)

    sigma_only, sigma_amp = results[("sigma", 1e-2)][1], results[("sigma_amp", 1e-2)][1]
    assert not np.allclose(sigma_only["mse_per_k"], sigma_amp["mse_per_k"], equal_nan=True)


@pytest.fixture(scope="module")
def logger():
    return Logger(log_dir=tempfile.mkdtemp(), name="test_pe", level="ERROR")


@pytest.fixture
def small_metadata():
    return {"height_range": list(HEIGHT_RANGE)}


def _height_axis(H):
    return np.linspace(HEIGHT_RANGE[0], HEIGHT_RANGE[1], H, dtype=np.float32)


@pytest.mark.real_data
def test_parameters_layout(parameters, param_extraction_meta):
    assert param_extraction_meta["k_max"] == K_MAX
    assert parameters.shape[0]            == 3 * K_MAX
    assert parameters.shape[1:]           == (1000, 500)


@pytest.mark.real_data
def test_diagnostics_layout(fit_diagnostics):
    assert fit_diagnostics["mse_per_k"].shape       == (K_MAX, 1000, 500)
    assert fit_diagnostics["penalised_per_k"].shape == (K_MAX, 1000, 500)
    assert fit_diagnostics["best_k_map"].shape      == (1000, 500)
    assert float(fit_diagnostics["lambda_k"])       == pytest.approx(LAMBDA_K, rel=1e-5)


@pytest.mark.real_data
def test_best_k_in_valid_range(fit_diagnostics):
    bk = fit_diagnostics["best_k_map"]

    assert bk.min() >= 0
    assert bk.max() <= K_MAX


@pytest.mark.real_data
def test_parameters_finite(parameters):
    w = np.array(parameters[:, :300, :300])

    assert np.isfinite(w).all()


@pytest.mark.real_data
def test_active_mu_in_physical_height_range(parameters):
    w    = np.array(parameters)
    amps = w[0::3]
    mus  = w[1::3]
    act  = amps > ACTIVITY_THRESH

    assert mus[act].min() >= HEIGHT_RANGE[0] - 1e-2
    assert mus[act].max() <= HEIGHT_RANGE[1] + 1e-2


@pytest.mark.real_data
def test_active_sigma_within_bounds(parameters):
    H      = 150
    h_span = HEIGHT_RANGE[1] - HEIGHT_RANGE[0]
    dh     = h_span / (H - 1)
    w      = np.array(parameters)
    amps   = w[0::3]
    sigs   = w[2::3]
    act    = amps > ACTIVITY_THRESH

    assert sigs[act].min() >= dh - 1e-3
    assert sigs[act].max() <= h_span / 2.0 + 1e-3


@pytest.mark.real_data
def test_amplitudes_nonnegative(parameters):
    w    = np.array(parameters)
    amps = w[0::3]

    assert amps.min() >= 0.0


@pytest.mark.real_data
def test_gaussians_sorted_by_mu(parameters):
    w    = np.array(parameters)
    amps = w[0::3]
    mus  = w[1::3]

    sort_keys   = np.where(amps > ACTIVITY_THRESH, mus, np.inf)
    finite_pair = np.isfinite(sort_keys[:-1]) & np.isfinite(sort_keys[1:])
    lower       = np.where(finite_pair, sort_keys[:-1], 0.0)
    upper       = np.where(finite_pair, sort_keys[1:],  0.0)
    violations  = (upper - lower < -1e-3) & finite_pair

    assert int(violations.sum()) == 0


@pytest.mark.real_data
def test_penalised_equals_mse_plus_penalty(fit_diagnostics):
    mse = fit_diagnostics["mse_per_k"]
    pen = fit_diagnostics["penalised_per_k"]
    bk  = fit_diagnostics["best_k_map"]
    act = bk > 0

    penalty = (pen - mse)[:, act]

    assert np.all(penalty >= -1e-6)


@pytest.mark.real_data
def test_penalty_per_k_bounded_by_lambda(fit_diagnostics):
    mse = fit_diagnostics["mse_per_k"]
    pen = fit_diagnostics["penalised_per_k"]
    bk  = fit_diagnostics["best_k_map"]
    act = bk > 0

    for k in range(K_MAX):
        penalty_k = (pen[k] - mse[k])[act]

        assert np.nanmax(penalty_k) <= LAMBDA_K * (k + 1) + 1e-5


@pytest.mark.real_data
def test_k1_penalty_equals_lambda(fit_diagnostics):
    mse = fit_diagnostics["mse_per_k"]
    pen = fit_diagnostics["penalised_per_k"]
    bk  = fit_diagnostics["best_k_map"]
    act = bk > 0

    penalty_k1 = (pen[0] - mse[0])[act]

    assert np.nanmax(penalty_k1) == pytest.approx(LAMBDA_K, abs=1e-5)


@pytest.mark.real_data
def test_best_k_is_argmin_of_penalised(fit_diagnostics):
    pen = fit_diagnostics["penalised_per_k"]
    bk  = fit_diagnostics["best_k_map"]
    act = bk > 0

    argmin = pen[:, act].argmin(axis=0) + 1

    assert np.array_equal(argmin, bk[act])


@pytest.mark.real_data
def test_inactive_pixels_have_zero_best_k_and_nan_mse(fit_diagnostics):
    mse   = fit_diagnostics["mse_per_k"]
    bk    = fit_diagnostics["best_k_map"]
    inact = bk == 0

    assert np.all(bk[inact] == 0)
    assert np.isnan(mse[:, inact]).all()


@pytest.mark.real_data
def test_stored_mse_matches_reconstruction_from_params(tomogram_full, parameters, fit_diagnostics):
    a0, a1, r0, r1 = 0, 40, 0, 40
    H              = tomogram_full.shape[0]
    height         = _height_axis(H)

    raw = np.abs(np.array(tomogram_full[:, a0:a1, r0:r1])).astype(np.float32)
    raw = ProfilePreprocessor.apply(raw, THRESHOLD_FACTOR, TRUNCATION_INDEX)

    profiles = raw.transpose(2, 1, 0).reshape((r1 - r0) * (a1 - a0), H)
    scale    = profiles.max(axis=1)
    active   = scale > ACTIVITY_THRESH

    parw = np.array(parameters[:, a0:a1, r0:r1])
    amps = parw[0::3].reshape(K_MAX, -1).T
    mus  = parw[1::3].reshape(K_MAX, -1).T
    sigs = parw[2::3].reshape(K_MAX, -1).T

    pred       = GaussianMixture.evaluate_batch(height, amps, mus, sigs)
    safe_scale = np.where(active, scale, 1.0)
    pred_norm  = pred     / safe_scale[:, None]
    prof_norm  = profiles / safe_scale[:, None]
    mse_recon  = ((pred_norm - prof_norm) ** 2).mean(axis=1)

    bk_flat  = fit_diagnostics["best_k_map"][a0:a1, r0:r1].T.reshape(-1)
    mse_w    = fit_diagnostics["mse_per_k"][:, a0:a1, r0:r1].transpose(0, 2, 1).reshape(K_MAX, -1)
    idx      = np.clip(bk_flat - 1, 0, K_MAX - 1)
    mse_best = mse_w[idx, np.arange(len(idx))]

    mask = active & (bk_flat > 0)

    assert mask.sum() > 0
    assert np.nanmedian(np.abs(mse_recon[mask] - mse_best[mask])) < 5e-3


@pytest.mark.real_data
def test_snr_estimator_runs(tomogram_full, logger):
    win = np.array(tomogram_full[:, :32, :32])
    snr = ContrastEstimator(logger).run(win)

    assert snr.shape == (32, 32)
    finite = snr[np.isfinite(snr)]
    assert finite.size > 0
    assert finite.min() >= 0.0


@pytest.mark.real_data
def test_k_selection_diagnostics_runs(fit_diagnostics, logger):
    diag = {
        "mse_per_k"       : fit_diagnostics["mse_per_k"][:, :64, :64],
        "penalised_per_k" : fit_diagnostics["penalised_per_k"][:, :64, :64],
        "best_k_map"      : fit_diagnostics["best_k_map"][:64, :64],
    }
    maps, summary = KSelectionDiagnostics(k_max=K_MAX, logger=logger).run(diag)

    assert "k_margin_second_map" in maps
    assert "n_active_pixels" in summary
    assert summary["n_active_pixels"] >= 0.0


@pytest.mark.real_data
def test_metrics_calculator_runs_on_window(tomogram_full, parameters, fit_diagnostics, small_metadata, logger, tmp_path):
    a0, a1, r0, r1 = 0, 48, 0, 48

    tomo_win  = np.array(tomogram_full[:, a0:a1, r0:r1])
    tomo_path = tmp_path / "tomo.npy"
    np.save(tomo_path, tomo_win)

    parw = np.ascontiguousarray(np.array(parameters[:, a0:a1, r0:r1]))
    diag = {
        "mse_per_k"       : fit_diagnostics["mse_per_k"][:, a0:a1, r0:r1],
        "penalised_per_k" : fit_diagnostics["penalised_per_k"][:, a0:a1, r0:r1],
        "best_k_map"      : fit_diagnostics["best_k_map"][a0:a1, r0:r1],
        "lambda_k"        : fit_diagnostics["lambda_k"],
    }

    calc = FittingMetricsCalculator(
        n_gaussians      = K_MAX,
        logger           = logger,
        threshold_factor = THRESHOLD_FACTOR,
        truncation_index = TRUNCATION_INDEX,
        amp_threshold    = ACTIVITY_THRESH,
    )
    out = calc.run(parw, small_metadata, tomo_path, diag)

    assert out["r2_map"].shape       == (a1 - a0, r1 - r0)
    assert out["activity_map"].shape == (a1 - a0, r1 - r0)
    assert out["global_summary"]["n_gaussians"] == float(K_MAX)
    assert out["activity_map"].max() <= K_MAX


@pytest.mark.real_data
def test_activity_map_counts_active_components(parameters, logger):
    a0, a1, r0, r1 = 0, 60, 0, 60
    parw           = np.array(parameters[:, a0:a1, r0:r1])

    calc = FittingMetricsCalculator(
        n_gaussians      = K_MAX,
        logger           = logger,
        threshold_factor = THRESHOLD_FACTOR,
        truncation_index = TRUNCATION_INDEX,
        amp_threshold    = ACTIVITY_THRESH,
    )
    activity = calc._compute_activity_map(parw)

    expected = (parw[0::3] >= ACTIVITY_THRESH).sum(axis=0)

    assert np.array_equal(activity, expected)


@pytest.mark.real_data
def test_fitting_result_plotter_smoke(tomogram_full, parameters, fit_diagnostics, small_metadata, logger, tmp_path):
    a0, a1, r0, r1 = 0, 48, 0, 48

    tomo_path = tmp_path / "tomo.npy"
    np.save(tomo_path, np.array(tomogram_full[:, a0:a1, r0:r1]))

    parw = np.ascontiguousarray(np.array(parameters[:, a0:a1, r0:r1]))
    diag = {
        "mse_per_k"       : fit_diagnostics["mse_per_k"][:, a0:a1, r0:r1],
        "penalised_per_k" : fit_diagnostics["penalised_per_k"][:, a0:a1, r0:r1],
        "best_k_map"      : fit_diagnostics["best_k_map"][a0:a1, r0:r1],
        "lambda_k"        : fit_diagnostics["lambda_k"],
    }

    calc = FittingMetricsCalculator(
        n_gaussians      = K_MAX,
        logger           = logger,
        threshold_factor = THRESHOLD_FACTOR,
        truncation_index = TRUNCATION_INDEX,
        amp_threshold    = ACTIVITY_THRESH,
    )
    metrics_dict = calc.run(parw, small_metadata, tomo_path, diag)

    plotter = FittingResultPlotter(
        output_directory = tmp_path / "out",
        n_gaussians      = K_MAX,
        logger           = logger,
        threshold_factor = THRESHOLD_FACTOR,
        truncation_index = TRUNCATION_INDEX,
        fig_dpi          = 40,
        save_dpi         = 40,
        n_fits_per_k     = 2,
        amp_threshold    = ACTIVITY_THRESH,
    )
    saved = plotter.run(parw, metrics_dict, small_metadata, tomo_path)

    assert len(saved) > 0
    assert all(p.is_file() for p in saved.values())


@pytest.mark.real_data
def test_param_run_inference_pipeline_smoke(tomogram_full, parameters, fit_diagnostics, logger, tmp_path):
    a0, a1, r0, r1 = 0, 48, 0, 48

    run_dir = tmp_path / "params_run"
    run_dir.mkdir(parents=True)

    tomo_path = tmp_path / "tomo.npy"
    np.save(tomo_path, np.array(tomogram_full[:, a0:a1, r0:r1]))

    np.save(run_dir / "parameters.npy", np.ascontiguousarray(np.array(parameters[:, a0:a1, r0:r1])))
    np.savez(
        run_dir / "fit_diagnostics.npz",
        mse_per_k       = fit_diagnostics["mse_per_k"][:, a0:a1, r0:r1],
        penalised_per_k = fit_diagnostics["penalised_per_k"][:, a0:a1, r0:r1],
        best_k_map      = fit_diagnostics["best_k_map"][a0:a1, r0:r1],
        lambda_k        = fit_diagnostics["lambda_k"],
    )

    meta = {
        "parameters_npy"     : "parameters.npy",
        "diagnostics_npz"    : "fit_diagnostics.npz",
        "source_tomogram"    : str(tomo_path),
        "height_range"       : list(HEIGHT_RANGE),
        "k_max"              : K_MAX,
        "activity_threshold" : ACTIVITY_THRESH,
        "threshold_factor"   : THRESHOLD_FACTOR,
        "truncation_index"   : TRUNCATION_INDEX,
    }
    (run_dir / "param_extraction_meta.json").write_text(json.dumps(meta))

    outputs = ParamRunInferencePipeline(run_dir, logger, make_plots=True).run()

    assert (run_dir / "fit_metrics_summary.json").is_file()
    assert outputs["plots"]
    assert all(p.is_file() for p in outputs["plots"].values())


@pytest.mark.slow
@pytest.mark.real_data
@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_rerun_extractor_reproduces_best_k(tomogram_full, fit_diagnostics, logger, tmp_path):
    a0, a1, r0, r1 = 0, 16, 0, 8

    tomo_path = tmp_path / "tomo.npy"
    np.save(tomo_path, np.array(tomogram_full[:, a0:a1, r0:r1]))

    extractor = ParameterExtractor(
        logger               = logger,
        modes                = ["sigma"],
        lambda_values        = [LAMBDA_K],
        k_max                = K_MAX,
        threshold_factor     = THRESHOLD_FACTOR,
        truncation_index     = TRUNCATION_INDEX,
        prominence_frac      = 0.05,
        sigma_init_divisor   = 4.0,
        activity_threshold   = ACTIVITY_THRESH,
        range_batch_size     = 256,
        adam_steps           = 3000,
        adam_lr              = 2e-1,
        adam_b1              = 0.95,
        adam_b2              = 0.999,
        gpu_pixel_batch_size = 8192,
        init_workers         = 4,
    )
    out, diag = extractor.run(tomo_path, HEIGHT_RANGE)[("sigma", LAMBDA_K)]

    assert out.shape  == (3 * K_MAX, a1 - a0, r1 - r0)
    assert diag["best_k_map"].min() >= 0
    assert diag["best_k_map"].max() <= K_MAX

    bk_stored = fit_diagnostics["best_k_map"][a0:a1, r0:r1]
    active    = bk_stored > 0

    if active.sum() > 0:
        agree = (diag["best_k_map"][active] == bk_stored[active]).mean()
        assert agree >= 0.6
