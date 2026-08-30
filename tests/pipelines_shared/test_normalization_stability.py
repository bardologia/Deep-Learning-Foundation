"""Tests that the per-slot normalization strategies are the right choice and stable on real data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.normalization.general      import ChannelStats, ChannelStrategy, NormMethod
from pipelines.backbone.dataset.normalizer    import Normalizer
from pipelines.backbone.dataset.stats         import Stats

ROBUST_LOG1P_SLOTS = ["chan/mag", "derived/mag", "out/amp", "out/sigma"]
ZSCORE_SLOTS       = ["out/mu", "chan/phase", "chan/raw_re_im", "derived/raw_re_im", "aux/value"]

AMP_THR       = 1e-2
SCALE_FLOOR   = 1e-8
REL_AGREE_TOL = 0.05


def _active(parameters):
    """Returns the amplitude, mean, and sigma values of pixels whose amplitude exceeds AMP_THR.

    Args:
        parameters : Gaussian parameter cube of shape (3 * n_gaussians, azimuth, range), interleaved as amplitude, mean in metres, and sigma in metres.

    Returns:
        Tuple of the flattened active amplitude, mean, and sigma arrays.
    """
    par = np.asarray(parameters)
    amp = par[0::3].astype(np.float64)
    mu  = par[1::3].astype(np.float64)
    sig = par[2::3].astype(np.float64)
    m   = amp > AMP_THR
    return amp[m], mu[m], sig[m]


def _random_split_agreement(strat, vals, seed):
    """Returns the relative loc and scale disagreement between two random halves of vals."""
    r         = np.random.default_rng(seed).random(vals.shape) < 0.5
    l1, s1    = strat.fit(vals[r])
    l2, s2    = strat.fit(vals[~r])
    loc_rel   = abs(l1 - l2) / (abs(l1) + abs(l2) + 1e-9)
    scale_rel = abs(s1 - s2) / (s1 + s2 + 1e-9)
    return loc_rel, scale_rel


def _channel_stats(slot_keys, pools):
    """Returns channel statistics fitted from the per-slot strategies over the matching pools."""
    strategies = [ChannelStrategy.from_slot(k) for k in slot_keys]
    fits       = [strat.fit(pool) for strat, pool in zip(strategies, pools)]
    return ChannelStats(
        loc        = [f[0] for f in fits],
        scale      = [f[1] for f in fits],
        names      = list(slot_keys),
        strategies = strategies,
        clampable  = [k in ("out/amp", "out/sigma") for k in slot_keys],
    )


def test_heavy_tailed_channels_use_robust_iqr_log1p():
    """Heavy-tailed magnitude and amplitude slots use robust IQR on log1p values."""
    for slot in ROBUST_LOG1P_SLOTS:
        strat = ChannelStrategy.from_slot(slot)
        assert strat.norm_method is NormMethod.ROBUST_IQR
        assert strat.apply_log1p is True


def test_mu_and_signed_channels_stay_zscore():
    """The mean slot and the signed input slots use plain z-score normalization."""
    assert ChannelStrategy.from_slot("out/mu").norm_method is NormMethod.ZSCORE
    assert ChannelStrategy.from_slot("out/mu").apply_log1p is False

    for slot in ZSCORE_SLOTS:
        assert ChannelStrategy.from_slot(slot).norm_method is NormMethod.ZSCORE


def test_derived_phase_uses_fixed_div_pi():
    """The derived phase slot uses the fixed division by pi."""
    strat = ChannelStrategy.from_slot("derived/phase")
    assert strat.norm_method is NormMethod.FIXED_DIV_PI
    assert strat.apply_log1p is False




@pytest.mark.dataset
@pytest.mark.parametrize("role", ["amp", "mu", "sigma"])
def test_output_stats_reproducible_across_random_halves(parameters, role):
    """Output statistics agree within tolerance across random halves of real data."""
    amp, mu, sig = _active(parameters)
    vals  = {"amp": amp, "mu": mu, "sigma": sig}[role]
    slot  = {"amp": "out/amp", "mu": "out/mu", "sigma": "out/sigma"}[role]
    strat = ChannelStrategy.from_slot(slot)

    for seed in range(4):
        loc_rel, scale_rel = _random_split_agreement(strat, vals, seed)
        assert loc_rel   < REL_AGREE_TOL
        assert scale_rel < REL_AGREE_TOL


@pytest.mark.dataset
def test_input_magnitude_stats_reproducible_across_random_halves(primary, secondaries, derived):
    """Input magnitude statistics agree within tolerance across random halves of real data."""
    chan_mag = np.concatenate([np.abs(np.asarray(primary)).ravel(), np.abs(np.asarray(secondaries)).ravel()]).astype(np.float64)
    derived_mag  = np.abs(np.asarray(derived)).ravel().astype(np.float64)

    for slot, vals in [("chan/mag", chan_mag), ("derived/mag", derived_mag)]:
        strat = ChannelStrategy.from_slot(slot)
        loc_rel, scale_rel = _random_split_agreement(strat, vals, seed=0)
        assert loc_rel   < REL_AGREE_TOL
        assert scale_rel < REL_AGREE_TOL


@pytest.mark.dataset
def test_configured_robust_scales_are_nondegenerate_on_scene(parameters, primary, secondaries, derived, aux_full):
    """Every configured robust slot fits a scale well above the floor on scene data."""
    amp, _, sig = _active(parameters)
    chan_mag    = np.abs(np.asarray(primary)).ravel().astype(np.float64)
    derived_mag = np.abs(np.asarray(derived)).ravel().astype(np.float64)
    aux         = np.asarray(aux_full).ravel().astype(np.float64)

    pools = {"out/amp": amp, "out/sigma": sig, "chan/mag": chan_mag, "derived/mag": derived_mag, "aux/value": aux}

    for slot, vals in pools.items():
        _, scale = ChannelStrategy.from_slot(slot).fit(vals)
        assert scale > 1e-3
        assert scale > SCALE_FLOOR * 100


@pytest.mark.dataset
def test_output_normalizer_round_trip_is_exact(parameters):
    """Real output parameters round-trip through the fitted output normalizer."""
    par   = np.asarray(parameters)
    amp, mu, sig = _active(parameters)
    stats = _channel_stats(["out/amp", "out/mu", "out/sigma"], [amp, mu, sig])

    norm = Normalizer(Stats(output_stats=stats))

    t = torch.from_numpy(par[0:3, :64, :64].copy()).unsqueeze(0).float()
    n = norm.normalize_output(t)
    d = norm.denormalize_output(n)

    assert torch.isfinite(n).all()
    assert torch.allclose(d, t, rtol=1e-4, atol=1e-3)


@pytest.mark.dataset
def test_input_normalizer_round_trip_is_exact(primary, derived, secondaries):
    """Real input magnitudes round-trip through the fitted input normalizer."""
    chan_mag = np.concatenate([np.abs(np.asarray(primary)).ravel(), np.abs(np.asarray(secondaries)).ravel()]).astype(np.float64)
    derived_mag  = np.abs(np.asarray(derived)).ravel().astype(np.float64)
    stats    = _channel_stats(["chan/mag", "derived/mag"], [chan_mag, derived_mag])

    norm = Normalizer(Stats(input_stats=stats))

    p = np.abs(np.asarray(primary)[:64, :64]).astype(np.float32)
    i = np.abs(np.asarray(derived)[0, :64, :64]).astype(np.float32)
    t = torch.from_numpy(np.stack([p, i])).unsqueeze(0)

    n = norm.normalize_input(t)
    d = norm.denormalize_input(n)

    assert torch.isfinite(n).all()
    assert torch.allclose(d, t, rtol=1e-4, atol=1e-3)


def test_robust_iqr_scale_is_more_outlier_stable_than_zscore():
    """Adding outliers shifts the robust IQR scale far less than the z-score scale."""
    rng  = np.random.default_rng(0)
    base = rng.normal(5.0, 1.0, size=20000)

    outliers = np.concatenate([base, np.full(400, 500.0)])

    z_clean = ChannelStrategy(NormMethod.ZSCORE,     apply_log1p=False).fit(base)[1]
    z_out   = ChannelStrategy(NormMethod.ZSCORE,     apply_log1p=False).fit(outliers)[1]
    r_clean = ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=False).fit(base)[1]
    r_out   = ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=False).fit(outliers)[1]

    z_drift = abs(z_out - z_clean) / z_clean
    r_drift = abs(r_out - r_clean) / r_clean

    assert r_drift < z_drift
    assert r_drift < 0.05


def test_strategy_fit_rejects_empty_pool():
    """Data-driven strategies reject an empty pool while the fixed pi strategy still fits."""
    import numpy as np
    import pytest as _pytest

    from configuration.normalization import NormMethod
    from configuration.normalization.general import ChannelStrategy

    for method in (NormMethod.ZSCORE, NormMethod.ROBUST_IQR, NormMethod.MIN_MAX_P999):
        with _pytest.raises(ValueError, match="empty sample pool"):
            ChannelStrategy(method).fit(np.array([]))

    loc, scale = ChannelStrategy(NormMethod.FIXED_DIV_PI).fit(np.array([]))
    assert scale == _pytest.approx(np.pi)
