from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.normalization_audit import NormalizationAudit


def _run(normalized_inputs: np.ndarray, n_gaussians: int = 1, amp_max: float = 200.0, enabled: bool = True):
    clamp      = SimpleNamespace(enabled=enabled, floor=0.0, ceil=200.0, amp_max=amp_max)
    stats      = SimpleNamespace(clamp=clamp)
    normalizer = SimpleNamespace(stats=stats)
    dataset    = SimpleNamespace(dem=None, normalizer=normalizer, assemble_window=lambda inputs, dem: normalized_inputs)

    return SimpleNamespace(
        dataset        = dataset,
        complex_inputs = None,
        n_gaussians    = n_gaussians,
        x_axis         = np.linspace(-20.0, 80.0, 10),
    )


def _result(params_pred):
    return SimpleNamespace(params_pred=params_pred)


def test_input_health_reports_mean_std_and_overflow():
    rng     = np.random.default_rng(0)
    ch_ok   = rng.normal(0.0, 1.0, size=(50, 40))
    ch_bad  = np.full((50, 40), 3.0)

    ch_bad[0, :10] = 100.0

    audit = NormalizationAudit(_run(np.stack([ch_ok, ch_bad])), _result(None))
    out   = audit.run()

    assert out["norm_in_ch0_mean"]          == pytest.approx(0.0, abs=0.1)
    assert out["norm_in_ch0_std"]           == pytest.approx(1.0, abs=0.1)
    assert out["norm_in_ch1_overflow_frac"] == pytest.approx(10 / 2000)
    assert out["norm_in_worst_abs_mean"]    >= 2.9
    assert "clamp_n_active" not in out


def test_output_health_counts_clamp_hits_over_active_slots():
    inputs = np.zeros((1, 4, 4))
    params = np.zeros((3, 4, 4), dtype=np.float64)

    params[0]       = 1.0
    params[0, 0, 0] = 200.0
    params[1]       = 30.0
    params[1, 0, 1] = -19.95
    params[2]       = 20.0
    params[2, 0, 2] = 5.0
    params[2, 0, 3] = 49.99

    out = NormalizationAudit(_run(inputs), _result(params)).run()

    assert out["clamp_n_active"]         == 16.0
    assert out["clamp_amp_ceil_frac"]    == pytest.approx(1 / 16)
    assert out["clamp_mu_edge_frac"]     == pytest.approx(1 / 16)
    assert out["clamp_sigma_floor_frac"] == pytest.approx(1 / 16)
    assert out["clamp_sigma_ceil_frac"]  == pytest.approx(1 / 16)


def test_output_health_ignores_inactive_slots():
    inputs = np.zeros((1, 4, 4))
    params = np.zeros((3, 4, 4), dtype=np.float64)

    params[1] = 500.0
    params[2] = 0.0

    out = NormalizationAudit(_run(inputs), _result(params)).run()

    assert out["clamp_n_active"] == 0.0
    assert math.isnan(out["clamp_mu_edge_frac"])


def test_disabled_clamp_reports_nan_amp_ceil():
    inputs = np.zeros((1, 4, 4))
    params = np.ones((3, 4, 4), dtype=np.float64)

    params[2] = 20.0

    out = NormalizationAudit(_run(inputs, enabled=False), _result(params)).run()

    assert math.isnan(out["clamp_amp_ceil_frac"])
    assert out["clamp_sigma_floor_frac"] == pytest.approx(0.0)


def test_degenerate_input_channel_raises():
    inputs = np.full((1, 4, 4), np.nan)

    with pytest.raises(ValueError):
        NormalizationAudit(_run(inputs), _result(None)).run()
