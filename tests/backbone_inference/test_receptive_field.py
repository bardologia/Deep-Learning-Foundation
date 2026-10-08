"""Tests covering effective-receptive-field gradient maps, mass profiles and probe placement."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.diagnostics                             import ReceptiveFieldConfig
from pipelines.backbone.inference.analysis.receptive_field import ErfComputation, RunReceptiveField
from pipelines.backbone.inference.probes                   import ProbeWindows

from tests.backbone_inference._helpers import build_run_directory
from tests.conftest                    import RecordingLogger


WINDOW = 16


class _ConvModel(torch.nn.Module):
    """Single all-ones convolution mapping two input channels to three parameter channels."""
    def __init__(self, kernel: int) -> None:
        """Builds the convolution with the given square kernel size and same padding."""
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=kernel, padding=kernel // 2, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        """Returns the convolution of x, an input of shape (batch, 2, height, width)."""
        return self.conv(x)


class _DeadModel(torch.nn.Module):
    """Model whose output is identically zero, so every input gradient vanishes."""
    def forward(self, x):
        """Returns x scaled by zero, of shape (batch, channels, height, width)."""
        return x * 0.0


class _AmpOnlyModel(torch.nn.Module):
    """Model whose amplitude channel depends on the input while centre and width are constant."""
    def __init__(self) -> None:
        """Builds the all-ones 3x3 convolution feeding the amplitude channel."""
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        """Returns three parameter channels of shape (batch, 3, height, width), only the first depending on x."""
        amp = self.conv(x)
        return torch.cat([amp, torch.ones_like(amp) * 5.0, torch.ones_like(amp) * 2.0], dim=1)


def _windows(n_probes: int = 2) -> torch.Tensor:
    """Returns uniform random probe windows of shape (n_probes, 2, WINDOW, WINDOW)."""
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.uniform(0.5, 1.0, size=(n_probes, 2, WINDOW, WINDOW))).float()


def test_conv3_mass_is_complete_inside_a_small_window():
    """A 3x3 convolution puts all gradient mass inside the smallest measured window."""
    erf     = ErfComputation(_ConvModel(3), WINDOW, [4, 8, 16])
    maps, _ = erf.gradient_maps(_windows())
    grad    = maps["combined"]

    masses = erf.mass_profile(grad)

    assert grad.shape  == (WINDOW, WINDOW)
    assert masses[4]   == pytest.approx(1.0)
    assert masses[16]  == pytest.approx(1.0)


def test_wider_kernel_has_wider_sigma():
    """A 9x9 kernel gives larger row and column gradient sigmas than a 3x3 kernel."""
    erf3 = ErfComputation(_ConvModel(3), WINDOW, [8])
    erf9 = ErfComputation(_ConvModel(9), WINDOW, [8])

    sigma3 = erf3.sigma(erf3.gradient_maps(_windows())[0]["combined"])
    sigma9 = erf9.sigma(erf9.gradient_maps(_windows())[0]["combined"])

    assert sigma9[0] > sigma3[0]
    assert sigma9[1] > sigma3[1]


def test_dead_model_raises():
    """A model with zero input gradients is rejected."""
    erf = ErfComputation(_DeadModel(), WINDOW, [8])

    with pytest.raises(ValueError):
        erf.gradient_maps(_windows())


def test_mass_windows_exceeding_the_probe_window_raise():
    """Mass windows larger than the probe window are rejected at construction."""
    with pytest.raises(ValueError):
        ErfComputation(_ConvModel(3), WINDOW, [8, 32])


def test_family_maps_follow_channel_dependence():
    """Parameter families without input dependence get no gradient map, and the combined map follows the dependent family."""
    erf     = ErfComputation(_AmpOnlyModel(), WINDOW, [8])
    maps, _ = erf.gradient_maps(_windows())

    assert maps["amp"]   is not None
    assert maps["mu"]    is None
    assert maps["sigma"] is None
    assert np.allclose(maps["combined"], maps["amp"])


def test_probe_sigmas_cover_every_probe():
    """Per-probe sigmas are returned for every probe on both axes and stay positive."""
    erf              = ErfComputation(_ConvModel(3), WINDOW, [8])
    _, probe_sigmas  = erf.gradient_maps(_windows(n_probes=3))

    assert probe_sigmas.shape == (3, 2)
    assert (probe_sigmas > 0.0).all()


def test_fine_mass_curve_is_monotone_and_reaches_one():
    """The fine mass curve is non-decreasing and reaches one at the full window side."""
    erf         = ErfComputation(_ConvModel(3), WINDOW, [8])
    maps, _     = erf.gradient_maps(_windows())
    sides, fine = erf.fine_mass_curve(maps["combined"])

    assert sides[-1] == WINDOW
    assert fine[-1]  == pytest.approx(1.0)
    assert (np.diff(fine) >= -1e-12).all()


def test_side_at_mass_interpolates_between_samples():
    """Side lookup interpolates between mass samples and clamps outside their col."""
    sides  = np.array([2.0, 4.0, 6.0])
    masses = np.array([0.4, 0.8, 1.0])

    assert ErfComputation.side_at_mass(sides, masses, 0.6)  == pytest.approx(3.0)
    assert ErfComputation.side_at_mass(sides, masses, 0.1)  == pytest.approx(2.0)
    assert ErfComputation.side_at_mass(sides, masses, 1.0)  == pytest.approx(6.0)


def test_gaussian_grad_matches_gaussian_mass_reference():
    """A Gaussian gradient map encloses the analytic two-sigma mass fraction."""
    erf    = ErfComputation(_ConvModel(3), WINDOW, [8])
    center = WINDOW // 2
    coords = np.arange(WINDOW) - center
    rr, cc = np.meshgrid(coords, coords, indexing="ij")
    grad   = np.exp(-(rr ** 2 + cc ** 2) / (2.0 * 2.0 ** 2))

    sigma_row, sigma_col = erf.sigma(grad)
    within             = erf.mass_within_2sigma(grad, sigma_row, sigma_col)

    assert within == pytest.approx(0.911, abs=0.05)


def test_gaussian_mass_requires_positive_sigmas():
    """A non-positive sigma is rejected by the Gaussian mass helper."""
    with pytest.raises(ValueError):
        ErfComputation.gaussian_mass(np.array([2.0]), 0.0, 1.0)


def _region_run(n_rows: int, n_cols: int) -> SimpleNamespace:
    """Returns a run stub whose split region spans n_rows by n_cols pixels."""
    return SimpleNamespace(split_region=SimpleNamespace(row_size=n_rows, col_size=n_cols))


def test_probe_centers_keep_the_window_margin():
    """Probe centres are placed so the full window fits inside the region."""
    centers = ProbeWindows(_region_run(32, 20), window=8).centers(3, 2)

    rows = sorted({row for row, _col in centers})
    cols = sorted({col for _row, col in centers})

    assert len(centers) == 6
    assert rows[0] >= 4 and rows[-1] <= 28
    assert cols[0] >= 4 and cols[-1] <= 16


def test_probe_centers_fit_an_odd_window():
    """Odd-sized probe windows stay inside the region on both axes."""
    window  = 9
    half    = window // 2
    centers = ProbeWindows(_region_run(32, 20), window=window).centers(3, 2)

    for row, col in centers:
        assert row - half >= 0 and row - half + window <= 32
        assert col - half >= 0 and col - half + window <= 20


def test_probe_centers_raise_when_the_region_is_too_small():
    """A window wider than the region raises."""
    with pytest.raises(ValueError):
        ProbeWindows(_region_run(32, 200), window=64).centers(2, 2)


def test_zero_gradient_probe_gets_nan_sigmas():
    """A probe whose window carries no gradient is kept as a NaN row instead of vanishing from the spread."""
    erf     = ErfComputation(_ConvModel(3), WINDOW, [8])
    windows = _windows(n_probes=2)

    windows[1] = 0.0

    _, probe_sigmas = erf.gradient_maps(windows)

    assert probe_sigmas.shape == (2, 2)
    assert np.isfinite(probe_sigmas[0]).all() and np.isnan(probe_sigmas[1]).all()


def test_non_finite_gradient_raises_instead_of_reading_no_dependence():
    """A NaN input gradient raises naming the family rather than being reported as 'no dependence'."""
    class _NanModel(torch.nn.Module):
        """Model scaling its input by NaN, so every input gradient is NaN."""
        def forward(self, x):
            """Returns NaN-scaled copies of channel 0, shape (batch, 3, H, W)."""
            return (x[:, :1] * float("nan")).expand(-1, 3, -1, -1)

    with pytest.raises(ValueError, match="centre amp at probe window 0 is non-finite"):
        ErfComputation(_NanModel(), WINDOW, [8]).gradient_maps(_windows())


@pytest.mark.dataset
@pytest.mark.slow
def test_erf_pass_records_how_many_probes_entered_the_spread(tmp_path, test_data_dir, params_dir):
    """An ERF pass on a real run stores the valid and total probe counts next to the sigma spread."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    config  = ReceptiveFieldConfig(window=32, n_row_probes=2, n_col_probes=2, mass_windows=[8, 16])

    outcome = RunReceptiveField(run_dir, config, RecordingLogger()).run()
    spread  = outcome.payload["probe_sigma_spread"]

    assert spread["n_probes"] == 4 and 0 < spread["n_valid_probes"] <= 4
