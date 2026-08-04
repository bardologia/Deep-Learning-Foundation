from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.probes          import ProbeWindows
from pipelines.backbone.inference.analysis.receptive_field import ErfComputation


WINDOW = 16


class _ConvModel(torch.nn.Module):
    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=kernel, padding=kernel // 2, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        return self.conv(x)


class _DeadModel(torch.nn.Module):
    def forward(self, x):
        return x * 0.0


def _windows(n_probes: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.uniform(0.5, 1.0, size=(n_probes, 2, WINDOW, WINDOW))).float()


def test_conv3_mass_is_complete_inside_a_small_window():
    erf  = ErfComputation(_ConvModel(3), WINDOW, [4, 8, 16])
    grad = erf.gradient_map(_windows())

    masses = erf.mass_profile(grad)

    assert grad.shape  == (WINDOW, WINDOW)
    assert masses[4]   == pytest.approx(1.0)
    assert masses[16]  == pytest.approx(1.0)


def test_wider_kernel_has_wider_sigma():
    erf3  = ErfComputation(_ConvModel(3), WINDOW, [8])
    erf9  = ErfComputation(_ConvModel(9), WINDOW, [8])

    sigma3 = erf3.sigma(erf3.gradient_map(_windows()))
    sigma9 = erf9.sigma(erf9.gradient_map(_windows()))

    assert sigma9[0] > sigma3[0]
    assert sigma9[1] > sigma3[1]


def test_dead_model_raises():
    erf = ErfComputation(_DeadModel(), WINDOW, [8])

    with pytest.raises(ValueError):
        erf.gradient_map(_windows())


def test_mass_windows_exceeding_the_probe_window_raise():
    with pytest.raises(ValueError):
        ErfComputation(_ConvModel(3), WINDOW, [8, 32])


def _region_run(n_az: int, n_rg: int) -> SimpleNamespace:
    return SimpleNamespace(split_region=SimpleNamespace(azimuth_size=n_az, range_size=n_rg))


def test_probe_centers_keep_the_window_margin():
    centers = ProbeWindows(_region_run(32, 20), window=8).centers(3, 2)

    azs = sorted({az for az, _rg in centers})
    rgs = sorted({rg for _az, rg in centers})

    assert len(centers) == 6
    assert azs[0] >= 4 and azs[-1] <= 27
    assert rgs[0] >= 4 and rgs[-1] <= 15


def test_probe_centers_raise_when_the_region_is_too_small():
    with pytest.raises(ValueError):
        ProbeWindows(_region_run(32, 200), window=64).centers(2, 2)
