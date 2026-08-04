from __future__ import annotations

import math

import pytest
import torch

from tools.loss.physical_loss import PhysicalLoss


def _z_axis(n: int = 150) -> torch.Tensor:
    return torch.linspace(-20.0, 80.0, n, dtype=torch.float64)


def _dx(z: torch.Tensor) -> float:
    return float(z[1] - z[0])


def _delta(z: torch.Tensor, z0: float, amp: float = 1.0, b: int = 1, h: int = 1, w: int = 1) -> torch.Tensor:
    curve = torch.zeros(b, z.shape[0], h, w, dtype=torch.float64)
    index = int(torch.argmin((z - z0).abs()))
    curve[:, index] = amp

    return curve, float(z[index])


def _gaussian(z: torch.Tensor, mu: float, sigma: float, amp: float = 1.0) -> torch.Tensor:
    profile = amp * torch.exp(-0.5 * ((z - mu) / sigma) ** 2)

    return profile.reshape(1, -1, 1, 1)


def _kz(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _kz_map(kz: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    return kz.reshape(1, -1, 1, 1).expand(b, kz.shape[0], h, w).contiguous()


def _steering(kz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    phase = kz.reshape(-1, 1) * z.reshape(1, -1)

    return torch.polar(torch.ones_like(phase), phase)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _capon_spectrum(curve: torch.Tensor, kz: torch.Tensor, z: torch.Tensor, dx: float, loading: float) -> torch.Tensor:
    n_tracks = kz.shape[0]
    kz_map   = _kz_map(kz, 1, 1, 1)

    cov   = PhysicalLoss._synthesise_covariance(curve, kz_map, z, dx)
    trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).real / n_tracks
    eye   = torch.eye(n_tracks, dtype=cov.dtype)
    cov   = cov + (loading * trace).unsqueeze(-1).unsqueeze(-1) * eye
    cov   = 0.5 * (cov + cov.conj().transpose(-2, -1))
    cov   = cov.to(torch.complex64)

    spectrum = torch.empty(z.shape[0], dtype=torch.float64)

    for k in range(z.shape[0]):
        phase  = kz_map * z[k]
        steer  = torch.polar(torch.ones_like(phase), phase).to(torch.complex64).permute(0, 2, 3, 1).unsqueeze(-1)
        solved = torch.linalg.solve(cov, steer)
        denom  = (steer.conj() * solved).sum(dim=-2).squeeze(-1).real

        spectrum[k] = (1.0 / denom.clamp(min=1e-12)).reshape(())

    return spectrum


def test_single_scatterer_synth_phase_equals_kz_z0():
    z         = _z_axis()
    dx        = _dx(z)
    kz        = _kz([0.0, 0.3, 0.9, 1.5])
    curve, z0 = _delta(z, 12.0)
    kz_map    = _kz_map(kz, 1, 1, 1)

    synth = torch.stack([PhysicalLoss.synthesise_track(curve, kz_map[:, t], z, dx) for t in range(kz.shape[0])])[:, 0, 0, 0]

    expected = _wrap(kz * z0)
    measured = _wrap(torch.angle(synth))

    assert torch.allclose(measured, expected, atol=1e-6)


def test_single_scatterer_synth_magnitude_is_amplitude_times_dx():
    z        = _z_axis()
    dx       = _dx(z)
    kz       = _kz([0.0, 0.4, 1.1])
    curve, _ = _delta(z, 30.0, amp=2.5)
    kz_map   = _kz_map(kz, 1, 1, 1)

    synth = torch.stack([PhysicalLoss.synthesise_track(curve, kz_map[:, t], z, dx) for t in range(kz.shape[0])])[:, 0, 0, 0]

    assert torch.allclose(synth.abs(), torch.full_like(synth.abs(), 2.5 * dx), atol=1e-9)


def test_single_scatterer_normalized_coherence_is_unit_magnitude():
    z        = _z_axis()
    dx       = _dx(z)
    kz       = _kz([0.0, 0.4, 0.8, 1.3])
    curve, _ = _delta(z, -5.0, amp=3.0)

    p0     = curve.sum(dim=1) * dx
    kz_map = _kz_map(kz, 1, 1, 1)
    synth  = torch.stack([PhysicalLoss.synthesise_track(curve, kz_map[:, t], z, dx) for t in range(kz.shape[0])])[:, 0, 0, 0]
    gamma  = synth / p0.reshape(())

    assert torch.allclose(gamma.abs(), torch.ones_like(gamma.abs()), atol=1e-9)


def test_reference_track_zero_kz_is_real_total_power():
    z     = _z_axis()
    dx    = _dx(z)
    curve = _gaussian(z, 20.0, 9.0) + 0.05

    synth = PhysicalLoss.synthesise_track(curve, torch.zeros(1, 1, 1, dtype=torch.float64), z, dx)
    power = curve.sum(dim=1) * dx

    assert torch.allclose(synth.real, power, atol=1e-9)
    assert torch.allclose(synth.imag, torch.zeros_like(synth.imag), atol=1e-12)


def test_two_scatterer_coherence_matches_closed_form():
    z  = _z_axis()
    dx = _dx(z)
    kz = _kz([0.0, 0.25, 0.6, 1.0, 1.4])

    a1, a2 = 2.0, 1.0
    c1, z1 = _delta(z, 8.0,  amp=a1)
    c2, z2 = _delta(z, 41.0, amp=a2)
    curve  = c1 + c2

    p0     = curve.sum(dim=1) * dx
    kz_map = _kz_map(kz, 1, 1, 1)
    synth  = torch.stack([PhysicalLoss.synthesise_track(curve, kz_map[:, t], z, dx) for t in range(kz.shape[0])])[:, 0, 0, 0]
    gamma  = synth / p0.reshape(())

    closed = (a1 * torch.polar(torch.ones_like(kz), kz * z1) + a2 * torch.polar(torch.ones_like(kz), kz * z2)) / (a1 + a2)

    assert torch.allclose(gamma, closed, atol=1e-9)


def test_normalized_coherence_magnitude_never_exceeds_one():
    z  = _z_axis()
    dx = _dx(z)
    kz = _kz([0.0, 0.2, 0.5, 0.9, 1.3, 1.7])

    gen   = torch.Generator().manual_seed(7)
    curve = torch.rand(3, z.shape[0], 4, 4, generator=gen, dtype=torch.float64)

    p0     = (curve.sum(dim=1) * dx).clamp(min=1e-12)
    kz_map = _kz_map(kz, 3, 4, 4)

    for t in range(kz.shape[0]):
        gamma = PhysicalLoss.synthesise_track(curve, kz_map[:, t], z, dx) / p0

        assert torch.all(gamma.abs() <= 1.0 + 1e-9)


def test_single_scatterer_covariance_is_rank_one_and_hermitian():
    z        = _z_axis()
    dx       = _dx(z)
    kz       = _kz([0.0, 0.3, 0.7, 1.2])
    curve, _ = _delta(z, 15.0, amp=2.0)
    kz_map   = _kz_map(kz, 1, 1, 1)

    cov = PhysicalLoss._synthesise_covariance(curve, kz_map, z, dx)[0, 0, 0]

    assert torch.allclose(cov, cov.conj().transpose(-2, -1), atol=1e-9)

    diag    = torch.diagonal(cov).abs()
    offdiag = cov.abs()

    assert torch.allclose(offdiag, torch.ones_like(offdiag) * diag[0], atol=1e-9)

    eigenvalues = torch.linalg.eigvalsh(cov)
    assert float(eigenvalues[:-1].abs().max()) < 1e-6


def test_covariance_entry_phase_equals_delta_kz_z0():
    z         = _z_axis()
    dx        = _dx(z)
    kz        = _kz([0.0, 0.35, 0.8, 1.25])
    curve, z0 = _delta(z, 22.0)
    kz_map    = _kz_map(kz, 1, 1, 1)

    cov = PhysicalLoss._synthesise_covariance(curve, kz_map, z, dx)[0, 0, 0]

    for i in range(kz.shape[0]):
        for j in range(kz.shape[0]):
            expected = _wrap((kz[i] - kz[j]) * z0)
            measured = _wrap(torch.angle(cov[i, j]).double())

            assert torch.allclose(measured, expected, atol=1e-6)


def test_two_scatterer_covariance_matches_closed_form():
    z  = _z_axis()
    dx = _dx(z)
    kz = _kz([0.0, 0.3, 0.75, 1.2])

    a1, a2 = 1.5, 0.7
    c1, z1 = _delta(z, 5.0,  amp=a1)
    c2, z2 = _delta(z, 37.0, amp=a2)
    curve  = c1 + c2
    kz_map = _kz_map(kz, 1, 1, 1)

    cov = PhysicalLoss._synthesise_covariance(curve, kz_map, z, dx)[0, 0, 0]

    delta_kz = kz.reshape(-1, 1) - kz.reshape(1, -1)
    closed   = dx * (a1 * torch.polar(torch.ones_like(delta_kz), delta_kz * z1) + a2 * torch.polar(torch.ones_like(delta_kz), delta_kz * z2))

    assert torch.allclose(cov.to(torch.complex128), closed.to(torch.complex128), atol=1e-6)


def test_synthesised_covariance_is_positive_semidefinite():
    z  = _z_axis()
    dx = _dx(z)
    kz = _kz([0.0, 0.2, 0.55, 0.9, 1.4])

    gen    = torch.Generator().manual_seed(11)
    curve  = torch.rand(2, z.shape[0], 3, 3, generator=gen, dtype=torch.float64)
    kz_map = _kz_map(kz, 2, 3, 3)

    cov     = PhysicalLoss._synthesise_covariance(curve, kz_map, z, dx)
    eigvals = torch.linalg.eigvalsh(cov)

    assert float(eigvals.min()) > -1e-6


def test_capon_recovers_single_scatterer_peak():
    z         = _z_axis(200)
    dx        = _dx(z)
    kz        = _kz([0.0, 0.2, 0.45, 0.7, 1.0, 1.3])
    curve, z0 = _delta(z, 25.0)

    spectrum = _capon_spectrum(curve, kz, z, dx, loading=0.01)
    peak     = float(z[spectrum.argmax()])

    rayleigh = 2.0 * math.pi / float(kz.max() - kz.min())

    assert abs(peak - z0) < rayleigh


def test_capon_resolves_two_separated_scatterers():
    z  = _z_axis(200)
    dx = _dx(z)
    kz = _kz([0.0, 0.2, 0.45, 0.7, 1.0, 1.3, 1.6, 1.9])

    c1, z1 = _delta(z, 5.0)
    c2, z2 = _delta(z, 45.0)
    curve  = c1 + c2

    spectrum = _capon_spectrum(curve, kz, z, dx, loading=0.005)

    split     = int(torch.argmin((z - 25.0).abs()))
    low_peak  = float(z[:split][spectrum[:split].argmax()])
    high_peak = float(z[split:][spectrum[split:].argmax()])

    rayleigh = 2.0 * math.pi / float(kz.max() - kz.min())

    assert abs(low_peak - z1)  < rayleigh
    assert abs(high_peak - z2) < rayleigh


def test_capon_spectrum_is_strictly_positive():
    z        = _z_axis(120)
    dx       = _dx(z)
    kz       = _kz([0.0, 0.3, 0.7, 1.1, 1.5])
    curve, _ = _delta(z, 10.0)

    spectrum = _capon_spectrum(curve, kz, z, dx, loading=0.02)

    assert torch.all(spectrum > 0.0)


def test_moment_sums_equal_riemann_integral():
    z     = _z_axis()
    dx    = _dx(z)
    curve = _gaussian(z, 25.0, 7.0) + 0.02

    s0, s1, s2 = PhysicalLoss.moment_sums(curve, z, dx)

    x = z.reshape(1, -1, 1, 1)
    assert torch.allclose(s0, (curve * 1.0).sum(dim=1) * dx, atol=1e-12)
    assert torch.allclose(s1, (curve * x).sum(dim=1) * dx,   atol=1e-12)
    assert torch.allclose(s2, (curve * x * x).sum(dim=1) * dx, atol=1e-12)


def test_moments_recover_gaussian_mean_and_spread():
    z         = _z_axis(400)
    dx        = _dx(z)
    mu, sigma = 30.0, 8.0
    curve     = _gaussian(z, mu, sigma)

    s0, s1, s2 = PhysicalLoss.moment_sums(curve, z, dx)
    mean       = (s1 / s0)[0, 0, 0]
    spread     = (s2 / s0 - mean * mean).clamp(min=0.0).sqrt()[0, 0, 0]

    assert abs(float(mean)   - mu)    < 1e-3
    assert abs(float(spread) - sigma) < 1e-2


def test_total_power_relative_error_matches_definition():
    z      = _z_axis()
    dx     = _dx(z)
    pred   = _gaussian(z, 20.0, 6.0) * 1.3
    target = _gaussian(z, 20.0, 6.0)

    p0       = pred.sum(dim=1) * dx
    t0       = target.sum(dim=1) * dx
    expected = ((p0 - t0).abs() / t0).mean()

    assert torch.allclose(PhysicalLoss.total_power(pred, target, dx, 1e-6), expected, atol=1e-9)


def test_kz_sign_is_noop_for_coherence():
    z      = _z_axis(64)
    dx     = _dx(z)
    gen    = torch.Generator().manual_seed(21)
    pred   = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    target = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    kz_map = torch.rand(2, 5, 3, 3, generator=gen, dtype=torch.float64) * 1.4

    a = PhysicalLoss.coherence_resynthesis_pp(pred, target,  kz_map, z, dx, 1e-3)
    b = PhysicalLoss.coherence_resynthesis_pp(pred, target, -kz_map, z, dx, 1e-3)

    assert torch.allclose(a, b, atol=1e-12)


def test_kz_sign_is_noop_for_covariance():
    z      = _z_axis(64)
    dx     = _dx(z)
    gen    = torch.Generator().manual_seed(22)
    pred   = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    target = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    kz_map = torch.rand(2, 5, 3, 3, generator=gen, dtype=torch.float64) * 1.4

    a = PhysicalLoss.covariance_matching_pp(pred, target,  kz_map, z, dx, 1e-3)
    b = PhysicalLoss.covariance_matching_pp(pred, target, -kz_map, z, dx, 1e-3)

    assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.slow
def test_kz_sign_is_noop_for_capon():
    z      = _z_axis(48)
    dx     = _dx(z)
    gen    = torch.Generator().manual_seed(23)
    pred   = torch.rand(1, 48, 2, 2, generator=gen, dtype=torch.float64) + 0.1
    target = torch.rand(1, 48, 2, 2, generator=gen, dtype=torch.float64) + 0.1
    kz_map = torch.rand(1, 4, 2, 2, generator=gen, dtype=torch.float64) * 1.2

    a = PhysicalLoss.capon_cycle_pp(pred, target,  kz_map, z, dx, 1e-2, 1e-3)
    b = PhysicalLoss.capon_cycle_pp(pred, target, -kz_map, z, dx, 1e-2, 1e-3)

    assert torch.allclose(a, b, atol=1e-9)


def test_coherence_is_invariant_to_prediction_amplitude_scaling():
    z        = _z_axis(64)
    dx       = _dx(z)
    gen      = torch.Generator().manual_seed(24)
    pred     = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    target   = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    steering = _steering(_kz([0.0, 0.3, 0.7, 1.1, 1.5]), z)

    base   = PhysicalLoss.coherence_resynthesis(pred,       target, steering, dx, 1e-3)
    scaled = PhysicalLoss.coherence_resynthesis(pred * 7.3, target, steering, dx, 1e-3)

    assert torch.allclose(base, scaled, atol=1e-12)


def test_perpixel_equals_global_on_constant_field_many_kz():
    z      = _z_axis(48)
    dx     = _dx(z)
    gen    = torch.Generator().manual_seed(25)
    pred   = torch.rand(2, 48, 3, 4, generator=gen, dtype=torch.float64) + 0.1
    target = torch.rand(2, 48, 3, 4, generator=gen, dtype=torch.float64) + 0.1

    for values in ([0.0, 0.5], [0.0, 0.3, 0.9, 1.4], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        kz       = _kz(values)
        steering = _steering(kz, z)
        outer    = torch.einsum("ik,jk->ijk", steering, steering.conj())
        kz_map   = _kz_map(kz, 2, 3, 4)

        coh_global = PhysicalLoss.coherence_resynthesis(pred, target, steering, dx, 1e-3)
        coh_pp     = PhysicalLoss.coherence_resynthesis_pp(pred, target, kz_map, z, dx, 1e-3)
        cov_global = PhysicalLoss.covariance_matching(pred, target, outer, dx, 1e-3)
        cov_pp     = PhysicalLoss.covariance_matching_pp(pred, target, kz_map, z, dx, 1e-3)

        assert torch.allclose(coh_global, coh_pp, atol=1e-10)
        assert torch.allclose(cov_global, cov_pp, atol=1e-10)


@pytest.mark.parametrize("term", ["total_power", "moments", "coherence", "covariance"])
def test_all_terms_vanish_on_identical_input(term):
    z        = _z_axis(64)
    dx       = _dx(z)
    gen      = torch.Generator().manual_seed(26)
    curve    = torch.rand(2, 64, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    kz       = _kz([0.0, 0.3, 0.7, 1.1, 1.5])
    steering = _steering(kz, z)
    outer    = torch.einsum("ik,jk->ijk", steering, steering.conj())

    if term == "total_power":
        value = PhysicalLoss.total_power(curve.clone(), curve, dx, 1e-6)
    if term == "moments":
        value = PhysicalLoss.moments(curve.clone(), curve, z, dx, 1e-6, (1.0, 1.0, 1.0))
    if term == "coherence":
        value = PhysicalLoss.coherence_resynthesis(curve.clone(), curve, steering, dx, 1e-6)
    if term == "covariance":
        value = PhysicalLoss.covariance_matching(curve.clone(), curve, outer, dx, 1e-6)

    assert value.item() < 1e-9


def test_floor_masks_low_power_target_pixels():
    z        = _z_axis(32)
    dx       = _dx(z)
    kz       = _kz([0.0, 0.4, 0.9])
    steering = _steering(kz, z)

    pred   = torch.ones(1, 32, 1, 2, dtype=torch.float64)
    target = torch.ones(1, 32, 1, 2, dtype=torch.float64)
    target[:, :, :, 1] = 1e-9

    pred[:, :, :, 0] = 5.0

    high_only = PhysicalLoss.coherence_resynthesis(pred, target, steering, dx, 1e-2)

    target_high = target[:, :, :, :1]
    pred_high   = pred[:,   :, :, :1]
    reference   = PhysicalLoss.coherence_resynthesis(pred_high, target_high, steering, dx, 1e-2)

    assert torch.allclose(high_only, reference, atol=1e-9)


@pytest.mark.parametrize("term", ["coherence", "covariance"])
def test_gradient_is_finite_for_perpixel_terms(term):
    z      = _z_axis(48)
    dx     = _dx(z)
    gen    = torch.Generator().manual_seed(27)
    pred   = (torch.rand(2, 48, 3, 3, generator=gen, dtype=torch.float64) + 0.1).requires_grad_(True)
    target = torch.rand(2, 48, 3, 3, generator=gen, dtype=torch.float64) + 0.1
    kz_map = torch.rand(2, 5, 3, 3, generator=gen, dtype=torch.float64) * 1.3

    if term == "coherence":
        PhysicalLoss.coherence_resynthesis_pp(pred, target, kz_map, z, dx, 1e-3).backward()
    if term == "covariance":
        PhysicalLoss.covariance_matching_pp(pred, target, kz_map, z, dx, 1e-3).backward()

    assert torch.isfinite(pred.grad).all()
