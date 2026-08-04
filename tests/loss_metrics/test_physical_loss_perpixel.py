from __future__ import annotations

import torch

from tools.loss.physical_loss import PhysicalLoss


def _curves(seed: int, b=2, n=16, h=4, w=5) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, h, w, generator=gen, dtype=torch.float64) + 0.1


def _x_axis(n=16) -> torch.Tensor:
    return torch.linspace(-20.0, 80.0, n, dtype=torch.float64)


def _kz_vector(n_tracks=5) -> torch.Tensor:
    return torch.tensor([0.0, 0.3, 0.7, 1.1, 1.6], dtype=torch.float64)[:n_tracks]


def _global_steering(kz: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
    phase = kz.reshape(-1, 1) * x_axis.reshape(1, -1)
    return torch.polar(torch.ones_like(phase), phase)


def _constant_kz_map(kz: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    return kz.reshape(1, -1, 1, 1).expand(b, kz.shape[0], h, w).contiguous()


def test_coherence_perpixel_matches_global_on_constant_field():
    pred, target = _curves(1), _curves(2)
    x            = _x_axis()
    dx           = float(x[1] - x[0])
    kz           = _kz_vector()

    steering = _global_steering(kz, x)
    kz_map   = _constant_kz_map(kz, pred.shape[0], pred.shape[2], pred.shape[3])

    reference = PhysicalLoss.coherence_resynthesis(pred, target, steering, dx, 1e-3)
    perpixel  = PhysicalLoss.coherence_resynthesis_pp(pred, target, kz_map, x, dx, 1e-3)

    assert torch.allclose(reference, perpixel, atol=1e-10)


def test_covariance_perpixel_matches_global_on_constant_field():
    pred, target = _curves(3), _curves(4)
    x            = _x_axis()
    dx           = float(x[1] - x[0])
    kz           = _kz_vector()

    steering = _global_steering(kz, x)
    outer    = torch.einsum("ik,jk->ijk", steering, steering.conj())
    kz_map   = _constant_kz_map(kz, pred.shape[0], pred.shape[2], pred.shape[3])

    reference = PhysicalLoss.covariance_matching(pred, target, outer, dx, 1e-3)
    perpixel  = PhysicalLoss.covariance_matching_pp(pred, target, kz_map, x, dx, 1e-3)

    assert torch.allclose(reference, perpixel, atol=1e-10)


def test_capon_perpixel_matches_global_on_constant_field():
    pred, target = _curves(5), _curves(6)
    x            = _x_axis()
    dx           = float(x[1] - x[0])
    kz           = _kz_vector()

    steering = _global_steering(kz, x)
    outer    = torch.einsum("ik,jk->ijk", steering, steering.conj())
    kz_map   = _constant_kz_map(kz, pred.shape[0], pred.shape[2], pred.shape[3])

    reference = PhysicalLoss.capon_cycle(pred, target, steering, outer, dx, 1e-2, 1e-3)
    perpixel  = PhysicalLoss.capon_cycle_pp(pred, target, kz_map, x, dx, 1e-2, 1e-3)

    assert torch.allclose(reference, perpixel, atol=1e-6)


def test_perpixel_responds_to_spatially_varying_kz():
    pred, target = _curves(7), _curves(8)
    x            = _x_axis()
    dx           = float(x[1] - x[0])
    kz           = _kz_vector()

    uniform = _constant_kz_map(kz, pred.shape[0], pred.shape[2], pred.shape[3])
    varying = uniform.clone()
    varying[:, 1:, :, :] = varying[:, 1:, :, :] * (1.0 + 0.5 * torch.linspace(0.0, 1.0, varying.shape[3], dtype=torch.float64).reshape(1, 1, 1, -1))

    uniform_loss = PhysicalLoss.coherence_resynthesis_pp(pred, target, uniform, x, dx, 1e-3)
    varying_loss = PhysicalLoss.coherence_resynthesis_pp(pred, target, varying, x, dx, 1e-3)

    assert not torch.allclose(uniform_loss, varying_loss)


def test_reference_track_zero_kz_gives_unit_coherence_pair():
    pred   = _curves(9)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    kz_map = _constant_kz_map(_kz_vector(), pred.shape[0], pred.shape[2], pred.shape[3])

    synth = PhysicalLoss.synthesise_track(pred, kz_map[:, 0], x, dx)
    power = pred.sum(dim=1) * dx

    assert torch.allclose(synth.real, power)
    assert torch.allclose(synth.imag, torch.zeros_like(synth.imag), atol=1e-12)

def test_coherence_map_masked_mean_matches_scalar():
    pred, target = _curves(5), _curves(6)
    x            = _x_axis()
    dx           = float(x[1] - x[0])
    kz_map       = _constant_kz_map(_kz_vector(), pred.shape[0], pred.shape[2], pred.shape[3])

    val, mask = PhysicalLoss.coherence_resynthesis_pp_map(pred, target, kz_map, x, dx, 1e-3)
    scalar    = PhysicalLoss.coherence_resynthesis_pp(pred, target, kz_map, x, dx, 1e-3)

    assert val.shape == (pred.shape[0], pred.shape[2], pred.shape[3])
    assert torch.allclose(PhysicalLoss.masked_mean(val, mask), scalar)


def test_covariance_map_masked_mean_matches_scalar():
    pred, target = _curves(7), _curves(8)
    x            = _x_axis()
    dx           = float(x[1] - x[0])
    kz_map       = _constant_kz_map(_kz_vector(), pred.shape[0], pred.shape[2], pred.shape[3])

    val, mask = PhysicalLoss.covariance_matching_pp_map(pred, target, kz_map, x, dx, 1e-3)
    scalar    = PhysicalLoss.covariance_matching_pp(pred, target, kz_map, x, dx, 1e-3)

    assert val.shape == (pred.shape[0], pred.shape[2], pred.shape[3])
    assert torch.allclose(PhysicalLoss.masked_mean(val, mask), scalar)


def test_coherence_map_zero_for_identical_curves():
    pred   = _curves(9)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    kz_map = _constant_kz_map(_kz_vector(), pred.shape[0], pred.shape[2], pred.shape[3])

    val, mask = PhysicalLoss.coherence_resynthesis_pp_map(pred, pred, kz_map, x, dx, 1e-3)

    assert torch.all(mask > 0)
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-12)
