"""Tests for Gaussian mixture evaluation, curve reconstruction, parameter clamping and the slot activity predicate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.data.gaussians import (
    GaussianActivity,
    GaussianHead,
    GaussianMixture,
    GaussianReconstructor,
    GaussianSlotSorter,
)
from tools.data.gaussians_torch import (
    GaussianClamp,
    GaussianCurve,
)


def test_activity_is_strictly_above_the_threshold():
    """Verifies an amplitude equal to the threshold is inactive and only larger ones are active, for arrays, floats and tensors alike."""
    amps = np.array([0.0, 1e-3, 1e-3 + 1e-9, 5e-3], dtype=np.float64)

    assert GaussianActivity.is_active(amps, 1e-3).tolist() == [False, False, True, True]
    assert GaussianActivity.is_active(1e-3, 1e-3) is False
    assert GaussianActivity.is_active(torch.tensor([1e-3, 2e-3]), 1e-3).tolist() == [False, True]


def test_zero_threshold_keeps_stored_zeros_inactive():
    """Verifies a zero threshold means any positive amplitude, so padded and unfitted slots at exactly zero stay inactive."""
    amps = np.array([0.0, 1e-10, 3e-5], dtype=np.float32)

    assert GaussianActivity.is_active(amps, 0.0).tolist() == [False, True, True]


def test_slot_sorter_treats_threshold_amplitude_as_inactive():
    """Verifies the mean sorter pushes a slot sitting exactly on the threshold behind the active slots."""
    cube = np.zeros((6, 1, 1), dtype=np.float32)
    cube[0, 0, 0], cube[1, 0, 0] = 1e-3, 5.0
    cube[3, 0, 0], cube[4, 0, 0] = 2e-3, 20.0

    ordered = GaussianSlotSorter.by_mean(cube, 2, 1e-3)

    assert ordered[1, 0, 0] == 20.0
    assert ordered[4, 0, 0] == 5.0


def test_head_total_channels():
    """Verifies the head channel count is the parameter count times the Gaussian count."""
    assert GaussianHead.total_channels(3, 4) == 12


def test_safe_sigma_sq_floor():
    """Verifies a zero sigma is lifted to twice the squared sigma floor."""
    out = GaussianMixture.safe_sigma_sq(np.array([0.0]))

    assert out[0] == 2.0 * GaussianMixture.SIGMA_FLOOR ** 2


def test_safe_sigma_sq_value():
    """Verifies the safe denominator is twice the squared sigma."""
    out = GaussianMixture.safe_sigma_sq(np.array([3.0]))

    assert np.isclose(out[0], 18.0)


def test_evaluate_batch_peak_at_mu():
    """Verifies a single Gaussian peaks at its mean with its amplitude over the height axis."""
    h    = np.linspace(-10.0, 10.0, 201).astype(np.float32)
    amps = np.array([[2.0]], dtype=np.float32)
    mus  = np.array([[0.0]], dtype=np.float32)
    sigs = np.array([[2.0]], dtype=np.float32)

    pred = GaussianMixture.evaluate_batch(h, amps, mus, sigs)

    assert pred.shape == (1, 201)
    assert np.isclose(pred[0].max(), 2.0, atol=1e-4)
    assert np.argmax(pred[0]) == 100


def test_evaluate_batch_sums_components():
    """Verifies overlapping components add up at the sampled height."""
    h    = np.array([0.0], dtype=np.float32)
    amps = np.array([[1.0, 3.0]], dtype=np.float32)
    mus  = np.array([[0.0, 0.0]], dtype=np.float32)
    sigs = np.array([[1.0, 1.0]], dtype=np.float32)

    pred = GaussianMixture.evaluate_batch(h, amps, mus, sigs)

    assert np.isclose(pred[0, 0], 4.0, atol=1e-5)


def test_evaluate_slice_matches_manual():
    """Verifies the single-height slice evaluation matches the analytic value per pixel."""
    params = np.zeros((6, 2, 2), dtype=np.float32)
    params[0] = 2.0
    params[1] = 1.0
    params[2] = 2.0

    out = GaussianMixture.evaluate_slice(params, h_val=1.0, n_gaussians=2)

    assert np.allclose(out, 2.0)


def test_evaluate_pixel_total_equals_sum_components():
    """Verifies the per-pixel total equals the sum of its component curves."""
    params = np.array([2.0, 0.0, 1.5, 1.0, 5.0, 2.0], dtype=np.float32)
    h      = np.linspace(-5.0, 10.0, 60)

    total, comps = GaussianMixture.evaluate_pixel(params, h, n_gaussians=2)

    assert len(comps) == 2
    assert np.allclose(total, comps[0] + comps[1])


def test_evaluate_pixel_peak_values():
    """Verifies the per-pixel curve peaks at the Gaussian amplitude."""
    params = np.array([3.0, 0.0, 1.0], dtype=np.float32)
    h      = np.linspace(-5.0, 5.0, 101)

    total, comps = GaussianMixture.evaluate_pixel(params, h, n_gaussians=1)

    assert np.isclose(total.max(), 3.0, atol=1e-4)


def test_reconstruct_batch_shape_and_dtype():
    """Verifies batch reconstruction returns float32 curves of shape (batch, heights)."""
    gauss = np.zeros((4, 2, 3), dtype=np.float32)
    gauss[:, :, 0] = 1.0
    gauss[:, :, 2] = 1.0
    x = np.linspace(-3.0, 3.0, 50).astype(np.float32)

    out = GaussianReconstructor.reconstruct_batch(gauss, x)

    assert out.shape == (4, 50)
    assert out.dtype == np.float32


def test_reconstruct_batch_matches_the_torch_evaluator():
    """Verifies the numpy batch reconstruction matches the torch curve evaluator."""
    rng   = np.random.default_rng(0)
    gauss = rng.uniform(-1.0, 3.0, size=(4, 2, 3)).astype(np.float32)
    x     = np.linspace(-3.0, 3.0, 32).astype(np.float32)

    numpy_curves = GaussianReconstructor.reconstruct_batch(gauss, x)
    torch_curves = GaussianCurve.reconstruct(
        torch.from_numpy(gauss.reshape(4, 6, 1, 1)),
        torch.from_numpy(x),
    )

    assert np.allclose(numpy_curves, torch_curves.numpy().reshape(4, 32), atol=1e-6)


def test_reconstructor_components_count():
    """Verifies the component list has one curve per Gaussian on the sampling axis."""
    params = np.array([1.0, 0.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32)
    x      = np.linspace(-5.0, 5.0, 20)

    comps = GaussianReconstructor.components(params, x, n_gaussians=2)

    assert len(comps) == 2
    assert all(c.shape == x.shape for c in comps)


def test_gaussian_curve_shape():
    """Verifies torch reconstruction returns curves of shape (batch, heights, rows, cols)."""
    params = torch.zeros((2, 6, 4, 5))
    params[:, 0] = 1.0
    params[:, 2] = 1.0
    params[:, 3] = 1.0
    params[:, 5] = 1.0
    x = torch.linspace(-3.0, 3.0, 16)

    curves = GaussianCurve.reconstruct(params, x)

    assert curves.shape == (2, 16, 4, 5)


def test_gaussian_curve_rejects_bad_channels():
    """Verifies a channel count that is not a multiple of three is rejected."""
    params = torch.zeros((1, 4, 2, 2))
    x      = torch.linspace(-1.0, 1.0, 8)

    with pytest.raises(ValueError):
        GaussianCurve.reconstruct(params, x)


def test_gaussian_curve_rejects_a_parameter_count_other_than_three():
    """Verifies torch reconstruction refuses any ppg other than amplitude, mean and sigma."""
    params = torch.zeros((1, 6, 2, 2))
    z      = torch.linspace(-1.0, 1.0, 8)

    with pytest.raises(ValueError, match="requires ppg=3"):
        GaussianCurve.reconstruct(params, z, ppg=2)


def test_gaussian_curve_peak_at_mu():
    """Verifies the reconstructed curve peaks at the Gaussian amplitude."""
    params = torch.zeros((1, 3, 1, 1))
    params[0, 0, 0, 0] = 4.0
    params[0, 1, 0, 0] = 0.0
    params[0, 2, 0, 0] = 1.0
    x = torch.linspace(-5.0, 5.0, 101)

    curves = GaussianCurve.reconstruct(params, x)

    assert torch.isclose(curves[0, :, 0, 0].max(), torch.tensor(4.0), atol=1e-3)


def test_gaussian_clamp_amplitude_bounds():
    """Verifies the clamp keeps the amplitude between zero and amp_max."""
    params = torch.zeros((1, 3, 1, 1))
    params[0, 0, 0, 0] = 100.0
    params[0, 2, 0, 0] = 5.0
    x = torch.linspace(0.0, 10.0, 11)

    clamped = GaussianClamp.apply(params, x, amp_max=2.0)

    assert clamped[0, 0, 0, 0].item() <= 2.0
    assert clamped[0, 0, 0, 0].item() >= 0.0


def test_gaussian_clamp_mean_bounds():
    """Verifies the clamp keeps the mean inside the height axis in metres."""
    params = torch.zeros((1, 3, 1, 1))
    params[0, 1, 0, 0] = 999.0
    x = torch.linspace(0.0, 10.0, 11)

    clamped = GaussianClamp.apply(params, x, amp_max=2.0)

    assert clamped[0, 1, 0, 0].item() <= 10.0
    assert clamped[0, 1, 0, 0].item() >= 0.0


def test_gaussian_clamp_sigma_floor():
    """Verifies the clamp lifts sigma to half the height sampling step."""
    params = torch.zeros((1, 3, 1, 1))
    params[0, 2, 0, 0] = -50.0
    x      = torch.linspace(0.0, 10.0, 11)
    x_step = (10.0 - 0.0) / 10.0

    clamped = GaussianClamp.apply(params, x, amp_max=2.0)

    assert clamped[0, 2, 0, 0].item() >= x_step * 0.5 - 1e-6


def test_gaussian_clamp_preserves_shape():
    """Verifies the clamp leaves the parameter tensor shape unchanged."""
    params = torch.randn(2, 9, 3, 3)
    x      = torch.linspace(0.0, 20.0, 16)

    clamped = GaussianClamp.apply(params, x, amp_max=1.25)

    assert clamped.shape == params.shape


def test_gaussian_curve_matches_mixture_batch():
    """Verifies the torch curve and the numpy mixture agree for the same parameters."""
    a, mu, sig = 2.0, 1.0, 1.5
    x_np   = np.linspace(-5.0, 8.0, 40).astype(np.float32)

    batch  = GaussianMixture.evaluate_batch(x_np, np.array([[a]], np.float32), np.array([[mu]], np.float32), np.array([[sig]], np.float32))

    params = torch.zeros((1, 3, 1, 1))
    params[0, 0, 0, 0] = a
    params[0, 1, 0, 0] = mu
    params[0, 2, 0, 0] = sig
    curve = GaussianCurve.reconstruct(params, torch.from_numpy(x_np))[0, :, 0, 0].numpy()

    assert np.allclose(batch[0], curve, atol=1e-4)


@pytest.mark.dataset
def test_evaluate_slice_on_scene_parameters(parameters, param_extraction_meta):
    """Verifies a scene parameter window evaluates to a finite non-negative z slice."""
    k_max  = param_extraction_meta["k_max"]
    params = np.asarray(parameters[:, :16, :16]).astype(np.float32)

    out = GaussianMixture.evaluate_slice(params, h_val=10.0, n_gaussians=k_max)

    assert out.shape == (16, 16)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


@pytest.mark.dataset
def test_evaluate_pixel_on_scene_parameters(parameters, param_extraction_meta):
    """Verifies a scene pixel's components sum to the finite total curve."""
    k_max  = param_extraction_meta["k_max"]
    h_min, h_max = param_extraction_meta["z_range"]
    h      = np.linspace(h_min, h_max, 100)
    params = np.asarray(parameters[:, 0, 0]).astype(np.float32)

    total, comps = GaussianMixture.evaluate_pixel(params, h, n_gaussians=k_max)

    assert len(comps)  == k_max
    assert total.shape == (100,)
    assert np.allclose(total, sum(comps))
    assert np.all(np.isfinite(total))


@pytest.mark.dataset
def test_scene_parameters_channel_count(parameters, param_extraction_meta):
    """Verifies the scene parameter map has three channels per extracted Gaussian."""
    assert parameters.shape[0] == param_extraction_meta["k_max"] * 3
