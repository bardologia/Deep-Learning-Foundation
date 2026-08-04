from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.data.gaussians import (
    GaussianClamp,
    GaussianCurve,
    GaussianHead,
    GaussianMixture,
    GaussianReconstructor,
)


def test_head_total_channels():
    assert GaussianHead.total_channels(3, 4) == 12


def test_safe_sigma_sq_floor():
    out = GaussianMixture.safe_sigma_sq(np.array([0.0]))

    assert out[0] == 2.0 * GaussianMixture.SIGMA_FLOOR ** 2


def test_safe_sigma_sq_value():
    out = GaussianMixture.safe_sigma_sq(np.array([3.0]))

    assert np.isclose(out[0], 18.0)


def test_evaluate_batch_peak_at_mu():
    h    = np.linspace(-10.0, 10.0, 201).astype(np.float32)
    amps = np.array([[2.0]], dtype=np.float32)
    mus  = np.array([[0.0]], dtype=np.float32)
    sigs = np.array([[2.0]], dtype=np.float32)

    pred = GaussianMixture.evaluate_batch(h, amps, mus, sigs)

    assert pred.shape == (1, 201)
    assert np.isclose(pred[0].max(), 2.0, atol=1e-4)
    assert np.argmax(pred[0]) == 100


def test_evaluate_batch_sums_components():
    h    = np.array([0.0], dtype=np.float32)
    amps = np.array([[1.0, 3.0]], dtype=np.float32)
    mus  = np.array([[0.0, 0.0]], dtype=np.float32)
    sigs = np.array([[1.0, 1.0]], dtype=np.float32)

    pred = GaussianMixture.evaluate_batch(h, amps, mus, sigs)

    assert np.isclose(pred[0, 0], 4.0, atol=1e-5)


def test_evaluate_slice_matches_manual():
    params = np.zeros((6, 2, 2), dtype=np.float32)
    params[0] = 2.0
    params[1] = 1.0
    params[2] = 2.0

    out = GaussianMixture.evaluate_slice(params, h_val=1.0, n_gaussians=2)

    assert np.allclose(out, 2.0)


def test_evaluate_pixel_total_equals_sum_components():
    params = np.array([2.0, 0.0, 1.5, 1.0, 5.0, 2.0], dtype=np.float32)
    h      = np.linspace(-5.0, 10.0, 60)

    total, comps = GaussianMixture.evaluate_pixel(params, h, n_gaussians=2)

    assert len(comps) == 2
    assert np.allclose(total, comps[0] + comps[1])


def test_evaluate_pixel_peak_values():
    params = np.array([3.0, 0.0, 1.0], dtype=np.float32)
    h      = np.linspace(-5.0, 5.0, 101)

    total, comps = GaussianMixture.evaluate_pixel(params, h, n_gaussians=1)

    assert np.isclose(total.max(), 3.0, atol=1e-4)


def test_reconstruct_batch_shape_and_dtype():
    gauss = np.zeros((4, 2, 3), dtype=np.float32)
    gauss[:, :, 0] = 1.0
    gauss[:, :, 2] = 1.0
    x = np.linspace(-3.0, 3.0, 50).astype(np.float32)

    out = GaussianReconstructor.reconstruct_batch(gauss, x)

    assert out.shape == (4, 50)
    assert out.dtype == np.float32


def test_reconstruct_batch_matches_the_torch_evaluator():
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
    params = np.array([1.0, 0.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32)
    x      = np.linspace(-5.0, 5.0, 20)

    comps = GaussianReconstructor.components(params, x, n_gaussians=2)

    assert len(comps) == 2
    assert all(c.shape == x.shape for c in comps)


def test_gaussian_curve_shape():
    params = torch.zeros((2, 6, 4, 5))
    params[:, 0] = 1.0
    params[:, 2] = 1.0
    params[:, 3] = 1.0
    params[:, 5] = 1.0
    x = torch.linspace(-3.0, 3.0, 16)

    curves = GaussianCurve.reconstruct(params, x)

    assert curves.shape == (2, 16, 4, 5)


def test_gaussian_curve_rejects_bad_channels():
    params = torch.zeros((1, 4, 2, 2))
    x      = torch.linspace(-1.0, 1.0, 8)

    with pytest.raises(ValueError):
        GaussianCurve.reconstruct(params, x)


def test_gaussian_curve_peak_at_mu():
    params = torch.zeros((1, 3, 1, 1))
    params[0, 0, 0, 0] = 4.0
    params[0, 1, 0, 0] = 0.0
    params[0, 2, 0, 0] = 1.0
    x = torch.linspace(-5.0, 5.0, 101)

    curves = GaussianCurve.reconstruct(params, x)

    assert torch.isclose(curves[0, :, 0, 0].max(), torch.tensor(4.0), atol=1e-3)


def test_gaussian_clamp_amplitude_bounds():
    params = torch.zeros((1, 3, 1, 1))
    params[0, 0, 0, 0] = 100.0
    params[0, 2, 0, 0] = 5.0
    x = torch.linspace(0.0, 10.0, 11)

    clamped = GaussianClamp.apply(params, x, amp_max=2.0)

    assert clamped[0, 0, 0, 0].item() <= 2.0
    assert clamped[0, 0, 0, 0].item() >= 0.0


def test_gaussian_clamp_mean_bounds():
    params = torch.zeros((1, 3, 1, 1))
    params[0, 1, 0, 0] = 999.0
    x = torch.linspace(0.0, 10.0, 11)

    clamped = GaussianClamp.apply(params, x, amp_max=2.0)

    assert clamped[0, 1, 0, 0].item() <= 10.0
    assert clamped[0, 1, 0, 0].item() >= 0.0


def test_gaussian_clamp_sigma_floor():
    params = torch.zeros((1, 3, 1, 1))
    params[0, 2, 0, 0] = -50.0
    x      = torch.linspace(0.0, 10.0, 11)
    x_step = (10.0 - 0.0) / 10.0

    clamped = GaussianClamp.apply(params, x, amp_max=2.0)

    assert clamped[0, 2, 0, 0].item() >= x_step * 0.5 - 1e-6


def test_gaussian_clamp_preserves_shape():
    params = torch.randn(2, 9, 3, 3)
    x      = torch.linspace(0.0, 20.0, 16)

    clamped = GaussianClamp.apply(params, x, amp_max=1.25)

    assert clamped.shape == params.shape


def test_gaussian_curve_matches_mixture_batch():
    a, mu, sig = 2.0, 1.0, 1.5
    x_np   = np.linspace(-5.0, 8.0, 40).astype(np.float32)

    batch  = GaussianMixture.evaluate_batch(x_np, np.array([[a]], np.float32), np.array([[mu]], np.float32), np.array([[sig]], np.float32))

    params = torch.zeros((1, 3, 1, 1))
    params[0, 0, 0, 0] = a
    params[0, 1, 0, 0] = mu
    params[0, 2, 0, 0] = sig
    curve = GaussianCurve.reconstruct(params, torch.from_numpy(x_np))[0, :, 0, 0].numpy()

    assert np.allclose(batch[0], curve, atol=1e-4)


@pytest.mark.real_data
def test_evaluate_slice_on_real_parameters(parameters, param_extraction_meta):
    k_max  = param_extraction_meta["k_max"]
    params = np.asarray(parameters[:, :16, :16]).astype(np.float32)

    out = GaussianMixture.evaluate_slice(params, h_val=10.0, n_gaussians=k_max)

    assert out.shape == (16, 16)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


@pytest.mark.real_data
def test_evaluate_pixel_on_real_parameters(parameters, param_extraction_meta):
    k_max  = param_extraction_meta["k_max"]
    h_min, h_max = param_extraction_meta["height_range"]
    h      = np.linspace(h_min, h_max, 100)
    params = np.asarray(parameters[:, 0, 0]).astype(np.float32)

    total, comps = GaussianMixture.evaluate_pixel(params, h, n_gaussians=k_max)

    assert len(comps)  == k_max
    assert total.shape == (100,)
    assert np.allclose(total, sum(comps))
    assert np.all(np.isfinite(total))


@pytest.mark.real_data
def test_real_parameters_channel_count(parameters, param_extraction_meta):
    assert parameters.shape[0] == param_extraction_meta["k_max"] * 3
