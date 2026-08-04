from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from pipelines.jepa.inference.embedding import JepaEmbeddingEvaluator
from pipelines.jepa.inference.pipeline  import JEPA_INFERENCE_COMPONENTS
from tools.data.gaussians               import GaussianReconstructor

from tests.conftest      import SilentLogger
from tests.jepa.conftest import PROFILE_LENGTH, make_autoencoder


class _FakeProfileAe:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def normalize_embedding(self, z):
        return z

    def encode(self, curve):
        return curve[:, : self.embedding_dim]

    def decode(self, z):
        pad = torch.zeros(z.shape[0], self.profile_length - z.shape[1], *z.shape[2:], dtype=z.dtype)
        return torch.cat([z, pad], dim=1)


class _FakeJepa:
    def __init__(self, autoencoder, z_out):
        self.profile_autoencoder = autoencoder
        self.z_out               = z_out

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, images):
        return self.z_out


class _IdentityNormalizer:
    def normalize(self, curve):
        return curve

    def denormalize_output(self, params):
        return params


def _build_case(z_offset: float = 0.0, flip: bool = False):
    torch.manual_seed(0)

    n_gaussians, n_elev, spatial, emb_dim = 2, 8, 3, 4
    x_axis = np.linspace(-4.0, 4.0, n_elev).astype(np.float32)
    x      = x_axis.reshape(1, 1, -1, 1, 1)

    gt_params           = torch.rand(2, n_gaussians * 3, spatial, spatial) + 0.1
    gt_params[:, 1::3]  = torch.rand(2, n_gaussians, spatial, spatial) * 6.0 - 3.0
    gt_params[:, 2::3] += 0.5

    gt_np     = gt_params.numpy().astype(np.float32)
    gt_gauss  = gt_np.reshape(2, n_gaussians, 3, spatial, spatial)
    gt_curves = torch.from_numpy(GaussianReconstructor.reconstruct_batch(gt_gauss, x).astype(np.float32))

    autoencoder                = _FakeProfileAe(emb_dim)
    autoencoder.profile_length = n_elev

    z_star = autoencoder.encode(gt_curves)
    z_out  = -z_star if flip else z_star + z_offset

    adapter = types.SimpleNamespace(
        jepa               = _FakeJepa(autoencoder, z_out),
        profile_normalizer = _IdentityNormalizer(),
    )

    run = types.SimpleNamespace(
        model       = types.SimpleNamespace(module=adapter, device="cpu"),
        loader      = [(torch.rand(2, 2, spatial, spatial), gt_params)],
        dataset     = types.SimpleNamespace(normalizer=_IdentityNormalizer()),
        n_gaussians = n_gaussians,
        x_axis      = x_axis,
    )

    return run, gt_curves, autoencoder


def _build_layernorm_case():
    torch.manual_seed(0)

    n_gaussians, spatial = 2, 3
    n_elev = PROFILE_LENGTH
    x_axis = np.linspace(-4.0, 4.0, n_elev).astype(np.float32)
    x      = x_axis.reshape(1, 1, -1, 1, 1)

    gt_params           = torch.rand(2, n_gaussians * 3, spatial, spatial) + 0.1
    gt_params[:, 1::3]  = torch.rand(2, n_gaussians, spatial, spatial) * 6.0 - 3.0
    gt_params[:, 2::3] += 0.5

    gt_gauss  = gt_params.numpy().astype(np.float32).reshape(2, n_gaussians, 3, spatial, spatial)
    gt_curves = torch.from_numpy(GaussianReconstructor.reconstruct_batch(gt_gauss, x).astype(np.float32))

    autoencoder = make_autoencoder("layernorm")
    autoencoder.eval()
    with torch.no_grad():
        autoencoder.embedding_layernorm.norm.weight.mul_(2.0).add_(0.3)
        autoencoder.embedding_layernorm.norm.bias.add_(0.5)
        z_out = autoencoder.encoder(gt_curves)

    adapter = types.SimpleNamespace(
        jepa               = _FakeJepa(autoencoder, z_out),
        profile_normalizer = _IdentityNormalizer(),
    )

    run = types.SimpleNamespace(
        model       = types.SimpleNamespace(module=adapter, device="cpu"),
        loader      = [(torch.rand(2, 2, spatial, spatial), gt_params)],
        dataset     = types.SimpleNamespace(normalizer=_IdentityNormalizer()),
        n_gaussians = n_gaussians,
        x_axis      = x_axis,
    )

    return run


def test_components_carry_embedding_evaluator():
    assert JEPA_INFERENCE_COMPONENTS.embedding_evaluator_cls is JepaEmbeddingEvaluator


def test_perfect_prediction_scores_zero_embedding_error():
    run, gt_curves, autoencoder = _build_case()

    metrics = JepaEmbeddingEvaluator(run, SilentLogger()).run()

    assert metrics["jepa_embedding_mse"]    == pytest.approx(0.0, abs=1e-10)
    assert metrics["jepa_embedding_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["jepa_chain_mse_norm"]   == pytest.approx(metrics["jepa_decode_mse_norm"], abs=1e-10)


def test_decoder_floor_matches_reconstruction_tail():
    run, gt_curves, autoencoder = _build_case()

    metrics = JepaEmbeddingEvaluator(run, SilentLogger()).run()

    tail     = gt_curves[:, autoencoder.embedding_dim:]
    expected = float((tail ** 2).sum()) / gt_curves.numel()

    assert metrics["jepa_decode_mse_norm"] == pytest.approx(expected, rel=1e-5)


def test_flipped_prediction_scores_negative_cosine():
    run, _, _ = _build_case(flip=True)

    metrics = JepaEmbeddingEvaluator(run, SilentLogger()).run()

    assert metrics["jepa_embedding_cosine"] == pytest.approx(-1.0, abs=1e-6)
    assert metrics["jepa_embedding_mse"] > 0.0
    assert metrics["jepa_chain_mse_norm"] > metrics["jepa_decode_mse_norm"]


def test_offset_prediction_separates_chain_from_decoder_error():
    run, _, _ = _build_case(z_offset=0.5)

    metrics = JepaEmbeddingEvaluator(run, SilentLogger()).run()

    assert metrics["jepa_embedding_mse"] == pytest.approx(0.25, rel=1e-5)
    assert metrics["jepa_chain_mse_norm"] > metrics["jepa_decode_mse_norm"]


def test_layernorm_target_is_normalized_exactly_once():
    run = _build_layernorm_case()

    metrics = JepaEmbeddingEvaluator(run, SilentLogger()).run()

    assert metrics["jepa_embedding_mse"]    == pytest.approx(0.0, abs=1e-10)
    assert metrics["jepa_embedding_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_empty_loader_raises():
    run, _, _ = _build_case()
    run.loader = []

    with pytest.raises(ValueError, match="no samples"):
        JepaEmbeddingEvaluator(run, SilentLogger()).run()
