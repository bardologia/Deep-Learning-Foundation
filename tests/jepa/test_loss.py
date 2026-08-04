from __future__ import annotations

import pytest
import torch

from configuration.training.jepa      import EmbeddingLossConfig
from pipelines.jepa.training.coupling import TargetProvider
from pipelines.jepa.training.loss     import Loss as EmbeddingLoss
from tools.data.gaussians             import GaussianCurve

from tests.jepa.conftest import EMBEDDING_DIM, N_GAUSSIANS, SPATIAL, make_autoencoder


def build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad"):
    provider = TargetProvider(target_kind)
    return EmbeddingLoss(
        autoencoder         = autoencoder,
        target_provider     = provider,
        embedding_cfg       = emb_cfg,
        x_axis              = x_axis,
        norm_stats          = norm_stats,
        params_per_gaussian = 3,
        profile_normalizer  = profile_normalizer,
    )


def random_inputs(requires_grad=True):
    z_hat = torch.randn(2, EMBEDDING_DIM, SPATIAL, SPATIAL, requires_grad=requires_grad)
    gt    = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)
    return z_hat, gt


def test_loss_returns_finite_scalar(autoencoder, x_axis, norm_stats, profile_normalizer, embedding_loss_cfg):
    loss      = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, embedding_loss_cfg)
    z_hat, gt = random_inputs()

    out = loss(z_hat, gt)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert out["total_loss"].ndim == 0
    assert torch.isfinite(out["total_loss"])
    assert "embedding_mse" in out["components"]


def test_loss_gradient_flows_to_predictor(autoencoder, x_axis, norm_stats, profile_normalizer, embedding_loss_cfg):
    loss      = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, embedding_loss_cfg)
    z_hat, gt = random_inputs()

    loss(z_hat, gt)["total_loss"].backward()

    assert z_hat.grad is not None
    assert torch.isfinite(z_hat.grad).all()


def test_loss_does_not_flow_to_stopgrad_target(autoencoder, x_axis, norm_stats, profile_normalizer, embedding_loss_cfg):
    loss      = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, embedding_loss_cfg, target_kind="stopgrad")
    z_hat, gt = random_inputs()

    loss(z_hat, gt)["total_loss"].backward()

    assert all(p.grad is None for p in autoencoder.encoder.parameters())


def test_loss_zero_on_matching_embeddings(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=1.0, use_curve_recon=False)
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")

    gt = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    with torch.no_grad():
        gt_phys    = norm_stats.denormalize_output(gt.float())
        gt_curve   = GaussianCurve.reconstruct(gt_phys, x_axis, 3)
        gt_curve_n = profile_normalizer.normalize(gt_curve)
        z_star     = autoencoder.encoder(gt_curve_n)

    out = loss(z_star, gt)

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-10)


def test_loss_increases_with_divergence(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=1.0, use_curve_recon=False)
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")

    gt = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    with torch.no_grad():
        gt_curve   = GaussianCurve.reconstruct(gt.float(), x_axis, 3)
        gt_curve_n = profile_normalizer.normalize(gt_curve)
        z_star     = autoencoder.encoder(gt_curve_n)

    near = loss(z_star + 0.01, gt)["total_loss"].item()
    far  = loss(z_star + 1.00, gt)["total_loss"].item()

    assert far > near


def test_loss_curve_recon_flows_to_decoder(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(
        use_embedding_mse    = False,
        weight_embedding_mse = 0.0,
        use_curve_recon      = True,
        weight_curve_recon   = 1.0,
        curve_kind           = "mse",
    )
    loss      = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")
    z_hat, gt = random_inputs()

    out = loss(z_hat, gt)

    assert "curve_recon" in out["components"]
    out["total_loss"].backward()

    decoder_has_grad = any(p.grad is not None for p in autoencoder.decoder.parameters())
    assert decoder_has_grad


def test_loss_total_scales_components_by_weight(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=3.0, use_curve_recon=False)
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")
    z_hat, gt   = random_inputs(requires_grad=False)

    out = loss(z_hat, gt)
    raw = out["components"]["embedding_mse"]

    assert out["total_loss"].item() == pytest.approx(3.0 * raw.item(), rel=1e-6)


def test_loss_cosine_term_zero_on_matching_embeddings(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=False, use_embedding_cosine=True, weight_embedding_cosine=1.0, use_curve_recon=False)
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")

    gt = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    with torch.no_grad():
        gt_curve   = GaussianCurve.reconstruct(gt.float(), x_axis, 3)
        gt_curve_n = profile_normalizer.normalize(gt_curve)
        z_star     = autoencoder.encoder(gt_curve_n)

    out = loss(z_star, gt)

    assert "embedding_cosine" in out["components"]
    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert loss(-z_star, gt)["total_loss"].item() > 1.0


def test_loss_smoothl1_term_zero_on_matching_embeddings(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=False, use_embedding_smoothl1=True, weight_embedding_smoothl1=1.0, smoothl1_beta=1.0, use_curve_recon=False)
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")

    gt = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    with torch.no_grad():
        gt_curve   = GaussianCurve.reconstruct(gt.float(), x_axis, 3)
        gt_curve_n = profile_normalizer.normalize(gt_curve)
        z_star     = autoencoder.encoder(gt_curve_n)

    out = loss(z_star, gt)

    assert "embedding_smoothl1" in out["components"]
    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-10)
    assert loss(z_star + 1.0, gt)["total_loss"].item() > 0.0


@pytest.mark.parametrize("kind", ["l1", "huber", "charbonnier"])
def test_loss_curve_recon_kinds_compute_finite_terms(kind, x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=False, use_curve_recon=True, weight_curve_recon=1.0, curve_kind=kind)
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")
    z_hat, gt   = random_inputs()

    out = loss(z_hat, gt)

    assert "curve_recon" in out["components"]
    assert torch.isfinite(out["total_loss"])

    out["total_loss"].backward()
    assert z_hat.grad is not None
    assert torch.isfinite(z_hat.grad).all()


def test_loss_unknown_curve_kind_raises(x_axis, norm_stats, profile_normalizer):
    autoencoder = make_autoencoder("none")
    emb_cfg     = EmbeddingLossConfig(use_embedding_mse=False, use_curve_recon=True, curve_kind="bogus")
    loss        = build_loss(autoencoder, x_axis, norm_stats, profile_normalizer, emb_cfg, target_kind="stopgrad")
    z_hat, gt   = random_inputs(requires_grad=False)

    with pytest.raises(ValueError):
        loss(z_hat, gt)
