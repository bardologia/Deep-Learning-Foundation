from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.normalization            import OutputClampConfig
from configuration.training.backbone        import default_curriculum
from configuration.training.general.loss    import LossConfig, ParamMatching
from pipelines.backbone.training.loss       import Loss
from pipelines.backbone.training.loss_terms import LOSS_TERMS, LossComponentCatalog
from tools.monitoring.logger                import NullLogger
from tools.monitoring.tracker               import NullTracker

from tests.backbone_training._helpers import build_loss, gaussian_config, identity_normalizer, log1p_normalizer, param_tensor, valid_param_tensor, x_axis_tensor, zscore_normalizer


def test_loss_terms_table_is_consistent():
    names = [term.name for term in LOSS_TERMS]

    assert len(names) == len(set(names))
    assert "param_l1"         in names
    assert "moments"          in names

    for term in LOSS_TERMS:
        assert term.use_flag.startswith("use_")
        assert term.weight_key.startswith("weight_")
        assert term.space in ("norm", "denorm")


def test_catalog_names_match_loss_terms():
    assert LossComponentCatalog.names() == tuple(term.name for term in LOSS_TERMS)


def test_catalog_standalone_enables_only_the_named_term():
    for term in LOSS_TERMS:
        cfg = LossComponentCatalog.standalone(term.name)

        assert getattr(cfg, term.use_flag)   is True
        assert getattr(cfg, term.weight_key) == 1.0

        other_flags_off = [not getattr(cfg, other.use_flag) for other in LOSS_TERMS if other.name != term.name]
        assert all(other_flags_off)


def test_catalog_rejects_unknown_component():
    with pytest.raises(KeyError):
        LossComponentCatalog.standalone("not_a_real_loss")


def test_catalog_standalone_inherits_shared_knobs_from_base():
    base = LossConfig(param_matching=ParamMatching.SORTED_GT, param_weights=(2.0, 3.0, 4.0), use_mse_curve=True, weight_mse_curve=5.0)
    cfg  = LossComponentCatalog.standalone("param_l1", base=base)

    assert cfg.use_param_l1    is True
    assert cfg.weight_param_l1 == 1.0
    assert cfg.param_matching  == ParamMatching.SORTED_GT
    assert cfg.param_weights   == (2.0, 3.0, 4.0)
    assert cfg.use_mse_curve   is False


def test_catalog_curriculum_is_disabled_with_matching_phases():
    curriculum = LossComponentCatalog.curriculum("moments")

    assert curriculum.enabled is False
    assert curriculum.warmup.use_moments   is True
    assert curriculum.complete.use_moments is True


def test_catalog_combined_enables_the_union_of_terms():
    cfg = LossComponentCatalog.combined(["param_l1", "moments", "l1_curve"])

    assert cfg.use_param_l1 is True
    assert cfg.use_moments  is True
    assert cfg.use_l1_curve is True

    other_flags_off = [not getattr(cfg, term.use_flag) for term in LOSS_TERMS if term.name not in ("param_l1", "moments", "l1_curve")]
    assert all(other_flags_off)


def test_probe_union_merges_stages_when_curriculum_enabled():
    curriculum = default_curriculum()
    curriculum.warmup.use_smoothness_tv    = True
    curriculum.warmup.weight_smoothness_tv = 0.7

    union = LossComponentCatalog.probe_union(curriculum)

    assert union.use_cosine_curve  is True
    assert union.use_smoothness_tv is True
    assert union.weight_smoothness_tv == pytest.approx(0.7)
    assert union.use_param_l1      is True


def test_probe_union_is_the_single_stage_when_curriculum_disabled():
    curriculum         = default_curriculum()
    curriculum.enabled = False
    curriculum.warmup.use_smoothness_tv = True

    union = LossComponentCatalog.probe_union(curriculum)

    assert union.use_smoothness_tv is False
    assert union.use_cosine_curve  is True


def test_catalog_combined_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        LossComponentCatalog.combined([])

    with pytest.raises(KeyError):
        LossComponentCatalog.combined(["param_l1", "not_a_real_loss"])


def test_catalog_combined_curriculum_mirrors_combined_in_both_phases():
    curriculum = LossComponentCatalog.combined_curriculum(["param_l1", "total_power_relerr"])

    assert curriculum.enabled is False
    assert curriculum.warmup.use_param_l1     is True
    assert curriculum.warmup.use_total_power  is True
    assert curriculum.complete.use_param_l1   is True
    assert curriculum.complete.use_total_power is True


def test_forward_returns_finite_scalar_total():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=0)
    gt   = param_tensor(2, 2, 6, 6, seed=1)

    out  = loss(pred, gt)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert out["total_loss"].ndim == 0
    assert torch.isfinite(out["total_loss"]).item()


def test_physical_errors_logged_without_active_param_term():
    cfg  = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 6, 6, seed=0).requires_grad_(True)

    out  = loss(pred, param_tensor(2, 2, 6, 6, seed=1))

    assert set(out["components"].keys()) == {"mse_curve"}
    assert set(out["physical"].keys())   == {"amp_mae", "mu_mae", "sigma_mae"}
    assert all(torch.isfinite(v).item() for v in out["physical"].values())

    out["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


def test_forward_output_dict_keys_for_active_term():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 5, 5, seed=2)
    gt   = param_tensor(2, 2, 5, 5, seed=3)

    out  = loss(pred, gt)

    assert "param_l1"     in out["components"]
    assert "param_l1/amp" in out["monitor"]
    assert "param_l1/mu"  in out["monitor"]


def test_gradient_flows_to_predictions():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=4).requires_grad_(True)
    gt   = param_tensor(2, 2, 6, 6, seed=5)

    loss(pred, gt)["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad.abs().sum().item() > 0.0


def test_amp_gradient_survives_below_physical_floor():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss = build_loss(n_gaussians=1, loss_cfg=cfg, norm_stats=log1p_normalizer(1))

    gt = loss.norm_stats.normalize_output(valid_param_tensor(1, 1, 4, 4, seed=40))

    pred       = torch.zeros(1, 3, 4, 4)
    pred[:, 0] = -8.0
    pred[:, 1] = 0.1
    pred[:, 2] = 0.2
    pred.requires_grad_(True)

    loss(pred, gt)["total_loss"].backward()

    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad[:, 0].abs().sum().item() > 1e-5


def test_identical_valid_params_give_zero_param_loss():
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = valid_param_tensor(2, 2, 6, 6, seed=6)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_prepare_clamps_prediction_so_invalid_identical_params_are_nonzero():
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = param_tensor(2, 2, 6, 6, seed=30)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() > 0.0


def test_prepare_skips_the_physical_clamp_when_clamp_output_is_disabled():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    norm = identity_normalizer(6)
    norm.stats.clamp = OutputClampConfig(enabled=False)

    loss   = build_loss(n_gaussians=2, loss_cfg=cfg, norm_stats=norm)
    params = param_tensor(2, 2, 6, 6, seed=30)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_prepare_reads_clamp_knobs_from_stats():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss = build_loss(n_gaussians=1, loss_cfg=cfg)

    loss.norm_stats.stats.clamp.amp_max           = 2.0
    loss.norm_stats.stats.clamp.leaky_slope       = 0.0
    loss.norm_stats.stats.clamp.param_leaky_slope = 0.0

    huge = torch.full((1, 3, 2, 2), 50.0)
    _, pred_phys, _, _, _ = loss._prepare(huge, torch.zeros_like(huge))

    assert pred_phys[:, 0].max().item() <= 2.0


def test_prepare_param_clamp_slope_acts_independently():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss = build_loss(n_gaussians=1, loss_cfg=cfg)

    loss.norm_stats.stats.clamp.amp_max           = 2.0
    loss.norm_stats.stats.clamp.leaky_slope       = 0.0
    loss.norm_stats.stats.clamp.param_leaky_slope = 0.5

    raw = torch.zeros((1, 3, 2, 2))
    raw[:, 0] = 50.0
    _, pred_phys, _, _, _ = loss._prepare(raw, torch.zeros_like(raw))

    assert pred_phys[0, 0, 0, 0].item() == pytest.approx(2.0 + 0.5 * (50.0 - 2.0), abs=1e-4)


def test_nonlog_normalization_arm_keeps_loss_finite_for_wild_predictions():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg, norm_stats=zscore_normalizer(2))

    gt   = loss.norm_stats.normalize_output(valid_param_tensor(1, 2, 4, 4, seed=41))
    pred = (param_tensor(1, 2, 4, 4, seed=42) * 1e18).requires_grad_(True)

    out = loss(pred, gt)

    assert torch.isfinite(out["total_loss"]).item()

    out["total_loss"].backward()

    assert torch.isfinite(pred.grad).all().item()


def test_nonlog_normalization_arm_clamps_amp_before_curve_reconstruction():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss = build_loss(n_gaussians=1, loss_cfg=cfg, norm_stats=zscore_normalizer(1))

    raw       = torch.zeros((1, 3, 2, 2))
    raw[:, 0] = 1e18

    _, pred_phys, _, pred_curves, _ = loss._prepare(raw, torch.zeros_like(raw))

    assert torch.isfinite(pred_phys).all()
    assert torch.isfinite(pred_curves).all()
    assert pred_phys[:, 0].max().item() < 1e6


def test_single_term_total_equals_component_after_normalisation():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=2.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 6, 6, seed=7)
    gt   = param_tensor(2, 2, 6, 6, seed=8)

    out  = loss(pred, gt)

    assert out["total_loss"].item() == pytest.approx(out["components"]["param_l1"].item(), rel=1e-5)


def test_curve_term_wiring_mse():
    cfg  = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=9)
    gt   = param_tensor(2, 2, 5, 5, seed=10)

    out  = loss(pred, gt)

    assert "mse_curve" in out["components"]
    assert torch.isfinite(out["total_loss"]).item()


def test_log_all_losses_populates_monitor():
    loss = build_loss(n_gaussians=2, log_all_losses=True)
    pred = param_tensor(2, 2, 5, 5, seed=11)
    gt   = param_tensor(2, 2, 5, 5, seed=12)

    out  = loss(pred, gt)

    monitor_terms = {key.rsplit("_", 1)[0] for key in out["monitor"]}

    assert "mse_curve"   in monitor_terms
    assert "param_huber" in monitor_terms
    assert "param_l1/amp" in out["monitor"]
    assert len(out["monitor"]) > len(out["components"])


def test_slot_presence_knobs_log_occupancy():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True, active_weight=2.0, inactive_weight=0.5)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = valid_param_tensor(2, 2, 5, 5, seed=21)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=22)

    out  = loss(pred, gt)

    assert "gt_active_frac"    in out["occupancy"]
    assert "pred_active_frac"  in out["occupancy"]
    assert "pred_active_slot0" in out["occupancy"]
    assert "pred_active_slot1" in out["occupancy"]
    assert "gt_active_slot0"   in out["occupancy"]
    assert out["occupancy"]["gt_active_frac"].item() == pytest.approx(1.0)


def test_occupancy_absent_when_no_slot_presence_knobs():
    loss = build_loss(n_gaussians=2)
    pred = valid_param_tensor(2, 2, 5, 5, seed=23)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=24)

    out  = loss(pred, gt)

    assert out["occupancy"] == {}


def test_count_metrics_exact_when_pred_matches_gt():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=31)

    out  = loss(gt.clone(), gt)

    assert out["occupancy"]["count/exact_frac"].item() == pytest.approx(1.0)
    assert out["occupancy"]["count/under_frac"].item() == pytest.approx(0.0)
    assert out["occupancy"]["count/over_frac"].item()  == pytest.approx(0.0)
    assert out["occupancy"]["count/acc_gt2"].item()    == pytest.approx(1.0)
    assert "count/acc_gt1" not in out["occupancy"]


def test_count_metrics_under_when_pred_drops_a_slot():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=32)
    pred = gt.clone()

    pred[:, 3] = 0.0

    out  = loss(pred, gt)

    assert out["occupancy"]["count/under_frac"].item() == pytest.approx(1.0)
    assert out["occupancy"]["count/exact_frac"].item() == pytest.approx(0.0)
    assert out["occupancy"]["count/over_frac"].item()  == pytest.approx(0.0)
    assert out["occupancy"]["count/acc_gt2"].item()    == pytest.approx(0.0)


def test_curriculum_swap_changes_active_terms():
    loss = build_loss(n_gaussians=2)

    before = loss(param_tensor(2, 2, 5, 5, seed=13), param_tensor(2, 2, 5, 5, seed=14))
    assert "param_l1" in before["components"]

    swap = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss.set_curriculum(swap)

    assert loss.loss_generation == 1

    after = loss(param_tensor(2, 2, 5, 5, seed=15), param_tensor(2, 2, 5, 5, seed=16))

    assert "mse_curve" in after["components"]
    assert "param_l1"  not in after["components"]


def test_total_loss_is_weighted_normalised_mean():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=21)
    gt   = param_tensor(2, 2, 5, 5, seed=22)

    out  = loss(pred, gt)

    weight_sum = cfg.weight_param_l1 + cfg.weight_mse_curve
    summed     = cfg.weight_param_l1 * out["components"]["param_l1"].item() + cfg.weight_mse_curve * out["components"]["mse_curve"].item()
    expected   = summed / weight_sum

    assert out["total_loss"].item() == pytest.approx(expected, rel=1e-4)


def test_param_weights_length_mismatch_raises():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0,))
    loss = build_loss(n_gaussians=3, loss_cfg=cfg)
    pred = param_tensor(2, 3, 5, 5, seed=23)
    gt   = param_tensor(2, 3, 5, 5, seed=24)

    with pytest.raises(ValueError, match="param_weights"):
        loss(pred, gt)


@pytest.mark.dataset
def test_loss_on_scene_param_target(parameters):
    win         = np.asarray(parameters[:15, :8, :8]).astype(np.float32)
    n_channels  = win.shape[0]
    n_gaussians = n_channels // 3

    gt   = torch.from_numpy(win)[None]
    pred = gt.clone() + 0.05 * torch.randn_like(gt)

    loss = Loss(
        z_axis       = x_axis_tensor(),
        logger       = NullLogger(),
        tracker      = NullTracker(),
        gaussian_cfg = gaussian_config(n_gaussians),
        loss_cfg     = LossConfig(use_param_l1=True, weight_param_l1=1.0),
        norm_stats   = identity_normalizer(n_channels),
    )

    noisy = loss(pred, gt)
    clean = loss(gt, gt.clone())

    assert torch.isfinite(noisy["total_loss"]).item()
    assert torch.isfinite(clean["total_loss"]).item()
    assert noisy["total_loss"].item() > clean["total_loss"].item()


def test_gradient_flows_with_log_all_losses_and_shared_matching():
    loss = build_loss(n_gaussians=2, log_all_losses=True)
    pred = param_tensor(2, 2, 6, 6, seed=31).requires_grad_(True)
    gt   = param_tensor(2, 2, 6, 6, seed=32)

    loss(pred, gt)["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad.abs().sum().item() > 0.0
