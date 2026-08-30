"""Tests for the backbone training loss and its component catalog.

Covers the loss-term table, the catalog helpers that build standalone, combined and
curriculum loss configurations, the forward pass output contract, gradient flow, the
physical clamp applied before curve reconstruction, occupancy and count diagnostics, the
geometry-coupled physics terms and the legacy two-Gaussian parameter loss.
"""

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
    """Verifies loss term names are unique and each carries a use flag, weight key and space."""
    names = [term.name for term in LOSS_TERMS]

    assert len(names) == len(set(names))
    assert "param_l1"         in names
    assert "moments"          in names

    for term in LOSS_TERMS:
        assert term.use_flag.startswith("use_")
        assert term.weight_key.startswith("weight_")
        assert term.space in ("norm", "denorm")


def test_catalog_names_match_loss_terms():
    """Verifies the catalog name tuple mirrors the loss-term table order."""
    assert LossComponentCatalog.names() == tuple(term.name for term in LOSS_TERMS)


def test_catalog_standalone_enables_only_the_named_term():
    """Verifies a standalone config enables the named term at unit weight and nothing else."""
    for term in LOSS_TERMS:
        cfg = LossComponentCatalog.standalone(term.name)

        assert getattr(cfg, term.use_flag)   is True
        assert getattr(cfg, term.weight_key) == 1.0

        other_flags_off = [not getattr(cfg, other.use_flag) for other in LOSS_TERMS if other.name != term.name]
        assert all(other_flags_off)


def test_catalog_rejects_unknown_component():
    """Verifies requesting a standalone config for an unknown term raises KeyError."""
    with pytest.raises(KeyError):
        LossComponentCatalog.standalone("not_a_real_loss")


def test_catalog_standalone_inherits_shared_knobs_from_base():
    """Verifies matching mode and parameter weights carry over from the base config while other terms are switched off."""
    base = LossConfig(param_matching=ParamMatching.SORTED_GT, param_weights=(2.0, 3.0, 4.0), use_mse_curve=True, weight_mse_curve=5.0)
    cfg  = LossComponentCatalog.standalone("param_l1", base=base)

    assert cfg.use_param_l1    is True
    assert cfg.weight_param_l1 == 1.0
    assert cfg.param_matching  == ParamMatching.SORTED_GT
    assert cfg.param_weights   == (2.0, 3.0, 4.0)
    assert cfg.use_mse_curve   is False


def test_catalog_curriculum_is_disabled_with_matching_phases():
    """Verifies a single-term curriculum is disabled with identical warmup and complete stages."""
    curriculum = LossComponentCatalog.curriculum("moments")

    assert curriculum.enabled is False
    assert curriculum.warmup.use_moments   is True
    assert curriculum.complete.use_moments is True


def test_catalog_combined_enables_the_union_of_terms():
    """Verifies a combined config enables exactly the requested terms."""
    cfg = LossComponentCatalog.combined(["param_l1", "moments", "l1_curve"])

    assert cfg.use_param_l1 is True
    assert cfg.use_moments  is True
    assert cfg.use_l1_curve is True

    other_flags_off = [not getattr(cfg, term.use_flag) for term in LOSS_TERMS if term.name not in ("param_l1", "moments", "l1_curve")]
    assert all(other_flags_off)


def test_probe_union_merges_stages_when_curriculum_enabled():
    """Verifies the probe union takes the union of warmup and complete terms with their weights."""
    curriculum = default_curriculum()
    curriculum.warmup.use_smoothness_tv    = True
    curriculum.warmup.weight_smoothness_tv = 0.7

    union = LossComponentCatalog.probe_union(curriculum)

    assert union.use_cosine_curve  is True
    assert union.use_smoothness_tv is True
    assert union.weight_smoothness_tv == pytest.approx(0.7)
    assert union.use_param_l1      is True


def test_probe_union_is_the_single_stage_when_curriculum_disabled():
    """Verifies the probe union ignores the warmup stage when the curriculum is disabled."""
    curriculum         = default_curriculum()
    curriculum.enabled = False
    curriculum.warmup.use_smoothness_tv = True

    union = LossComponentCatalog.probe_union(curriculum)

    assert union.use_smoothness_tv is False
    assert union.use_cosine_curve  is True


def test_catalog_combined_rejects_empty_and_unknown():
    """Verifies an empty term list and an unknown term name are both rejected."""
    with pytest.raises(ValueError):
        LossComponentCatalog.combined([])

    with pytest.raises(KeyError):
        LossComponentCatalog.combined(["param_l1", "not_a_real_loss"])


def test_catalog_combined_curriculum_mirrors_combined_in_both_phases():
    """Verifies a combined curriculum enables the same terms in both stages with the curriculum off."""
    curriculum = LossComponentCatalog.combined_curriculum(["param_l1", "total_power_relerr"])

    assert curriculum.enabled is False
    assert curriculum.warmup.use_param_l1     is True
    assert curriculum.warmup.use_total_power  is True
    assert curriculum.complete.use_param_l1   is True
    assert curriculum.complete.use_total_power is True


def test_forward_returns_finite_scalar_total():
    """Verifies the forward pass returns the five output groups with a finite scalar total."""
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=0)
    gt   = param_tensor(2, 2, 6, 6, seed=1)

    out  = loss(pred, gt)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert out["total_loss"].ndim == 0
    assert torch.isfinite(out["total_loss"]).item()


def test_physical_errors_logged_without_active_param_term():
    """Verifies physical amplitude, mu and sigma errors are reported and gradients flow with only a curve term active."""
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
    """Verifies an active parameter term populates both the component and the per-parameter monitor entries."""
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 5, 5, seed=2)
    gt   = param_tensor(2, 2, 5, 5, seed=3)

    out  = loss(pred, gt)

    assert "param_l1"     in out["components"]
    assert "param_l1/amp" in out["monitor"]
    assert "param_l1/mu"  in out["monitor"]


def test_gradient_flows_to_predictions():
    """Verifies the total loss backpropagates finite non-zero gradients into the prediction tensor."""
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=4).requires_grad_(True)
    gt   = param_tensor(2, 2, 6, 6, seed=5)

    loss(pred, gt)["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad.abs().sum().item() > 0.0


def test_amp_gradient_survives_below_physical_floor():
    """Verifies amplitude gradients remain non-zero for predictions clamped below the physical floor."""
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
    """Verifies in-range identical predictions and targets give a zero parameter loss."""
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = valid_param_tensor(2, 2, 6, 6, seed=6)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_prepare_clamps_prediction_so_invalid_identical_params_are_nonzero():
    """Verifies clamping out-of-range predictions makes identical prediction and target pairs non-zero."""
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = param_tensor(2, 2, 6, 6, seed=30)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() > 0.0


def test_prepare_skips_the_physical_clamp_when_clamp_output_is_disabled():
    """Verifies disabling the output clamp restores a zero loss for identical out-of-range tensors."""
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    norm = identity_normalizer(6)
    norm.stats.clamp = OutputClampConfig(enabled=False)

    loss   = build_loss(n_gaussians=2, loss_cfg=cfg, norm_stats=norm)
    params = param_tensor(2, 2, 6, 6, seed=30)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_prepare_reads_clamp_knobs_from_stats():
    """Verifies the amplitude ceiling from the normalizer stats caps the physical prediction."""
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss = build_loss(n_gaussians=1, loss_cfg=cfg)

    loss.norm_stats.stats.clamp.amp_max           = 2.0
    loss.norm_stats.stats.clamp.leaky_slope       = 0.0
    loss.norm_stats.stats.clamp.param_leaky_slope = 0.0

    huge = torch.full((1, 3, 2, 2), 50.0)
    _, pred_phys, _, _, _ = loss._prepare(huge, torch.zeros_like(huge))

    assert pred_phys[:, 0].max().item() <= 2.0


def test_prepare_param_clamp_slope_acts_independently():
    """Verifies the parameter-space leaky slope applies above the amplitude ceiling independently of the curve slope."""
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
    """Verifies the z-score normalization arm keeps loss and gradients finite for 1e18-scale predictions."""
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg, norm_stats=zscore_normalizer(2))

    gt   = loss.norm_stats.normalize_output(valid_param_tensor(1, 2, 4, 4, seed=41))
    pred = (param_tensor(1, 2, 4, 4, seed=42) * 1e18).requires_grad_(True)

    out = loss(pred, gt)

    assert torch.isfinite(out["total_loss"]).item()

    out["total_loss"].backward()

    assert torch.isfinite(pred.grad).all().item()


def test_nonlog_normalization_arm_clamps_amp_before_curve_reconstruction():
    """Verifies amplitudes are clamped before curve synthesis in the z-score normalization arm."""
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss = build_loss(n_gaussians=1, loss_cfg=cfg, norm_stats=zscore_normalizer(1))

    raw       = torch.zeros((1, 3, 2, 2))
    raw[:, 0] = 1e18

    _, pred_phys, _, pred_curves, _ = loss._prepare(raw, torch.zeros_like(raw))

    assert torch.isfinite(pred_phys).all()
    assert torch.isfinite(pred_curves).all()
    assert pred_phys[:, 0].max().item() < 1e6


def test_single_term_total_equals_component_after_normalisation():
    """Verifies the weight-normalised total equals the single active component value."""
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=2.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 6, 6, seed=7)
    gt   = param_tensor(2, 2, 6, 6, seed=8)

    out  = loss(pred, gt)

    assert out["total_loss"].item() == pytest.approx(out["components"]["param_l1"].item(), rel=1e-5)


def test_curve_term_wiring_mse():
    """Verifies the MSE curve term is wired into the components with a finite total."""
    cfg  = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=9)
    gt   = param_tensor(2, 2, 5, 5, seed=10)

    out  = loss(pred, gt)

    assert "mse_curve" in out["components"]
    assert torch.isfinite(out["total_loss"]).item()


def test_log_all_losses_populates_monitor():
    """Verifies log_all_losses reports inactive terms in the monitor without adding components."""
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
    """Verifies presence balancing reports aggregate and per-slot occupancy fractions."""
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
    """Verifies occupancy stays empty when no slot-presence knob is enabled."""
    loss = build_loss(n_gaussians=2)
    pred = valid_param_tensor(2, 2, 5, 5, seed=23)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=24)

    out  = loss(pred, gt)

    assert out["occupancy"] == {}


def test_count_metrics_exact_when_pred_matches_gt():
    """Verifies count metrics report a perfect exact fraction when the prediction equals the target."""
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
    """Verifies zeroing a predicted slot amplitude is reported as an under-count."""
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
    """Verifies swapping the curriculum config bumps the loss generation and replaces the active components."""
    loss = build_loss(n_gaussians=2)

    before = loss(param_tensor(2, 2, 5, 5, seed=13), param_tensor(2, 2, 5, 5, seed=14))
    assert "param_l1" in before["components"]

    swap = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss.set_curriculum(swap)

    assert loss.loss_generation == 1

    after = loss(param_tensor(2, 2, 5, 5, seed=15), param_tensor(2, 2, 5, 5, seed=16))

    assert "mse_curve" in after["components"]
    assert "param_l1"  not in after["components"]


def test_geometry_coupled_covariance_term_uses_geometry():
    """Verifies the covariance term builds channel-sized geometry tensors and backpropagates finite gradients."""
    cfg  = LossConfig(use_moments=True, weight_covariance_match=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)

    assert loss.geometry.outer.shape[0]    == loss.geometry.n_tracks
    assert loss.geometry.steering.shape[0] == loss.geometry.n_tracks

    pred = param_tensor(2, 2, 5, 5, seed=17).requires_grad_(True)
    gt   = param_tensor(2, 2, 5, 5, seed=18)

    out  = loss(pred, gt)

    assert "moments" in out["components"]
    out["total_loss"].backward()
    assert torch.isfinite(pred.grad).all().item()


def test_geometry_coupled_coherence_resyn_term():
    """Verifies the coherence resynthesis term produces a non-negative component."""
    cfg  = LossConfig(use_coherence_resyn=True, weight_coherence_resyn=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=19)
    gt   = param_tensor(2, 2, 5, 5, seed=20)

    out  = loss(pred, gt)

    assert "coherence_resyn" in out["components"]
    assert out["total_loss"].item() >= 0.0


def test_total_loss_is_weighted_normalised_mean():
    """Verifies the total equals the weight-normalised mean of the active components."""
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
    """Verifies parameter weights of the wrong length raise during the forward pass."""
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0,))
    loss = build_loss(n_gaussians=3, loss_cfg=cfg)
    pred = param_tensor(2, 3, 5, 5, seed=23)
    gt   = param_tensor(2, 3, 5, 5, seed=24)

    with pytest.raises(ValueError, match="param_weights"):
        loss(pred, gt)


@pytest.mark.dataset
def test_loss_on_scene_param_target(parameters):
    """Verifies a noised prediction scores worse than an exact one on a scene parameter window."""
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
    """Verifies gradients still flow when every term is monitored through the shared matching."""
    loss = build_loss(n_gaussians=2, log_all_losses=True)
    pred = param_tensor(2, 2, 6, 6, seed=31).requires_grad_(True)
    gt   = param_tensor(2, 2, 6, 6, seed=32)

    loss(pred, gt)["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad.abs().sum().item() > 0.0


ZERO_FLOOR_MIN = (0.0, 0.0, 2.0, 0.0, 0.0, 2.0)
ZERO_FLOOR_MAX = (4.0, 80.0, 10.0, 4.0, 80.0, 10.0)


def _legacy_cfg(**overrides) -> LossConfig:
    """Returns a legacy-parameter loss config using sorted-GT matching."""
    return LossConfig(use_param_legacy=True, weight_param_legacy=1.0, param_matching=ParamMatching.SORTED_GT, **overrides)


def _legacy_zero_floor_cfg() -> LossConfig:
    """Returns a legacy loss config whose scaling bounds start at zero."""
    return _legacy_cfg(legacy_bounds_min=ZERO_FLOOR_MIN, legacy_bounds_max=ZERO_FLOOR_MAX)


def _legacy_mask_zero_floor_cfg() -> LossConfig:
    """Returns a masked legacy loss config whose scaling bounds start at zero."""
    return _legacy_cfg(legacy_mask=True, legacy_bounds_min=ZERO_FLOOR_MIN, legacy_bounds_max=ZERO_FLOOR_MAX)


def _legacy_pred(slot2_amp_px0: float, slot2_amp_px1: float) -> torch.Tensor:
    """Returns a (1, 6, 1, 2) prediction with fixed slot-1 parameters and the given slot-2 amplitudes."""
    pred = torch.zeros(1, 6, 1, 2)

    pred[0, :, 0, 0] = torch.tensor([1.0, 10.0, 5.0, slot2_amp_px0, 30.0, 6.0])
    pred[0, :, 0, 1] = torch.tensor([3.0, 20.0, 7.0, slot2_amp_px1, 40.0, 8.0])

    return pred


def _legacy_gt(slot2: tuple = (1.0, 44.0, 6.0)) -> torch.Tensor:
    """Returns a (1, 6, 1, 2) target with constant slot-1 parameters and the given slot-2 triplet."""
    gt = torch.zeros(1, 6, 1, 2)

    gt[0, 0:3] = torch.tensor([2.0, 16.0, 4.0]).reshape(3, 1, 1)
    gt[0, 3:6] = torch.tensor(slot2).reshape(3, 1, 1)

    return gt


def test_legacy_term_matches_hand_computed_scaled_mse():
    """Verifies the legacy term reproduces the hand-computed MSE in bound-scaled space."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_zero_floor_cfg())
    pred = _legacy_pred(0.0, 2.0)
    gt   = _legacy_gt()

    out  = loss(pred, gt)

    assert set(out["components"].keys()) == {"param_legacy"}
    assert out["total_loss"].item() == pytest.approx(0.0425, rel=1e-4)


def test_legacy_supervises_gt_slot2_when_prediction_absent():
    """Verifies the legacy term still responds to the target slot-2 values when the prediction is absent."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_zero_floor_cfg())

    out_a = loss(_legacy_pred(0.0, 0.0), _legacy_gt((1.0, 44.0, 6.0)))
    out_b = loss(_legacy_pred(0.0, 0.0), _legacy_gt((1.5, 60.0, 9.0)))

    assert out_a["total_loss"].item() != pytest.approx(out_b["total_loss"].item(), rel=1e-6)


def test_legacy_predicted_absent_slot2_receives_gradient():
    """Verifies both slots receive gradients when the predicted second slot is empty."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_zero_floor_cfg())
    pred = _legacy_pred(0.0, 0.0).requires_grad_(True)

    loss(pred, _legacy_gt())["total_loss"].backward()

    assert pred.grad[:, 3:6].abs().sum().item() > 0.0
    assert pred.grad[:, 0:3].abs().sum().item() > 0.0
    assert torch.isfinite(pred.grad).all().item()


def test_legacy_mask_matches_hand_computed_group_means():
    """Verifies the masked legacy term reproduces the hand-computed group-mean total."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_mask_zero_floor_cfg())
    pred = _legacy_pred(0.0, 2.0)
    gt   = _legacy_gt()

    out  = loss(pred, gt)

    assert set(out["components"].keys()) == {"param_legacy"}
    assert out["total_loss"].item() == pytest.approx(0.416875, rel=1e-4)


def test_legacy_mask_ignores_gt_slot2_when_prediction_absent():
    """Verifies the masked legacy term is blind to the target slot-2 values when the prediction is absent."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_mask_zero_floor_cfg())

    out_a = loss(_legacy_pred(0.0, 0.0), _legacy_gt((1.0, 44.0, 6.0)))
    out_b = loss(_legacy_pred(0.0, 0.0), _legacy_gt((1.5, 60.0, 9.0)))

    assert out_a["total_loss"].item() == pytest.approx(out_b["total_loss"].item(), rel=1e-6)


def test_legacy_mask_predicted_absent_slot2_receives_no_gradient():
    """Verifies the masked legacy term gives an empty predicted second slot no gradient."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_mask_zero_floor_cfg())
    pred = _legacy_pred(0.0, 0.0).requires_grad_(True)

    loss(pred, _legacy_gt())["total_loss"].backward()

    assert torch.all(pred.grad[:, 3:6] == 0.0).item()
    assert pred.grad[:, 0:3].abs().sum().item() > 0.0


def test_legacy_mask_predicted_present_slot2_receives_gradient():
    """Verifies the masked legacy term gives a present predicted second slot a gradient."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_mask_zero_floor_cfg())
    pred = _legacy_pred(2.0, 2.0).requires_grad_(True)

    loss(pred, _legacy_gt())["total_loss"].backward()

    assert pred.grad[:, 3:6].abs().sum().item() > 0.0
    assert torch.isfinite(pred.grad).all().item()


def test_legacy_mask_negative_scaled_width_masks_slot2_despite_present_amplitude():
    """Verifies the width gate masks slot 2 even when the predicted amplitude channels the threshold."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_mask_zero_floor_cfg())

    pred = _legacy_pred(2.0, 2.0)
    pred[0, 5] = 1.0

    out_a = loss(pred, _legacy_gt((1.0, 44.0, 6.0)))
    out_b = loss(pred.clone(), _legacy_gt((1.5, 60.0, 9.0)))

    assert out_a["total_loss"].item() == pytest.approx(out_b["total_loss"].item(), rel=1e-6)


def test_legacy_mask_negative_amp_floor_keeps_the_gate_on_at_zero_amplitude():
    """Verifies a negative amplitude floor scales physical zero above the gate, supervising slot 2 everywhere."""
    bounds_min = (-0.2, 0.0, 2.0, -0.2, 0.0, 2.0)
    bounds_max = (1.0, 80.0, 10.0, 1.0, 80.0, 10.0)
    loss       = build_loss(n_gaussians=2, loss_cfg=_legacy_cfg(legacy_mask=True, legacy_bounds_min=bounds_min, legacy_bounds_max=bounds_max))

    out_a = loss(_legacy_pred(0.0, 0.0), _legacy_gt((1.0, 44.0, 6.0)))
    out_b = loss(_legacy_pred(0.0, 0.0), _legacy_gt((1.5, 60.0, 9.0)))

    assert out_a["total_loss"].item() != pytest.approx(out_b["total_loss"].item(), rel=1e-6)


def test_legacy_bounds_length_mismatch_raises():
    """Verifies mismatched legacy bound tuple lengths raise during the forward chan."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_cfg(legacy_bounds_min=(0.0, 0.0, 2.0), legacy_bounds_max=ZERO_FLOOR_MAX))

    with pytest.raises(ValueError):
        loss(_legacy_pred(0.0, 2.0), _legacy_gt())


def test_legacy_inverted_bounds_raise():
    """Verifies legacy bounds with the minimum above the maximum raise during the forward chan."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_cfg(legacy_bounds_min=ZERO_FLOOR_MAX, legacy_bounds_max=ZERO_FLOOR_MIN))

    with pytest.raises(ValueError):
        loss(_legacy_pred(0.0, 2.0), _legacy_gt())


def test_legacy_requires_sorted_gt_matching():
    """Verifies the legacy term refuses to build under Hungarian matching."""
    cfg = LossConfig(use_param_legacy=True, weight_param_legacy=1.0)

    with pytest.raises(ValueError):
        build_loss(n_gaussians=2, loss_cfg=cfg)


def test_legacy_matching_validated_at_curriculum_swap():
    """Verifies the matching requirement is re-checked when the curriculum swaps in the legacy term."""
    loss = build_loss(n_gaussians=2)

    with pytest.raises(ValueError):
        loss.set_curriculum(LossConfig(use_param_legacy=True, weight_param_legacy=1.0))


def test_legacy_rejects_non_two_gaussian_runs():
    """Verifies the legacy term refuses runs with a Gaussian count other than two."""
    loss = build_loss(n_gaussians=3, loss_cfg=_legacy_cfg())

    with pytest.raises(ValueError):
        loss(param_tensor(2, 3, 4, 4, seed=50), param_tensor(2, 3, 4, 4, seed=51))


def test_legacy_monitored_only_on_two_gaussian_runs():
    """Verifies the legacy monitor entry appears only for two-Gaussian runs."""
    two   = build_loss(n_gaussians=2, log_all_losses=True)
    three = build_loss(n_gaussians=3, log_all_losses=True)

    out_two   = two(param_tensor(2, 2, 5, 5, seed=52), param_tensor(2, 2, 5, 5, seed=53))
    out_three = three(param_tensor(2, 3, 5, 5, seed=54), param_tensor(2, 3, 5, 5, seed=55))

    assert "param_legacy_denorm" in out_two["monitor"]
    assert "param_legacy_denorm" not in out_three["monitor"]


def test_legacy_loss_is_plain_mse_in_scaled_space():
    """Verifies LegacyParamLoss.mse equals the MSE of the min-max scaled parameters."""
    from tools.loss.param_loss import LegacyParamLoss

    cfg  = LossConfig()
    gen  = torch.Generator().manual_seed(7)
    mins = cfg.legacy_bounds_min
    maxs = cfg.legacy_bounds_max

    span      = torch.tensor([m2 - m1 for m1, m2 in zip(mins, maxs)]).reshape(1, 2, 3, 1, 1)
    lo        = torch.tensor(mins).reshape(1, 2, 3, 1, 1)
    pred_phys = lo + span * (torch.rand(4, 2, 3, 5, 5, generator=gen) * 1.6 - 0.3)
    gt_phys   = lo + span * torch.rand(4, 2, 3, 5, 5, generator=gen)

    ours     = LegacyParamLoss.mse(pred_phys, gt_phys, mins, maxs)
    expected = ((((pred_phys - lo) / span) - ((gt_phys - lo) / span)) ** 2).mean()

    assert ours.item() == pytest.approx(expected.item(), rel=1e-5)


def _his_masked_mse_loss(y_pred, y):
    """Literal torch transcription of the supervisor's masked MSE, channels ordered h, w, a per slot."""
    h1_p, w1_p, a1_p = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    h2_p, w2_p, a2_p = y_pred[:, 3], y_pred[:, 4], y_pred[:, 5]

    h1_t, w1_t, a1_t = y[:, 0], y[:, 1], y[:, 2]
    h2_t, w2_t, a2_t = y[:, 3], y[:, 4], y[:, 5]

    w_ths         = 0.0
    a_ths         = 0.0001
    percent       = 0.01
    upper_bound_w = w_ths * (1 + percent)

    mask2 = ((a2_p > a_ths) & (~(w2_p < upper_bound_w))).float()
    mask1 = 1.0 - mask2

    def masked_mse(pred, target, mask):
        return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

    loss1 = (
        masked_mse(h1_p, h1_t, mask1) + masked_mse(w1_p, w1_t, mask1) + masked_mse(a1_p, a1_t, mask1)
        + masked_mse(h1_p, h1_t, mask2) + masked_mse(w1_p, w1_t, mask2) + masked_mse(a1_p, a1_t, mask2)
    )
    loss2 = masked_mse(h2_p, h2_t, mask2) + masked_mse(w2_p, w2_t, mask2) + masked_mse(a2_p, a2_t, mask2)

    return loss1 + loss2


def _his_layout(phys: torch.Tensor, mins, maxs) -> torch.Tensor:
    """Scales physical parameters by the bounds and remaps them into the legacy h, w, a channel order."""
    scaled   = torch.empty(phys.shape[0], 6, *phys.shape[3:])
    ours2his = {0: 2, 1: 0, 2: 1}

    for slot in range(2):
        for ours, his in ours2his.items():
            full                      = slot * 3 + ours
            scaled[:, slot * 3 + his] = (phys[:, slot, ours] - mins[full]) / (maxs[full] - mins[full])

    return scaled


def test_legacy_mask_matches_a_verbatim_transcription_of_his_code():
    """Verifies the masked legacy loss equals a literal transcription of the supervisor's masked MSE."""
    from tools.loss.param_loss import LegacyParamLoss

    cfg  = LossConfig()
    gen  = torch.Generator().manual_seed(7)
    mins = cfg.legacy_bounds_min
    maxs = cfg.legacy_bounds_max

    span      = torch.tensor([m2 - m1 for m1, m2 in zip(mins, maxs)]).reshape(1, 2, 3, 1, 1)
    lo        = torch.tensor(mins).reshape(1, 2, 3, 1, 1)
    pred_phys = lo + span * (torch.rand(4, 2, 3, 5, 5, generator=gen) * 1.6 - 0.3)
    gt_phys   = lo + span * torch.rand(4, 2, 3, 5, 5, generator=gen)

    ours = LegacyParamLoss.masked_mse(pred_phys, gt_phys, mins, maxs, cfg.legacy_amp_thr)
    his  = _his_masked_mse_loss(_his_layout(pred_phys, mins, maxs), _his_layout(gt_phys, mins, maxs))

    assert ours.item() == pytest.approx(his.item(), rel=1e-5)


def test_legacy_pipeline_keeps_physical_slot_order_under_per_slot_bounds():
    """Verifies per-slot normalization round trips to zero loss for an exact prediction."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_cfg(), norm_stats=spock_normalizer())

    phys = torch.tensor([2.0, 0.0, 2.0, 1.5, 20.0, 5.0]).reshape(1, 6, 1, 1).expand(2, 6, 3, 3).clone()
    gt   = loss.norm_stats.normalize_output(phys)

    out  = loss(gt.clone(), gt)

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_legacy_bounds_defaults_match_the_spock_loader():
    """Verifies the default legacy bounds match the values used by the Spock loader."""
    cfg = LossConfig()

    assert cfg.legacy_bounds_min == (1e-05, -15.0, 0.01, 1e-05, 1.0, 1.0)
    assert cfg.legacy_bounds_max == (10.0, 5.0, 5.0, 10.0, 40.0, 20.0)
    assert cfg.legacy_amp_thr    == 1e-4
    assert cfg.legacy_mask is False


def test_legacy_term_logs_occupancy():
    """Verifies the legacy term reports per-slot occupancy and count diagnostics."""
    loss = build_loss(n_gaussians=2, loss_cfg=_legacy_cfg())
    out  = loss(valid_param_tensor(2, 2, 5, 5, seed=56), valid_param_tensor(2, 2, 5, 5, seed=57))

    assert "pred_active_slot1" in out["occupancy"]
    assert "count/exact_frac"  in out["occupancy"]
