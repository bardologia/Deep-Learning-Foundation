
from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.normalization            import OutputClampConfig
from configuration.training.backbone        import default_curriculum
from configuration.training.general.loss    import LossConfig, ParamMatching
from pipelines.backbone.training.loss       import Loss
from pipelines.backbone.training.loss_terms import LOSS_TERMS, LossComponentCatalog
from tools.loss.param_loss                  import ParamLoss
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
    assert set(out["physical"].keys())   == {"amp_mae", "mu_mae", "sigma_mae", "gt_empty_batch"}
    assert all(torch.isfinite(v).item() for v in out["physical"].values())
    assert out["physical"]["gt_empty_batch"].item() == 0.0

    out["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


def test_physical_errors_build_no_autograd_graph():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=0).requires_grad_(True)

    out = loss(pred, param_tensor(2, 2, 6, 6, seed=1))

    assert out["total_loss"].requires_grad
    assert not any(value.requires_grad for value in out["physical"].values())
    assert all(value.grad_fn is None for value in out["physical"].values())


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


def test_presence_balance_reports_the_derived_slot_weights():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True, active_weight=2.0, inactive_weight=0.5)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=41)

    gt[:, 3, :, :3] = 0.0

    out     = loss(valid_param_tensor(2, 2, 5, 5, seed=42), gt)
    monitor = out["monitor"]

    assert monitor["presence/active_weight_slot0"].item()   == pytest.approx(0.5 / (1.0 - ParamLoss.FRAC_CLAMP))
    assert monitor["presence/clamped_slot0"].item()         == 1.0
    assert monitor["presence/active_weight_slot1"].item()   == pytest.approx(0.5 / 0.4)
    assert monitor["presence/inactive_weight_slot1"].item() == pytest.approx(0.5 / 0.6)
    assert monitor["presence/clamped_slot1"].item()         == 0.0


def test_presence_weights_absent_without_balance_or_param_term():
    constants = build_loss(n_gaussians=2, loss_cfg=LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=False))
    curve     = build_loss(n_gaussians=2, loss_cfg=LossConfig(use_mse_curve=True, weight_mse_curve=1.0, presence_balance=True))
    gt        = valid_param_tensor(2, 2, 5, 5, seed=43)
    pred      = valid_param_tensor(2, 2, 5, 5, seed=44)

    assert not any(key.startswith("presence/") for key in constants(pred, gt)["monitor"])
    assert not any(key.startswith("presence/") for key in curve(pred, gt)["monitor"])


class _TableRecorder(NullLogger):

    def __init__(self):
        self.tables   = []
        self.warnings = []

    def kv_table(self, data, title=None, **kwargs):
        self.tables.append((title, dict(data)))

    def warning(self, message):
        self.warnings.append(message)


def _recording_loss(cfg: LossConfig, log_all_losses: bool = False) -> tuple[Loss, _TableRecorder]:
    logger = _TableRecorder()
    loss   = Loss(z_axis=x_axis_tensor(), logger=logger, tracker=NullTracker(), gaussian_cfg=gaussian_config(2), loss_cfg=cfg, norm_stats=identity_normalizer(6), log_all_losses=log_all_losses)

    return loss, logger


def _gt_without_active_slots(seed: int) -> torch.Tensor:
    gt = valid_param_tensor(2, 2, 5, 5, seed=seed)

    gt[:, 0] = 0.0
    gt[:, 3] = 0.0

    return gt


def test_slot_presence_table_states_the_derived_weights_under_balance():
    _loss, logger = _recording_loss(LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True, active_weight=2.0, inactive_weight=0.5))

    table = next(table for title, table in logger.tables if title == "Slot-Presence Loss Config")

    assert "active_weight" not in table
    assert "derived per slot and batch" in table["active_weight / inactive_weight"]
    assert "2 / 0.5 are not applied" in table["active_weight / inactive_weight"]


def test_moment_table_lists_the_floor_only_when_a_moment_term_is_evaluated():
    _with, with_logger       = _recording_loss(LossConfig(use_total_power=True, weight_total_power=1.0, power_floor=0.02))
    _moments, moments_logger = _recording_loss(LossConfig(use_moments=True, weight_moments=1.0))
    _probe, probe_logger     = _recording_loss(LossConfig(use_param_l1=True, weight_param_l1=1.0), log_all_losses=True)
    _without, without_logger = _recording_loss(LossConfig(use_param_l1=True, weight_param_l1=1.0))

    table = next(table for title, table in with_logger.tables if title == "Moment Terms Config")

    assert table["power_floor"].startswith("0.02  (")
    assert "excluded" in table["power_floor"]
    assert any(title == "Moment Terms Config" for title, _ in moments_logger.tables)
    assert any(title == "Moment Terms Config" for title, _ in probe_logger.tables)
    assert not any(title == "Moment Terms Config" for title, _ in without_logger.tables)


def test_moment_terms_report_the_valid_fraction_and_flag_empty_batches():
    loss, logger = _recording_loss(LossConfig(use_total_power=True, weight_total_power=1.0))
    pred         = valid_param_tensor(2, 2, 5, 5, seed=45)

    full  = loss(pred, valid_param_tensor(2, 2, 5, 5, seed=46))
    empty = loss(pred, _gt_without_active_slots(seed=47))
    again = loss(pred, _gt_without_active_slots(seed=48))

    assert full["monitor"]["moment_valid_frac"].item()      == pytest.approx(1.0)
    assert full["monitor"]["moment_empty_batch"].item()     == 0.0
    assert empty["monitor"]["moment_valid_frac"].item()     == 0.0
    assert empty["monitor"]["moment_empty_batch"].item()    == 1.0
    assert empty["components"]["total_power_relerr"].item() == 0.0
    assert again["monitor"]["moment_empty_batch"].item()    == 1.0
    assert len(logger.warnings) == 1
    assert "power_floor" in logger.warnings[0]


def test_moment_empty_batch_warning_rearms_at_the_curriculum_swap():
    loss, logger = _recording_loss(LossConfig(use_moments=True, weight_moments=1.0))
    pred         = valid_param_tensor(2, 2, 5, 5, seed=53)

    loss(pred, _gt_without_active_slots(seed=54))
    loss.set_curriculum(LossConfig(use_total_power=True, weight_total_power=1.0))
    swapped = loss(pred, _gt_without_active_slots(seed=55))

    assert swapped["monitor"]["moment_empty_batch"].item() == 1.0
    assert len(logger.warnings) == 2
    assert any(title == "Moment Terms Config (curriculum swap)" for title, _ in logger.tables)


def test_moment_valid_fraction_absent_without_a_moment_term():
    loss = build_loss(n_gaussians=2)
    out  = loss(valid_param_tensor(2, 2, 5, 5, seed=49), valid_param_tensor(2, 2, 5, 5, seed=50))

    assert "moment_valid_frac" not in out["monitor"]
    assert "moment_empty_batch" not in out["monitor"]


def test_physical_errors_flag_a_batch_without_active_gt_slots():
    loss = build_loss(n_gaussians=2)
    out  = loss(valid_param_tensor(2, 2, 5, 5, seed=51), _gt_without_active_slots(seed=52))

    assert out["physical"]["gt_empty_batch"].item() == 1.0
    assert out["physical"]["mu_mae"].item()         == 0.0
    assert out["physical"]["sigma_mae"].item()      == 0.0


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


def test_zero_weight_sum_is_rejected():
    with pytest.raises(ValueError, match="summing to 0"):
        build_loss(loss_cfg=LossConfig(use_param_l1=True, weight_param_l1=0.0))


def test_stage_without_enabled_terms_is_rejected():
    with pytest.raises(ValueError, match="No loss term is enabled"):
        build_loss(loss_cfg=LossConfig())


def test_zero_weight_term_beside_a_weighted_one_is_warned_about(recording_logger):
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_cosine_curve=True, weight_cosine_curve=0.0)
    loss = build_loss(loss_cfg=cfg, logger=recording_logger)

    warnings = [message for level, message in recording_logger.messages if level == "warning"]

    assert dict(loss.active_weights(cfg)) == {"cosine_curve": 0.0, "param_l1": 1.0}
    assert warnings == ["Loss term 'cosine_curve' is enabled with weight 0; it is computed on every batch, gradients included, but contributes nothing to the total"]


def test_curriculum_swap_to_a_zero_weight_stage_is_rejected():
    loss = build_loss()

    with pytest.raises(ValueError, match="summing to 0"):
        loss.set_curriculum(LossConfig(use_mse_curve=True, weight_mse_curve=0.0))


def test_dx_is_the_z_axis_spacing():
    axis = x_axis_tensor()
    loss = build_loss()

    assert loss.dx == pytest.approx(float(axis[1] - axis[0]))


def test_single_sample_z_axis_raises():
    with pytest.raises(ValueError, match="at least 2 sample points"):
        build_loss(length=1)


def test_decreasing_z_axis_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        Loss(
            z_axis       = torch.flip(x_axis_tensor(), dims=[0]),
            logger       = NullLogger(),
            tracker      = NullTracker(),
            gaussian_cfg = gaussian_config(2),
            loss_cfg     = LossConfig(use_param_l1=True, weight_param_l1=1.0),
            norm_stats   = identity_normalizer(6),
        )


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


ZERO_FLOOR_MIN = (0.0, 0.0, 2.0, 0.0, 0.0, 2.0)
ZERO_FLOOR_MAX = (4.0, 80.0, 10.0, 4.0, 80.0, 10.0)












































def _his_masked_mse_loss(y_pred, y):
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










