
from __future__ import annotations

import torch

from pipelines.backbone.training.loss_terms import LOSS_TERMS
from tools.data.gaussians                   import GaussianActivity
from tools.data.gaussians_torch             import GaussianClamp, GaussianCurve
from tools.loss.curve_loss                  import CurveLoss, MomentLoss
from tools.loss.param_loss                  import ParamLoss, ParamMatcher


class Loss:

    MOMENT_USE_FLAGS = ("use_total_power", "use_moments")

    def __init__(self, z_axis, logger, tracker, gaussian_cfg, loss_cfg, norm_stats, log_all_losses=False, sampler=None, slot_vitals=None):
        dx = float(z_axis[1] - z_axis[0])

        self.z_axis          = z_axis
        self.logger          = logger
        self.tracker         = tracker
        self.gaussian_cfg    = gaussian_cfg
        self.reconstruct     = self.reconstruct_gaussians
        self.loss_cfg        = loss_cfg
        self.norm_stats      = norm_stats
        self.sampler         = sampler
        self.slot_vitals     = slot_vitals
        self.dx              = dx
        self.log_all_losses  = log_all_losses

        self._moment_empty_warned = False

        cfg = self.loss_cfg

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Sample points"  : z_axis.shape[0],
            "Log all losses" : self.log_all_losses,
            "Param matching" : cfg.param_matching.value,
        })

        self.log_active_terms(cfg, title="Active Terms")
        self.log_slot_presence_config(cfg, title="Slot-Presence Loss Config")
        self.log_moment_config(cfg, title="Moment Terms Config")

    @property
    def param_term_evaluated(self) -> bool:
        cfg = self.loss_cfg
        return bool(cfg.use_param_l1 or cfg.use_param_huber or cfg.use_param_mse or self.log_all_losses)

    @property
    def slot_presence_active(self) -> bool:
        cfg = self.loss_cfg
        return bool(cfg.presence_balance or cfg.use_active_normalization or cfg.amp_focal_gamma > 0.0 or self.log_all_losses)

    @staticmethod
    def active_weights(cfg) -> list[tuple[str, float]]:
        active = [(term.name, float(getattr(cfg, term.weight_key))) for term in LOSS_TERMS if getattr(cfg, term.use_flag)]

        return active

    def log_active_terms(self, cfg, title: str) -> None:
        active     = self.active_weights(cfg)
        weight_sum = sum(weight for _name, weight in active)

        active_rows = []
        for name, weight in active:
            active_rows.append({"Term": name, "Weight": f"{weight:g}", "Effective": f"{weight / weight_sum:g}"})

        self.logger.metrics_table(active_rows, ["Term", "Weight", "Effective"], title=title)
        self.logger.subsection(f"The total loss is divided by the active weight sum ({weight_sum:g}); the Effective column is the coefficient actually applied.")

        for name, weight in active:
            if weight == 0.0:
                self.logger.warning(f"Loss term '{name}' is enabled with weight 0; it is computed on every batch, gradients included, but contributes nothing to the total")
            elif weight < 0.0:
                self.logger.warning(f"Loss term '{name}' is enabled with a negative weight {weight:g}; the optimiser drives it up, not down")

    def log_slot_presence_config(self, cfg, title: str) -> None:
        if cfg.presence_balance:
            weights = {"active_weight / inactive_weight": f"derived per slot and batch as 0.5/frac and 0.5/(1-frac) with frac clamped to [{ParamLoss.FRAC_CLAMP:g}, {1.0 - ParamLoss.FRAC_CLAMP:g}]; the configured {cfg.active_weight:g} / {cfg.inactive_weight:g} are not applied; applied values under loss_monitor/presence"}
        else:
            weights = {"active_weight": cfg.active_weight, "inactive_weight": cfg.inactive_weight}

        self.logger.kv_table({
            "presence_balance"         : cfg.presence_balance,
            **weights,
            "amp_focal_gamma"          : cfg.amp_focal_gamma,
            "amp_focal_delta"          : cfg.amp_focal_delta,
            "use_active_normalization" : cfg.use_active_normalization,
        }, title=title)

    def _moment_terms_enabled(self, cfg) -> bool:
        return bool(self.log_all_losses or any(getattr(cfg, flag) for flag in self.MOMENT_USE_FLAGS))

    def log_moment_config(self, cfg, title: str) -> None:
        if not self._moment_terms_enabled(cfg):
            return

        self.logger.kv_table({
            "power_floor" : f"{cfg.power_floor:g}  (integrated target power a pixel must exceed to enter the total_power_relerr and moments terms; pixels at or below it are excluded, loss_monitor/moment_valid_frac tracks the share kept)",
        }, title=title)

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        return GaussianCurve.reconstruct(params, self.z_axis, self.gaussian_cfg.params_per_gaussian)

    def _param_term(self, matched, kind):
        pred, pred_phys, gt, gt_phys = matched
        cfg = self.loss_cfg
        w   = torch.tensor(cfg.param_weights, dtype=pred.dtype, device=pred.device)

        weights = w.reshape(1, 1, -1, 1, 1)

        active     = GaussianActivity.is_active(gt_phys[:, :, 0:1], self.gaussian_cfg.activity_threshold).to(pred.dtype)
        param_mask = torch.ones_like(pred)
        param_mask[:, :, 1:] = active.expand_as(pred[:, :, 1:])

        presence = ParamLoss.presence_scale(active, cfg.presence_balance, cfg.active_weight, cfg.inactive_weight)
        focal    = ParamLoss.focal_scale(pred[:, :, 0:1], gt[:, :, 0:1], cfg.amp_focal_gamma, cfg.amp_focal_delta)

        elem_w               = weights * param_mask * presence
        elem_w[:, :, 0:1]    = elem_w[:, :, 0:1] * focal

        if kind == "l1":
            return ParamLoss.l1(pred, gt, elem_w, ["amp", "mu", "sigma"], cfg.use_active_normalization)

        if kind == "huber":
            return ParamLoss.huber(pred, gt, elem_w, cfg.param_huber_delta, cfg.use_active_normalization), {}

        return {"mse": ParamLoss.mse}[kind](pred, gt, elem_w, cfg.use_active_normalization), {}

    @torch.no_grad()
    def _physical_errors(self, matched) -> dict:
        _, pred_phys, _, gt_phys = matched

        diff = (pred_phys - gt_phys).abs()

        active_mask = GaussianActivity.is_active(gt_phys[:, :, 0], self.gaussian_cfg.activity_threshold).to(pred_phys.dtype)
        n_active    = active_mask.sum()

        return {
            "amp_mae"        : diff[:, :, 0].mean(),
            "mu_mae"         : (diff[:, :, 1] * active_mask).sum() / n_active.clamp(min=1.0),
            "sigma_mae"      : (diff[:, :, 2] * active_mask).sum() / n_active.clamp(min=1.0),
            "gt_empty_batch" : (n_active == 0).to(pred_phys.dtype),
        }

    def _match_params(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss):
        batch_size, num_channels, height, width = pred_gauss.shape

        ppg           = self.gaussian_cfg.params_per_gaussian
        num_gaussians = num_channels // ppg

        pred      = pred_gauss.reshape(     batch_size, num_gaussians, ppg, height, width)
        pred_phys = pred_phys_gauss.reshape(batch_size, num_gaussians, ppg, height, width)

        gt_gaussians = gt_gauss.shape[1] // ppg

        gt      = gt_gauss[     :, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)
        gt_phys = gt_phys_gauss[:, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)

        return ParamMatcher.match(pred, pred_phys, gt, gt_phys, method=self.loss_cfg.param_matching.value, active_thr=self.gaussian_cfg.activity_threshold)

    def _prepare(self, pred_params, gt_params):
        clamp_cfg   = self.norm_stats.stats.clamp
        leaky_slope = clamp_cfg.leaky_slope

        pred_params_phys = self.norm_stats.denormalize_output(pred_params.float(), leaky_slope=leaky_slope)

        if clamp_cfg.enabled:
            pred_params_phys = GaussianClamp.apply(
                pred_params_phys,
                z_axis      = self.z_axis,
                amp_max     = clamp_cfg.amp_max,
                ppg         = self.gaussian_cfg.params_per_gaussian,
                leaky_slope = clamp_cfg.param_leaky_slope,
            )

        pred_params_norm = self.norm_stats.normalize_output(pred_params_phys, leaky_slope=leaky_slope)

        with torch.no_grad():
            gt_phys    = self.norm_stats.denormalize_output(gt_params.float())
            exp_curves = self.reconstruct(gt_phys)

        pred_curves = self.reconstruct(pred_params_phys.float())
        exp_curves  = exp_curves.float()

        return pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves

    @torch.no_grad()
    def curves(self, pred_output, gt_params):
        _, _, _, pred_curves, exp_curves = self._prepare(pred_output, gt_params)
        return pred_curves, exp_curves

    def _term_computes(self, cfg, diff, pred_curves, exp_curves, pred_params_norm, matched) -> dict:
        lc = CurveLoss
        mc = MomentLoss

        return {
            "mse_curve"          : lambda: lc.mse_diff(diff),
            "l1_curve"           : lambda: lc.l1_diff(diff),
            "huber_curve"        : lambda: lc.huber_diff(diff, cfg.huber_delta),
            "charbonnier_curve"  : lambda: lc.charbonnier_diff(diff, cfg.charbonnier_eps),
            "cosine_curve"       : lambda: lc.cosine(pred_curves, exp_curves, axis=1),
            "log_energy_curve"   : lambda: lc.log_energy(pred_curves, exp_curves, axis=1),
            "total_power_relerr" : lambda: mc.total_power(pred_curves, exp_curves, self.dx, cfg.power_floor),
            "moments"            : lambda: mc.moments(pred_curves, exp_curves, self.z_axis, self.dx, cfg.power_floor, cfg.moments_weights),
            "param_huber"        : lambda: self._param_term(matched, "huber")[0],
            "param_mse"          : lambda: self._param_term(matched, "mse")[0],
            "smoothness_tv"      : lambda: ParamLoss.tv(pred_params_norm),
        }

    @torch.no_grad()
    def _presence_monitor(self, matched) -> dict:
        _, _, _, gt_phys = matched

        active = GaussianActivity.is_active(gt_phys[:, :, 0:1], self.gaussian_cfg.activity_threshold).to(gt_phys.dtype)

        active_weight, inactive_weight, clamped = ParamLoss.presence_weights(active)

        out: dict = {}
        for g in range(active_weight.shape[0]):
            out[f"presence/active_weight_slot{g}"]   = active_weight[g]
            out[f"presence/inactive_weight_slot{g}"] = inactive_weight[g]
            out[f"presence/clamped_slot{g}"]         = clamped[g]

        return out

    @torch.no_grad()
    def _moment_monitor(self, exp_curves) -> dict:
        floor      = self.loss_cfg.power_floor
        valid_frac = MomentLoss.valid_fraction(exp_curves, self.dx, floor)
        empty      = (valid_frac == 0.0).to(valid_frac.dtype)

        if not self._moment_empty_warned and empty.item() == 1.0:
            self._moment_empty_warned = True
            self.logger.warning(f"No pixel of this batch has integrated target power above power_floor={floor:g}; the total_power_relerr and moments terms contribute 0.0 for such batches, and loss_monitor/moment_empty_batch counts their share per epoch")

        return {"moment_valid_frac": valid_frac, "moment_empty_batch": empty}

    @torch.no_grad()
    def _occupancy(self, pred_params_phys, gt_phys) -> dict:
        ppg = self.gaussian_cfg.params_per_gaussian
        thr = self.gaussian_cfg.activity_threshold

        Bp, Cp, Hp, Wp = pred_params_phys.shape
        pred_amp       = pred_params_phys.reshape(Bp, Cp // ppg, ppg, Hp, Wp)[:, :, 0]
        pred_active    = (pred_amp > thr).to(pred_amp.dtype)

        Bg, Cg, Hg, Wg = gt_phys.shape
        gt_amp         = gt_phys.reshape(Bg, Cg // ppg, ppg, Hg, Wg)[:, :, 0]
        gt_active      = (gt_amp > thr).to(gt_amp.dtype)

        pred_slot = pred_active.mean(dim=(0, 2, 3))
        gt_slot   = gt_active.mean(dim=(0, 2, 3))

        out: dict = {}
        out["gt_active_frac"]   = gt_active.mean()
        out["pred_active_frac"] = pred_active.mean()

        for g in range(pred_slot.shape[0]):
            out[f"pred_active_slot{g}"] = pred_slot[g]
        for g in range(gt_slot.shape[0]):
            out[f"gt_active_slot{g}"] = gt_slot[g]

        pred_count = pred_active.sum(dim=1)
        gt_count   = gt_active.sum(dim=1)

        out["count/exact_frac"] = (pred_count == gt_count).to(pred_amp.dtype).mean()
        out["count/under_frac"] = (pred_count <  gt_count).to(pred_amp.dtype).mean()
        out["count/over_frac"]  = (pred_count >  gt_count).to(pred_amp.dtype).mean()

        for k in range(1, gt_active.shape[1] + 1):
            mask_k = gt_count == k
            denom  = mask_k.sum()
            if denom > 0:
                out[f"count/acc_gt{k}"] = ((pred_count == gt_count) & mask_k).to(pred_amp.dtype).sum() / denom.to(pred_amp.dtype)

        return out

    def __call__(self, pred_output, gt_params):
        cfg = self.loss_cfg

        pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves = self._prepare(pred_output, gt_params)

        matched  = self._match_params(pred_params_norm, gt_params, gt_phys, pred_params_phys)
        physical = self._physical_errors(matched)

        if self.sampler is not None and self.sampler.active:
            self.sampler.observe(pred_params_phys)

        if self.slot_vitals is not None and self.slot_vitals.active:
            self.slot_vitals.observe(pred_params_phys)

        needs_diff = self.log_all_losses or cfg.use_mse_curve or cfg.use_l1_curve or cfg.use_huber_curve or cfg.use_charbonnier_curve
        diff       = (pred_curves - exp_curves) if needs_diff else None

        computes = self._term_computes(cfg, diff, pred_curves, exp_curves, pred_params_norm, matched)

        components : dict = {}
        monitor    : dict = {}
        occupancy  : dict = {}
        total_loss        = torch.zeros((), dtype=pred_curves.dtype, device=pred_curves.device)
        weight_sum: float = 0.0

        per_param_l1: dict = {}

        for term in LOSS_TERMS:
            is_used = getattr(cfg, term.use_flag)

            if term.name == "param_l1":
                if is_used:
                    val, per_param_l1 = self._param_term(matched, "l1")
                elif self.log_all_losses:
                    with torch.no_grad():
                        val, per_param_l1 = self._param_term(matched, "l1")
                else:
                    continue
            else:
                compute = computes[term.name]

                if is_used:
                    val = compute()
                elif self.log_all_losses:
                    with torch.no_grad():
                        val = compute()
                else:
                    continue

            if is_used:
                eff_w                 = getattr(cfg, term.weight_key)
                components[term.name] = val
                total_loss            = total_loss + eff_w * val
                weight_sum           += eff_w

            else:
                monitor[f"{term.name}_{term.space}"] = val

        for pname, pval in per_param_l1.items():
            monitor[f"param_l1/{pname}"] = pval

        if cfg.presence_balance and self.param_term_evaluated:
            monitor.update(self._presence_monitor(matched))

        if self._moment_terms_enabled(cfg):
            monitor.update(self._moment_monitor(exp_curves))

        if self.slot_presence_active:
            occupancy = self._occupancy(pred_params_phys, gt_phys)

        if weight_sum > 0.0:
            total_loss = total_loss / weight_sum

        return {
            "total_loss" : total_loss,
            "components" : components,
            "monitor"    : monitor,
            "occupancy"  : occupancy,
            "physical"   : physical,
        }
