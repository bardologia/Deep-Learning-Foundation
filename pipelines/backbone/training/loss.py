from __future__ import annotations

import torch

from pipelines.backbone.training.loss_terms import LOSS_TERMS
from tools.data.gaussians                   import GaussianClamp, GaussianCurve
from tools.loss.curve_loss                  import CurveLoss
from tools.loss.param_loss                  import LegacyParamLoss, ParamLoss, ParamMatcher
from tools.loss.physical_loss               import PhysicalLoss
from tools.sar.tomo_geometry                import TomoGeometry


class Loss:
    def __init__(self, x_axis, logger, tracker, gaussian_cfg, loss_cfg, norm_stats, geometry_cfg, log_all_losses=False, sampler=None, slot_vitals=None):
        self.x_axis          = x_axis
        self.logger          = logger
        self.tracker         = tracker
        self.gaussian_cfg    = gaussian_cfg
        self.reconstruct     = self.reconstruct_gaussians
        self.loss_cfg        = loss_cfg
        self.norm_stats      = norm_stats
        self.geometry_cfg    = geometry_cfg
        self.sampler         = sampler
        self.slot_vitals     = slot_vitals
        self.geometry        = TomoGeometry(self.geometry_cfg, x_axis)
        self.dx              = float(x_axis[1] - x_axis[0])
        self.log_all_losses  = log_all_losses
        self.loss_generation = 0

        cfg = self.loss_cfg
        self._validate_legacy(cfg)

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Sample points"  : x_axis.shape[0],
            "Log all losses" : self.log_all_losses,
            "Param matching" : cfg.param_matching.value,
        })
        self.logger.kv_table(self.geometry.describe(), title="Tomographic Geometry (scene-constant fallback; a per-pixel kz map supplied by the dataset supersedes it)")

        self.log_active_terms(cfg, title="Active Terms")
        self.log_slot_presence_config(cfg, title="Slot-Presence Loss Config")

    @property
    def slot_presence_active(self) -> bool:
        cfg = self.loss_cfg
        return bool(cfg.presence_balance or cfg.use_active_normalization or cfg.amp_focal_gamma > 0.0 or cfg.use_param_legacy or self.log_all_losses)

    def log_active_terms(self, cfg, title: str) -> None:
        active     = [(term.name, float(getattr(cfg, term.weight_key))) for term in LOSS_TERMS if getattr(cfg, term.use_flag)]
        weight_sum = sum(weight for _name, weight in active)

        active_rows = []
        for name, weight in active:
            effective = weight / weight_sum if weight_sum > 0.0 else weight
            active_rows.append({"Term": name, "Weight": f"{weight:g}", "Effective": f"{effective:g}"})

        self.logger.metrics_table(active_rows, ["Term", "Weight", "Effective"], title=title)

        if weight_sum > 0.0:
            self.logger.subsection(f"The total loss is divided by the active weight sum ({weight_sum:g}); the Effective column is the coefficient actually applied.")

    def log_slot_presence_config(self, cfg, title: str) -> None:
        self.logger.kv_table({
            "presence_balance"         : cfg.presence_balance,
            "active_weight"            : cfg.active_weight,
            "inactive_weight"          : cfg.inactive_weight,
            "amp_focal_gamma"          : cfg.amp_focal_gamma,
            "amp_focal_delta"          : cfg.amp_focal_delta,
            "use_active_normalization" : cfg.use_active_normalization,
        }, title=title)

    def set_curriculum(self, complete_cfg, context: str = "curriculum swap") -> None:
        self._validate_legacy(complete_cfg)

        self.loss_cfg       = complete_cfg
        self.loss_generation += 1

        if context == "curriculum swap":
            self.logger.subsection("Active loss composition changes at the curriculum swap; train/val loss curves are not comparable across the swap epoch.")

        self.logger.subsection(f"Param matching ({context}): {complete_cfg.param_matching.value}")
        self.log_active_terms(complete_cfg, title=f"Active Terms ({context})")
        self.log_slot_presence_config(complete_cfg, title=f"Slot-Presence Loss Config ({context})")

    def _validate_legacy(self, cfg) -> None:
        if cfg.use_param_legacy and cfg.param_matching.value != ParamMatcher.SORTED_GT:
            raise ValueError(f"use_param_legacy requires param_matching={ParamMatcher.SORTED_GT!r}; the legacy per-slot bounds assume fixed slot identity and {cfg.param_matching.value!r} reassigns prediction slots per pixel.")

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        return GaussianCurve.reconstruct(params, self.x_axis, self.gaussian_cfg.params_per_gaussian)

    def _param_term(self, matched, kind):
        pred, pred_phys, gt, gt_phys = matched
        cfg = self.loss_cfg
        w   = torch.tensor(cfg.param_weights, dtype=pred.dtype, device=pred.device)

        if w.numel() != pred.shape[2]:
            raise ValueError(f"param_weights has {w.numel()} entries but each Gaussian has {pred.shape[2]} parameters; configure exactly one weight per parameter.")

        weights = w.reshape(1, 1, -1, 1, 1)

        active     = (gt_phys[:, :, 0:1] > cfg.amp_zero_thr).to(pred.dtype)
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

        if kind == "mse":
            return ParamLoss.mse(pred, gt, elem_w, cfg.use_active_normalization), {}

        raise ValueError(f"Unknown param_term kind: {kind!r}. Expected 'l1', 'huber', or 'mse'.")

    def _param_legacy(self, matched):
        _pred, pred_phys, _gt, gt_phys = matched
        cfg = self.loss_cfg
        return LegacyParamLoss.mse(pred_phys, gt_phys, cfg.legacy_bounds_min, cfg.legacy_bounds_max)

    @torch.no_grad()
    def _physical_errors(self, matched) -> dict:
        _, pred_phys, _, gt_phys = matched

        diff = (pred_phys - gt_phys).abs()

        active_mask = (gt_phys[:, :, 0] > self.loss_cfg.amp_zero_thr).to(pred_phys.dtype)
        n_active    = active_mask.sum().clamp(min=1.0)

        return {
            "amp_mae"     : diff[:, :, 0].mean(),
            "mu_mae_m"    : (diff[:, :, 1] * active_mask).sum() / n_active,
            "sigma_mae_m" : (diff[:, :, 2] * active_mask).sum() / n_active,
        }

    def _match_params(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss):
        batch_size, num_channels, height, width = pred_gauss.shape

        ppg           = self.gaussian_cfg.params_per_gaussian
        num_gaussians = num_channels // ppg

        pred      = pred_gauss.reshape(     batch_size, num_gaussians, ppg, height, width)
        pred_phys = pred_phys_gauss.reshape(batch_size, num_gaussians, ppg, height, width)

        gt_gaussians = gt_gauss.shape[1] // ppg

        if gt_gaussians != num_gaussians:
            raise ValueError(f"The model predicts {num_gaussians} Gaussians but the ground truth carries {gt_gaussians}; both sides must describe the same slot count for the loss to pair them.")

        gt      = gt_gauss[     :, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)
        gt_phys = gt_phys_gauss[:, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)

        return ParamMatcher.match(pred, pred_phys, gt, gt_phys, method=self.loss_cfg.param_matching.value, active_thr=self.loss_cfg.amp_zero_thr)

    def _prepare(self, pred_params, gt_params):
        clamp_cfg   = self.norm_stats.stats.clamp
        leaky_slope = clamp_cfg.leaky_slope

        pred_params_phys = self.norm_stats.denormalize_output(pred_params.float(), leaky_slope=leaky_slope)

        if clamp_cfg.enabled:
            pred_params_phys = GaussianClamp.apply(
                pred_params_phys,
                x_axis      = self.x_axis,
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

    def _term_computes(self, cfg, diff, pred_curves, exp_curves, pred_params_norm, matched, kz_map) -> dict:
        lc = CurveLoss
        pc = PhysicalLoss

        if kz_map is not None:
            coherence_resyn  = lambda: pc.coherence_resynthesis_pp(pred_curves, exp_curves, kz_map, self.x_axis, self.dx, cfg.physics_floor)
            covariance_match = lambda: pc.covariance_matching_pp(pred_curves, exp_curves, kz_map, self.x_axis, self.dx, cfg.physics_floor)
            capon_cycle      = lambda: pc.capon_cycle_pp(pred_curves, exp_curves, kz_map, self.x_axis, self.dx, cfg.capon_loading, cfg.physics_floor)
        else:
            coherence_resyn  = lambda: pc.coherence_resynthesis(pred_curves, exp_curves, self.geometry.steering, self.dx, cfg.physics_floor)
            covariance_match = lambda: pc.covariance_matching(pred_curves, exp_curves, self.geometry.outer, self.dx, cfg.physics_floor)
            capon_cycle      = lambda: pc.capon_cycle(pred_curves, exp_curves, self.geometry.steering, self.geometry.outer, self.dx, cfg.capon_loading, cfg.physics_floor)

        return {
            "mse_curve"          : lambda: lc.mse_diff(diff),
            "l1_curve"           : lambda: lc.l1_diff(diff),
            "huber_curve"        : lambda: lc.huber_diff(diff, cfg.huber_delta),
            "charbonnier_curve"  : lambda: lc.charbonnier_diff(diff, cfg.charbonnier_eps),
            "cosine_curve"       : lambda: lc.cosine(pred_curves, exp_curves, axis=1),
            "total_power_relerr" : lambda: pc.total_power(pred_curves, exp_curves, self.dx, cfg.physics_floor),
            "moments"            : lambda: pc.moments(pred_curves, exp_curves, self.x_axis, self.dx, cfg.physics_floor, cfg.moments_weights),
            "coherence_resyn"    : coherence_resyn,
            "covariance_match"   : covariance_match,
            "capon_cycle"        : capon_cycle,
            "param_huber"        : lambda: self._param_term(matched, "huber")[0],
            "param_mse"          : lambda: self._param_term(matched, "mse")[0],
            "smoothness_tv"      : lambda: ParamLoss.tv(pred_params_norm),
        }

    @torch.no_grad()
    def _occupancy(self, pred_params_phys, gt_phys) -> dict:
        cfg = self.loss_cfg
        ppg = self.gaussian_cfg.params_per_gaussian
        thr = cfg.amp_zero_thr

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

    def __call__(self, pred_output, gt_params, kz_map=None):
        cfg = self.loss_cfg

        pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves = self._prepare(pred_output, gt_params)

        matched  = self._match_params(pred_params_norm, gt_params, gt_phys, pred_params_phys)
        physical = self._physical_errors(matched)

        if self.sampler is not None and self.sampler.active:
            self.sampler.observe(pred_params_phys)

        if self.slot_vitals is not None and self.slot_vitals.active:
            self.slot_vitals.observe(pred_params_phys)

        if kz_map is not None:
            kz_map = kz_map.to(device=pred_curves.device, dtype=pred_curves.dtype)

        needs_diff = self.log_all_losses or cfg.use_mse_curve or cfg.use_l1_curve or cfg.use_huber_curve or cfg.use_charbonnier_curve
        diff       = (pred_curves - exp_curves) if needs_diff else None

        computes = self._term_computes(cfg, diff, pred_curves, exp_curves, pred_params_norm, matched, kz_map)

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
            elif term.name == "param_legacy":
                if is_used:
                    val = self._param_legacy(matched)
                elif self.log_all_losses and matched[0].shape[1] == LegacyParamLoss.LEGACY_GAUSSIANS:
                    with torch.no_grad():
                        val = self._param_legacy(matched)
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
