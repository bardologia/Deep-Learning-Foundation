"""Composite training loss over predicted Gaussian z profiles.

Assembles curve-space and parameter-space terms into one weighted
objective, tracks the inactive terms as monitoring values, and reports slot
occupancy and physical parameter errors alongside the loss.
"""

from __future__ import annotations

import torch

from pipelines.backbone.training.loss_terms import LOSS_TERMS
from tools.data.gaussians                   import GaussianActivity
from tools.data.gaussians_torch             import GaussianClamp, GaussianCurve
from tools.loss.curve_loss                  import CurveLoss, MomentLoss
from tools.loss.param_loss                  import ParamLoss, ParamMatcher


class Loss:
    """Weighted sum of the enabled curve and parameter loss terms.

    Predictions arrive as normalized Gaussian parameter maps; they are
    denormalized, optionally clamped, matched against the ground-truth slots and
    reconstructed into z profiles before the terms are evaluated.

    Attributes:
        z_axis          : The z sample positions, of shape (profile_length,).
        gaussian_cfg    : Gaussian head configuration (parameters per Gaussian, axis bounds, activity threshold of the label run).
        loss_cfg        : Active loss-stage configuration; swapped by the curriculum.
        norm_stats      : Normalizer converting between normalized and physical parameters.
        dx              : The z sample spacing.
        log_all_losses  : Whether disabled terms are still evaluated for monitoring.
        loss_generation : Counter incremented on every curriculum swap.
    """

    def __init__(self, z_axis, logger, tracker, gaussian_cfg, loss_cfg, norm_stats, log_all_losses=False, sampler=None, slot_vitals=None):
        """Builds the loss and logs its active terms.

        Args:
            z_axis         : The z sample positions, of shape (profile_length,).
            logger         : Logger receiving the configuration tables.
            tracker        : Metrics tracker used by the diagnostics hooks.
            gaussian_cfg   : Gaussian head configuration.
            loss_cfg       : Initial loss-stage configuration.
            norm_stats     : Normalizer for the output parameter space.
            log_all_losses : Whether to evaluate disabled terms for monitoring.
            sampler        : Optional parameter histogram sampler observing predictions.
            slot_vitals    : Optional per-slot diagnostics observer.

        Raises:
            ValueError : If the z axis holds fewer than two samples or does not increase, or if the stage enables no term or only terms whose weights sum to zero, which would make the total loss identically zero.
        """
        if z_axis.shape[0] < 2:
            raise ValueError(f"z_axis must contain at least 2 sample points, got length {z_axis.shape[0]}")

        dx = float(z_axis[1] - z_axis[0])
        if dx <= 0:
            raise ValueError(f"z_axis must be strictly increasing, got dx={dx}")

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
        self.loss_generation = 0

        cfg = self.loss_cfg

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Sample points"  : z_axis.shape[0],
            "Log all losses" : self.log_all_losses,
            "Param matching" : cfg.param_matching.value,
        })

        self.log_active_terms(cfg, title="Active Terms")
        self.log_slot_presence_config(cfg, title="Slot-Presence Loss Config")

    @property
    def param_term_evaluated(self) -> bool:
        """Whether a presence-weighted parameter term is evaluated this stage, as a loss or for monitoring."""
        cfg = self.loss_cfg
        return bool(cfg.use_param_l1 or cfg.use_param_huber or cfg.use_param_mse or self.log_all_losses)

    @property
    def slot_presence_active(self) -> bool:
        """Whether any presence-sensitive setting makes the occupancy statistics worth computing."""
        cfg = self.loss_cfg
        return bool(cfg.presence_balance or cfg.use_active_normalization or cfg.amp_focal_gamma > 0.0 or self.log_all_losses)

    @staticmethod
    def active_weights(cfg) -> list[tuple[str, float]]:
        """Returns the enabled terms of a stage with their weights, refusing a stage whose total would be identically zero.

        Args:
            cfg : Loss-stage configuration whose flags and weights are read.

        Returns:
            (term name, weight) pairs of the enabled terms, in LOSS_TERMS order.

        Raises:
            ValueError : If no term is enabled, or if the enabled weights do not sum to a positive value, since the total is divided by that sum and would otherwise be identically zero.
        """
        active = [(term.name, float(getattr(cfg, term.weight_key))) for term in LOSS_TERMS if getattr(cfg, term.use_flag)]

        if not active:
            raise ValueError("No loss term is enabled in this stage; the total loss would be identically zero and the run would train on nothing")

        weight_sum = sum(weight for _name, weight in active)
        if weight_sum <= 0.0:
            raise ValueError(f"The enabled loss terms {dict(active)} have weights summing to {weight_sum:g}; the total is divided by that sum and must be positive, otherwise the run trains on nothing")

        return active

    def log_active_terms(self, cfg, title: str) -> None:
        """Logs the enabled terms with their configured and weight-sum-normalized coefficients.

        A term enabled at weight zero is computed on every batch, gradients
        included, without reaching the total, so it is warned about by name.

        Args:
            cfg   : Loss-stage configuration whose flags and weights are reported.
            title : Table title.

        Raises:
            ValueError : If the stage enables no term or its enabled weights do not sum to a positive value.
        """
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
        """Logs the presence-balancing and amplitude-focal settings of a loss stage.

        Under presence_balance the configured active and inactive weights are
        not applied, the weights being derived per slot and batch from the
        active fraction, so the table says so instead of printing idle constants.

        Args:
            cfg   : Loss-stage configuration whose presence settings are reported.
            title : Table title.
        """
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

    def set_curriculum(self, complete_cfg, context: str = "curriculum swap") -> None:
        """Replaces the active loss configuration and logs the new composition.

        Args:
            complete_cfg : Loss-stage configuration to switch to.
            context      : Label describing why the swap happens, used in the log.

        Raises:
            ValueError : If the new configuration enables no term or only terms whose weights sum to zero.
        """

        self.loss_cfg       = complete_cfg
        self.loss_generation += 1

        if context == "curriculum swap":
            self.logger.subsection("Active loss composition changes at the curriculum swap; train/val loss curves are not comparable across the swap epoch.")

        self.logger.subsection(f"Param matching ({context}): {complete_cfg.param_matching.value}")
        self.log_active_terms(complete_cfg, title=f"Active Terms ({context})")
        self.log_slot_presence_config(complete_cfg, title=f"Slot-Presence Loss Config ({context})")

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        """Renders physical Gaussian parameters into z profiles.

        Args:
            params : Physical Gaussian parameters of shape (batch, n_gaussians * params_per_gaussian, z, width).

        Returns:
            Reflectivity profiles of shape
            (batch, profile_length, z, width), sampled on the z axis.
        """
        return GaussianCurve.reconstruct(params, self.z_axis, self.gaussian_cfg.params_per_gaussian)

    def _param_term(self, matched, kind):
        """Evaluates a parameter-space loss with presence, focal and per-parameter weighting.

        Args:
            matched : Tuple of normalized predictions, physical predictions, normalized targets and physical targets, each of shape (batch, n_gaussians, params_per_gaussian, height, width), already paired by the matcher.
            kind    : One of 'l1', 'huber' or 'mse'.

        Returns:
            A tuple of the scalar loss and a dict of per-parameter L1 values;
            the dict is empty for the huber and mse variants.

        Raises:
            ValueError : If `param_weights` does not hold one weight per Gaussian parameter, or `kind` is unknown.
        """
        pred, pred_phys, gt, gt_phys = matched
        cfg = self.loss_cfg
        w   = torch.tensor(cfg.param_weights, dtype=pred.dtype, device=pred.device)

        if w.numel() != pred.shape[2]:
            raise ValueError(f"param_weights has {w.numel()} entries but each Gaussian has {pred.shape[2]} parameters; configure exactly one weight per parameter.")

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

        if kind == "mse":
            return ParamLoss.mse(pred, gt, elem_w, cfg.use_active_normalization), {}

        raise ValueError(f"Unknown param_term kind: {kind!r}. Expected 'l1', 'huber', or 'mse'.")

    def _physical_errors(self, matched) -> dict:
        """Reports mean absolute errors of the matched Gaussians in physical units.

        Args:
            matched : The matcher output; the physical prediction and target tensors have shape (batch, n_gaussians, params_per_gaussian, height, width).

        Returns:
            Dict with the amplitude MAE over all slots, the mean and sigma MAE in
            metres restricted to the slots the ground truth marks active, and
            gt_empty_batch, 1.0 when the batch holds no active ground-truth slot
            so both metre errors are 0.0 by construction; its epoch mean is the
            share of such batches.
        """
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
        """Pairs predicted with ground-truth Gaussian slots under the configured strategy.

        Args:
            pred_gauss      : Normalized predictions of shape (batch, n_gaussians * params_per_gaussian, height, width).
            gt_gauss        : Normalized targets in the same layout.
            gt_phys_gauss   : Physical targets in the same layout.
            pred_phys_gauss : Physical predictions in the same layout.

        Returns:
            The matched normalized and physical prediction and target tensors,
            each of shape
            (batch, n_gaussians, params_per_gaussian, height, width).

        Raises:
            ValueError : If prediction and ground truth carry different slot counts.
        """
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

        return ParamMatcher.match(pred, pred_phys, gt, gt_phys, method=self.loss_cfg.param_matching.value, active_thr=self.gaussian_cfg.activity_threshold)

    def _prepare(self, pred_params, gt_params):
        """Denormalizes, clamps and renders both predictions and targets.

        Args:
            pred_params : Raw model output of shape (batch, n_gaussians * params_per_gaussian, height, width).
            gt_params   : Normalized target parameters in the same layout.

        Returns:
            Tuple of the re-normalized predictions, the physical predictions,
            the physical targets, the predicted profiles and the target profiles;
            the profile tensors have shape
            (batch, profile_length, height, width).
        """
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
        """Returns the predicted and expected z profiles for a batch.

        Args:
            pred_output : Raw model output of shape (batch, n_gaussians * params_per_gaussian, height, width).
            gt_params   : Normalized target parameters in the same layout.

        Returns:
            The predicted and expected profiles, both of shape
            (batch, profile_length, height, width).
        """
        _, _, _, pred_curves, exp_curves = self._prepare(pred_output, gt_params)
        return pred_curves, exp_curves

    def _term_computes(self, cfg, diff, pred_curves, exp_curves, pred_params_norm, matched) -> dict:
        """Builds the lazy evaluators of every loss term for one batch.

        Args:
            cfg              : Active loss-stage configuration supplying the term parameters.
            diff             : Profile residual of shape (batch, profile_length, rows, cols), or None when no residual-based term is needed.
            pred_curves      : Predicted profiles of shape (batch, profile_length, rows, cols).
            exp_curves       : Expected profiles in the same layout.
            pred_params_norm : Normalized predicted parameters of shape (batch, n_gaussians * params_per_gaussian, rows, cols).
            matched          : Matcher output used by the parameter terms.

        Returns:
            Mapping of term name to a zero-argument callable returning the term value.
        """
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
        """Reports the balanced presence weights this batch applied per slot and whether the fraction clamp engaged.

        Args:
            matched : The matcher output; the physical target tensor has shape (batch, n_gaussians, params_per_gaussian, height, width).

        Returns:
            Dict keyed presence/active_weight_slot{g}, presence/inactive_weight_slot{g}
            and presence/clamped_slot{g}, the flag being 1.0 when the slot's active
            fraction hit FRAC_CLAMP, so its epoch mean is the share of clamped batches.
        """
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
    def _occupancy(self, pred_params_phys, gt_phys) -> dict:
        """Reports how many Gaussian slots the prediction activates against the ground truth.

        Args:
            pred_params_phys : Physical predicted parameters of shape (batch, n_gaussians * params_per_gaussian, height, width).
            gt_phys          : Physical target parameters in the same layout.

        Returns:
            Dict with overall and per-slot active fractions, the exact,
            under- and over-count fractions, and the count accuracy conditioned
            on each ground-truth component count.
        """
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
        """Evaluates the composite loss for one batch.

        Enabled terms are summed with their weights and the total divided by the
        weight sum; disabled terms are evaluated without gradients only when
        `log_all_losses` is set, and land in the monitor dict.

        Args:
            pred_output : Raw model output of shape (batch, n_gaussians * params_per_gaussian, rows, cols).
            gt_params   : Normalized target parameters in the same layout.

        Returns:
            Dict with the scalar `total_loss`, the weighted `components`, the
            unweighted `monitor` values (inactive terms, per-parameter L1 parts
            and, under presence_balance, the applied per-slot presence weights),
            the slot `occupancy` statistics and the `physical` parameter errors.
        """
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
