"""Slot matching and penalties for Gaussian scatterer parameter predictions.

Set prediction over Gaussian slots is permutation invariant, so predictions are
first assigned to ground-truth slots per pixel, and only then compared with the
weighted regression penalties defined here.
"""

from __future__ import annotations

import itertools

import torch


class ParamMatcher:
    """Per-pixel assignment of predicted Gaussian slots to ground-truth slots.

    Matching is exhaustive over the G! slot permutations, which bounds the
    usable number of Gaussians at MAX_GAUSSIANS.
    """

    MAX_GAUSSIANS  = 6
    ACTIVE_AMP_THR = 1e-4

    HUNGARIAN = "hungarian"
    SORTED_GT = "sorted_gt"

    @staticmethod
    def is_active(amplitude, thr: float = ACTIVE_AMP_THR):
        """Returns whether an amplitude reaches the active-scatterer threshold.

        Args:
            amplitude : Physical amplitude tensor or array of any shape.
            thr       : Physical amplitude at or above which a slot counts as active.

        Returns:
            Boolean tensor or array of the same shape as amplitude.
        """
        return amplitude >= thr

    @staticmethod
    def _sort_gt(gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorts ground-truth slots per pixel by ascending elevation, inactive slots last.

        Args:
            gt         : Normalised ground-truth parameters of shape (batch, gaussians, params, height, width).
            gt_phys    : Physical-unit ground-truth parameters of the same shape, with amplitude at param index 0 and elevation mu in metres at param index 1.
            active_thr : Amplitude above which a ground-truth slot counts as active.

        Returns:
            Tuple of the sorted normalised and physical ground-truth tensors.
        """
        gt_phys_amp = gt_phys[:, :, 0]
        gt_mu       = gt_phys[:, :, 1]
        is_active   = gt_phys_amp > active_thr
        sort_key    = torch.where(is_active, gt_mu, torch.full_like(gt_mu, float("inf")))
        gt_index    = torch.argsort(sort_key, dim=1, stable=True)
        gt_idx_b    = gt_index[:, :, None, :, :].expand_as(gt)

        return torch.gather(gt, dim=1, index=gt_idx_b), torch.gather(gt_phys, dim=1, index=gt_idx_b)

    @staticmethod
    def _assign_pred_to_gt(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Permutes predicted slots into the ground-truth slot order that minimises L1 cost.

        The assignment cost of a pixel is the L1 distance between normalised
        parameter vectors, summed only over active ground-truth slots, and the
        minimum is found by enumerating all G! permutations.

        Args:
            pred       : Normalised predicted parameters of shape (batch, gaussians, params, height, width).
            pred_phys  : Physical-unit predicted parameters of the same shape.
            gt         : Normalised ground-truth parameters of the same shape.
            gt_phys    : Physical-unit ground-truth parameters of the same shape.
            active_thr : Amplitude above which a ground-truth slot counts as active.

        Returns:
            Tuple of the permuted normalised and physical prediction tensors.

        Raises:
            ValueError : If prediction and ground truth disagree on the number of Gaussian slots, or if that number exceeds MAX_GAUSSIANS.
        """
        B, G, P, H, W = pred.shape

        if gt.shape[1] != G:
            raise ValueError(f"ParamMatcher.match requires equal pred/gt gaussian counts, got {G} and {gt.shape[1]}")

        if G > ParamMatcher.MAX_GAUSSIANS:
            raise ValueError(f"ParamMatcher.match enumerates G! permutations; G={G} exceeds MAX_GAUSSIANS={ParamMatcher.MAX_GAUSSIANS}")

        active = (gt_phys[:, :, 0] > active_thr).to(pred.dtype)

        pred_e = pred.permute(0, 3, 4, 1, 2)[:, :, :, :, None, :]
        gt_e   = gt.permute(  0, 3, 4, 1, 2)[:, :, :, None, :, :]
        cost   = ((pred_e - gt_e).abs().sum(-1) * active.permute(0, 2, 3, 1)[:, :, :, None, :]).detach()

        perms     = list(itertools.permutations(range(G)))
        gt_arange = torch.arange(G, device=pred.device)
        perm_t    = torch.tensor(perms, device=pred.device)

        best = torch.zeros(cost.shape[:3], dtype=torch.long, device=pred.device)
        low  = cost[:, :, :, perm_t[0], gt_arange].sum(-1)

        for position in range(1, len(perms)):
            candidate = cost[:, :, :, perm_t[position], gt_arange].sum(-1)
            improved  = candidate < low

            low  = torch.where(improved, candidate, low)
            best = torch.where(improved, position, best)

        chosen = perm_t[best].permute(0, 3, 1, 2)
        idx_b  = chosen[:, :, None, :, :].expand(B, G, P, H, W)

        return torch.gather(pred, dim=1, index=idx_b), torch.gather(pred_phys, dim=1, index=idx_b)

    @staticmethod
    def _match_hungarian(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sorts ground truth by elevation and then assigns predictions to it by minimum cost."""
        gt, gt_phys     = ParamMatcher._sort_gt(gt, gt_phys, active_thr)
        pred, pred_phys = ParamMatcher._assign_pred_to_gt(pred, pred_phys, gt, gt_phys, active_thr)

        return pred, pred_phys, gt, gt_phys

    @staticmethod
    def _match_sorted_gt(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sorts ground truth by elevation and leaves the predicted slot order untouched."""
        gt, gt_phys = ParamMatcher._sort_gt(gt, gt_phys, active_thr)

        return pred, pred_phys, gt, gt_phys

    @staticmethod
    def match(
        pred       : torch.Tensor,
        pred_phys  : torch.Tensor,
        gt         : torch.Tensor,
        gt_phys    : torch.Tensor,
        method     : str   = "hungarian",
        active_thr : float = ACTIVE_AMP_THR,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aligns predicted and ground-truth Gaussian slots with the requested strategy.

        Args:
            pred       : Normalised predicted parameters of shape (batch, gaussians, params, height, width).
            pred_phys  : Physical-unit predicted parameters of the same shape.
            gt         : Normalised ground-truth parameters of the same shape.
            gt_phys    : Physical-unit ground-truth parameters of the same shape.
            method     : Either "hungarian" for per-pixel minimum-cost assignment or "sorted_gt" for elevation-sorted ground truth only.
            active_thr : Amplitude above which a ground-truth slot counts as active.

        Returns:
            Tuple of aligned (pred, pred_phys, gt, gt_phys) tensors.

        Raises:
            ValueError : If the matching method is unknown.
        """
        if method == ParamMatcher.HUNGARIAN:
            return ParamMatcher._match_hungarian(pred, pred_phys, gt, gt_phys, active_thr)

        if method == ParamMatcher.SORTED_GT:
            return ParamMatcher._match_sorted_gt(pred, pred_phys, gt, gt_phys, active_thr)

        raise ValueError(f"Unknown matching method: {method!r}. Expected {ParamMatcher.HUNGARIAN!r} or {ParamMatcher.SORTED_GT!r}.")


class LegacyParamLoss:
    """Parameter MSE in the two-Gaussian target space of the legacy pipeline."""

    LEGACY_GAUSSIANS = 2
    BOUND_ENTRIES    = 6
    EMPTY_FLOOR      = 1e-8

    @staticmethod
    def _scale(phys: torch.Tensor, bounds_min: tuple, bounds_max: tuple) -> torch.Tensor:
        """Maps physical parameters into the legacy unit box.

        Args:
            phys       : Physical parameters of shape (batch, 2, 3, z, width).
            bounds_min : Six lower bounds ordered amp1, mu1, sigma1, amp2, mu2, sigma2.
            bounds_max : Six upper bounds in the same order.

        Returns:
            Tensor of the same shape scaled to [0, 1] per entry.

        Raises:
            ValueError : If either bound tuple has the wrong length, or if any upper bound does not exceed its lower bound.
        """
        if len(bounds_min) != LegacyParamLoss.BOUND_ENTRIES or len(bounds_max) != LegacyParamLoss.BOUND_ENTRIES:
            raise ValueError(f"legacy bounds need exactly {LegacyParamLoss.BOUND_ENTRIES} entries ordered amp1, mu1, sigma1, amp2, mu2, sigma2; got {len(bounds_min)} min and {len(bounds_max)} max entries.")

        lo = torch.tensor(bounds_min, dtype=phys.dtype, device=phys.device).reshape(1, 2, 3, 1, 1)
        hi = torch.tensor(bounds_max, dtype=phys.dtype, device=phys.device).reshape(1, 2, 3, 1, 1)

        if not torch.all(hi > lo).item():
            raise ValueError(f"legacy bounds require max > min per entry; got min {tuple(bounds_min)} and max {tuple(bounds_max)}.")

        return (phys - lo) / (hi - lo)

    @staticmethod
    def mse(pred_phys: torch.Tensor, gt_phys: torch.Tensor, bounds_min: tuple, bounds_max: tuple) -> torch.Tensor:
        """Returns the parameter MSE computed in the bound-scaled legacy space.

        Args:
            pred_phys  : Predicted physical parameters of shape (batch, 2, 3, z, width).
            gt_phys    : Ground-truth physical parameters of the same shape.
            bounds_min : Six lower bounds ordered amp1, mu1, sigma1, amp2, mu2, sigma2.
            bounds_max : Six upper bounds in the same order.

        Returns:
            Scalar mean squared error over the scaled parameters.

        Raises:
            ValueError : If the prediction does not carry exactly two Gaussian slots.
        """
        if pred_phys.shape[1] != LegacyParamLoss.LEGACY_GAUSSIANS:
            raise ValueError(f"LegacyParamLoss.mse computes MSE in the two-Gaussian legacy target space; got {pred_phys.shape[1]} Gaussian slots.")

        pred = LegacyParamLoss._scale(pred_phys, bounds_min, bounds_max)
        gt   = LegacyParamLoss._scale(gt_phys, bounds_min, bounds_max)

        return ((pred - gt) ** 2).mean()

    @staticmethod
    def _group_mean(sq_err: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Averages squared errors over the pixels a presence mask selects, per parameter."""
        return (sq_err * mask).sum(dim=(0, 2, 3)) / (mask.sum() + LegacyParamLoss.EMPTY_FLOOR)

    @staticmethod
    def masked_mse(pred_phys: torch.Tensor, gt_phys: torch.Tensor, bounds_min: tuple, bounds_max: tuple, amp_thr: float) -> torch.Tensor:
        """Returns the faithful imitation of the legacy masked MSE in the bound-scaled space.

        Slot-2 parameters are supervised only where the model itself predicts
        slot 2 present, meaning the scaled slot-2 amplitude exceeds ``amp_thr``
        and the scaled slot-2 width is non-negative (the legacy a_ths and w_ths
        gates). Slot-1 parameters receive equal-weight group means over the
        predicted-present and predicted-absent pixel groups. False-negative
        slot-2 predictions receive no gradient by design.

        Args:
            pred_phys  : Predicted physical parameters of shape (batch, 2, 3, z, width).
            gt_phys    : Ground-truth physical parameters of the same shape.
            bounds_min : Six lower bounds ordered amp1, mu1, sigma1, amp2, mu2, sigma2.
            bounds_max : Six upper bounds in the same order.
            amp_thr    : Presence threshold applied to the scaled slot-2 amplitude.

        Returns:
            Scalar sum of the per-parameter group-mean squared errors.

        Raises:
            ValueError : If the prediction does not carry exactly two Gaussian slots.
        """
        if pred_phys.shape[1] != LegacyParamLoss.LEGACY_GAUSSIANS:
            raise ValueError(f"LegacyParamLoss.masked_mse imitates the two-Gaussian legacy masked loss; got {pred_phys.shape[1]} Gaussian slots.")

        pred = LegacyParamLoss._scale(pred_phys, bounds_min, bounds_max)
        gt   = LegacyParamLoss._scale(gt_phys, bounds_min, bounds_max)

        present = ((pred[:, 1, 0:1] > amp_thr) & (pred[:, 1, 2:3] >= 0.0)).to(pred.dtype).detach()
        absent  = 1.0 - present

        sq_first  = (pred[:, 0] - gt[:, 0]) ** 2
        sq_second = (pred[:, 1] - gt[:, 1]) ** 2

        loss_first  = LegacyParamLoss._group_mean(sq_first, absent) + LegacyParamLoss._group_mean(sq_first, present)
        loss_second = LegacyParamLoss._group_mean(sq_second, present)

        return (loss_first + loss_second).sum()


class ParamLoss:
    """Weighted regression penalties over matched Gaussian parameter tensors."""

    DENOM_FLOOR = 1e-6
    FRAC_CLAMP  = 1e-3

    @staticmethod
    def presence_scale(active: torch.Tensor, balance: bool, active_weight: float, inactive_weight: float) -> torch.Tensor:
        """Returns per-element weights that trade off active against inactive slots.

        Args:
            active          : Active-slot indicator of shape (batch, gaussians, params, height, width), valued 0 or 1.
            balance         : Whether to derive the weights per slot from the observed active fraction instead of using the supplied constants.
            active_weight   : Weight applied to active slots when balance is False.
            inactive_weight : Weight applied to inactive slots when balance is False.

        Returns:
            Weight tensor broadcastable to the shape of active.
        """
        if balance:
            frac            = active.mean(dim=(0, 2, 3, 4), keepdim=True).clamp(ParamLoss.FRAC_CLAMP, 1.0 - ParamLoss.FRAC_CLAMP)
            active_weight   = 0.5 / frac
            inactive_weight = 0.5 / (1.0 - frac)

        return active * active_weight + (1.0 - active) * inactive_weight

    @staticmethod
    def focal_scale(amp_pred: torch.Tensor, amp_gt: torch.Tensor, gamma: float, delta: float) -> torch.Tensor:
        """Returns focal weights that upweight slots whose amplitude is badly predicted.

        Args:
            amp_pred : Predicted amplitudes of any shape.
            amp_gt   : Ground-truth amplitudes of the same shape.
            gamma    : Focusing exponent; a value of zero disables the term.
            delta    : Amplitude error scale at which the weight reaches one half.

        Returns:
            Detached weight tensor of the same shape as amp_pred.
        """
        if gamma <= 0.0:
            return torch.ones_like(amp_pred)

        abs_diff = (amp_pred - amp_gt).abs().detach()

        return (abs_diff / (abs_diff + delta + ParamLoss.DENOM_FLOOR)) ** gamma

    @staticmethod
    def l1(
        pred        : torch.Tensor,
        gt          : torch.Tensor,
        weights     : torch.Tensor,
        param_names : list[str],
        active_norm : bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns the weighted L1 parameter loss and its per-parameter breakdown.

        Args:
            pred        : Matched predicted parameters of shape (batch, gaussians, params, height, width).
            gt          : Matched ground-truth parameters of the same shape.
            weights     : Per-element weights broadcastable to that shape.
            param_names : Names of the parameter channels along dimension 2, used as keys of the breakdown.
            active_norm : Whether to normalise by the summed weights instead of the element count.

        Returns:
            Tuple of the scalar total loss and a mapping of parameter name to
            its scalar contribution.
        """
        diff          = pred - gt
        weighted_diff = weights * torch.abs(diff)
        total         = ParamLoss._reduce(weighted_diff, weights, active_norm)

        per_param = {
            name: ParamLoss._reduce(weights[:, :, i:i+1] * torch.abs(diff[:, :, i:i+1]), weights[:, :, i:i+1], active_norm)
            for i, name in enumerate(param_names)
            if i < pred.shape[2]
        }
        return total, per_param

    @staticmethod
    def huber(
        pred        : torch.Tensor,
        gt          : torch.Tensor,
        weights     : torch.Tensor,
        delta       : float,
        active_norm : bool = False,
    ) -> torch.Tensor:
        """Returns the weighted Huber parameter loss.

        Args:
            pred        : Matched predicted parameters of shape (batch, gaussians, params, height, width).
            gt          : Matched ground-truth parameters of the same shape.
            weights     : Per-element weights broadcastable to that shape.
            delta       : Magnitude at which the penalty switches from quadratic to linear.
            active_norm : Whether to normalise by the summed weights instead of the element count.

        Returns:
            Scalar loss.
        """
        diff     = pred - gt
        abs_diff = torch.abs(diff)
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))

        return ParamLoss._reduce(weights * val, weights, active_norm)

    @staticmethod
    def mse(
        pred        : torch.Tensor,
        gt          : torch.Tensor,
        weights     : torch.Tensor,
        active_norm : bool = False,
    ) -> torch.Tensor:
        """Returns the weighted squared-error parameter loss.

        Args:
            pred        : Matched predicted parameters of shape (batch, gaussians, params, height, width).
            gt          : Matched ground-truth parameters of the same shape.
            weights     : Per-element weights broadcastable to that shape.
            active_norm : Whether to normalise by the summed weights instead of the element count.

        Returns:
            Scalar loss.
        """
        diff = pred - gt
        val  = diff * diff

        return ParamLoss._reduce(weights * val, weights, active_norm)

    @staticmethod
    def _reduce(weighted: torch.Tensor, weights: torch.Tensor, active_norm: bool) -> torch.Tensor:
        """Returns the weight-normalised sum or the plain mean of a weighted penalty tensor."""
        if active_norm:
            return weighted.sum() / weights.sum().clamp(min=ParamLoss.DENOM_FLOOR)

        return weighted.mean()

    @staticmethod
    def tv(params: torch.Tensor) -> torch.Tensor:
        """Returns the isotropic total variation of a parameter map over its spatial axes.

        Args:
            params : Parameter tensor whose last two dimensions are the spatial height and width axes.

        Returns:
            Scalar sum of the mean absolute differences along both spatial axes.
        """
        dx = torch.abs(params[..., 1:, :] - params[..., :-1, :]).mean()
        dy = torch.abs(params[..., :, 1:] - params[..., :, :-1]).mean()

        return dx + dy
