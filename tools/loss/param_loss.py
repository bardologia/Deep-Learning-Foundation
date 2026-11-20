"""Slot matching and penalties for Gaussian component parameter predictions.

Set prediction over Gaussian slots is permutation invariant, so predictions are
first assigned to ground-truth slots per pixel, and only then compared with the
weighted regression penalties defined here.
"""

from __future__ import annotations

import itertools

import torch

from tools.data.gaussians import GaussianActivity


class ParamMatcher:
    """Per-pixel assignment of predicted Gaussian slots to ground-truth slots.

    Matching is exhaustive over the G! slot permutations, which bounds the
    usable number of Gaussians at MAX_GAUSSIANS.
    """

    MAX_GAUSSIANS = 6

    HUNGARIAN = "hungarian"
    SORTED_GT = "sorted_gt"

    @staticmethod
    def _sort_gt(gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorts ground-truth slots per pixel by ascending z, inactive slots last.

        Args:
            gt         : Normalised ground-truth parameters of shape (batch, gaussians, params, height, width).
            gt_phys    : Physical-unit ground-truth parameters of the same shape, with amplitude at param index 0 and z mu in metres at param index 1.
            active_thr : Amplitude above which a ground-truth slot counts as active.

        Returns:
            Tuple of the sorted normalised and physical ground-truth tensors.
        """
        gt_phys_amp = gt_phys[:, :, 0]
        gt_mu       = gt_phys[:, :, 1]
        is_active   = GaussianActivity.is_active(gt_phys_amp, active_thr)
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

        active = GaussianActivity.is_active(gt_phys[:, :, 0], active_thr).to(pred.dtype)

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
        """Sorts ground truth by z and then assigns predictions to it by minimum cost."""
        gt, gt_phys     = ParamMatcher._sort_gt(gt, gt_phys, active_thr)
        pred, pred_phys = ParamMatcher._assign_pred_to_gt(pred, pred_phys, gt, gt_phys, active_thr)

        return pred, pred_phys, gt, gt_phys

    @staticmethod
    def _match_sorted_gt(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sorts ground truth by z and leaves the predicted slot order untouched."""
        gt, gt_phys = ParamMatcher._sort_gt(gt, gt_phys, active_thr)

        return pred, pred_phys, gt, gt_phys

    @staticmethod
    def match(
        pred       : torch.Tensor,
        pred_phys  : torch.Tensor,
        gt         : torch.Tensor,
        gt_phys    : torch.Tensor,
        active_thr : float,
        method     : str = "hungarian",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aligns predicted and ground-truth Gaussian slots with the requested strategy.

        Args:
            pred       : Normalised predicted parameters of shape (batch, gaussians, params, height, width).
            pred_phys  : Physical-unit predicted parameters of the same shape.
            gt         : Normalised ground-truth parameters of the same shape.
            gt_phys    : Physical-unit ground-truth parameters of the same shape.
            active_thr : Amplitude above which a ground-truth slot counts as active, as recorded by the label run.
            method     : Either "hungarian" for per-pixel minimum-cost assignment or "sorted_gt" for z-sorted ground truth only.

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


class ParamLoss:
    """Weighted regression penalties over matched Gaussian parameter tensors."""

    DENOM_FLOOR = 1e-6
    FRAC_CLAMP  = 1e-3

    @staticmethod
    def presence_weights(active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the balanced per-slot weights derived from the batch active fraction.

        Each slot k is balanced by its own active fraction f_k over the batch and
        the spatial axes: active elements weigh 0.5 / f_k and inactive ones
        0.5 / (1 - f_k), so both groups carry the same total weight. The
        fraction is clamped to [FRAC_CLAMP, 1 - FRAC_CLAMP] before the division,
        which caps either weight at 0.5 / FRAC_CLAMP for a slot that is almost
        always empty or almost always occupied in the batch.

        Args:
            active : Active-slot indicator of shape (batch, gaussians, params, height, width), valued 0 or 1.

        Returns:
            Tuple of the active weights, the inactive weights and a clamp flag, each of shape (gaussians,); the flag is 1.0 for a slot whose fraction hit the clamp.
        """
        frac    = active.mean(dim=(0, 2, 3, 4))
        clamped = ((frac < ParamLoss.FRAC_CLAMP) | (frac > 1.0 - ParamLoss.FRAC_CLAMP)).to(active.dtype)
        frac    = frac.clamp(ParamLoss.FRAC_CLAMP, 1.0 - ParamLoss.FRAC_CLAMP)

        return 0.5 / frac, 0.5 / (1.0 - frac), clamped

    @staticmethod
    def presence_scale(active: torch.Tensor, balance: bool, active_weight: float, inactive_weight: float) -> torch.Tensor:
        """Returns per-element weights that trade off active against inactive slots.

        Args:
            active          : Active-slot indicator of shape (batch, gaussians, params, height, width), valued 0 or 1.
            balance         : Whether to derive the weights per slot from the observed active fraction through presence_weights instead of using the supplied constants.
            active_weight   : Weight applied to active slots when balance is False.
            inactive_weight : Weight applied to inactive slots when balance is False.

        Returns:
            Weight tensor broadcastable to the shape of active.
        """
        if balance:
            active_weight, inactive_weight, _clamped = ParamLoss.presence_weights(active)

            active_weight   = active_weight.reshape(1, -1, 1, 1, 1)
            inactive_weight = inactive_weight.reshape(1, -1, 1, 1, 1)

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
