from __future__ import annotations

import itertools

import torch


class ParamMatcher:
    MAX_GAUSSIANS  = 6
    ACTIVE_AMP_THR = 1e-4

    HUNGARIAN = "hungarian"
    SORTED_GT = "sorted_gt"

    @staticmethod
    def is_active(amplitude):
        return amplitude >= ParamMatcher.ACTIVE_AMP_THR

    @staticmethod
    def _sort_gt(gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor]:
        gt_phys_amp = gt_phys[:, :, 0]
        gt_mu       = gt_phys[:, :, 1]
        is_active   = gt_phys_amp > active_thr
        sort_key    = torch.where(is_active, gt_mu, torch.full_like(gt_mu, float("inf")))
        gt_index    = torch.argsort(sort_key, dim=1, stable=True)
        gt_idx_b    = gt_index[:, :, None, :, :].expand_as(gt)

        return torch.gather(gt, dim=1, index=gt_idx_b), torch.gather(gt_phys, dim=1, index=gt_idx_b)

    @staticmethod
    def _assign_pred_to_gt(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor]:
        B, G, P, H, W = pred.shape

        if gt.shape[1] != G:
            raise ValueError(f"ParamMatcher.match requires equal pred/gt gaussian counts, got {G} and {gt.shape[1]}")

        if G > ParamMatcher.MAX_GAUSSIANS:
            raise ValueError(f"ParamMatcher.match enumerates G! permutations; G={G} exceeds MAX_GAUSSIANS={ParamMatcher.MAX_GAUSSIANS}")

        active = (gt_phys[:, :, 0] > active_thr).to(pred.dtype)

        pred_e = pred.permute(0, 3, 4, 1, 2)[:, :, :, :, None, :]
        gt_e   = gt.permute(  0, 3, 4, 1, 2)[:, :, :, None, :, :]
        cost   = (pred_e - gt_e).abs().sum(-1) * active.permute(0, 2, 3, 1)[:, :, :, None, :]

        perms      = list(itertools.permutations(range(G)))
        gt_arange  = torch.arange(G, device=pred.device)
        perm_costs = []

        for perm in perms:
            pidx = torch.tensor(perm, device=pred.device)
            perm_costs.append(cost[:, :, :, pidx, gt_arange].sum(-1))

        best   = torch.stack(perm_costs, dim=-1).argmin(dim=-1)
        perm_t = torch.tensor(perms, device=pred.device)
        chosen = perm_t[best].permute(0, 3, 1, 2)
        idx_b  = chosen[:, :, None, :, :].expand(B, G, P, H, W)

        return torch.gather(pred, dim=1, index=idx_b), torch.gather(pred_phys, dim=1, index=idx_b)

    @staticmethod
    def _match_hungarian(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gt, gt_phys     = ParamMatcher._sort_gt(gt, gt_phys, active_thr)
        pred, pred_phys = ParamMatcher._assign_pred_to_gt(pred, pred_phys, gt, gt_phys, active_thr)

        return pred, pred_phys, gt, gt_phys

    @staticmethod
    def _match_sorted_gt(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor, active_thr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if method == ParamMatcher.HUNGARIAN:
            return ParamMatcher._match_hungarian(pred, pred_phys, gt, gt_phys, active_thr)

        if method == ParamMatcher.SORTED_GT:
            return ParamMatcher._match_sorted_gt(pred, pred_phys, gt, gt_phys, active_thr)

        raise ValueError(f"Unknown matching method: {method!r}. Expected {ParamMatcher.HUNGARIAN!r} or {ParamMatcher.SORTED_GT!r}.")


class LegacyParamLoss:
    LEGACY_GAUSSIANS = 2
    BOUND_ENTRIES    = 6

    @staticmethod
    def _scale(phys: torch.Tensor, bounds_min: tuple, bounds_max: tuple) -> torch.Tensor:
        if len(bounds_min) != LegacyParamLoss.BOUND_ENTRIES or len(bounds_max) != LegacyParamLoss.BOUND_ENTRIES:
            raise ValueError(f"legacy bounds need exactly {LegacyParamLoss.BOUND_ENTRIES} entries ordered amp1, mu1, sigma1, amp2, mu2, sigma2; got {len(bounds_min)} min and {len(bounds_max)} max entries.")

        lo = torch.tensor(bounds_min, dtype=phys.dtype, device=phys.device).reshape(1, 2, 3, 1, 1)
        hi = torch.tensor(bounds_max, dtype=phys.dtype, device=phys.device).reshape(1, 2, 3, 1, 1)

        if not torch.all(hi > lo).item():
            raise ValueError(f"legacy bounds require max > min per entry; got min {tuple(bounds_min)} and max {tuple(bounds_max)}.")

        return (phys - lo) / (hi - lo)

    @staticmethod
    def mse(pred_phys: torch.Tensor, gt_phys: torch.Tensor, bounds_min: tuple, bounds_max: tuple) -> torch.Tensor:
        if pred_phys.shape[1] != LegacyParamLoss.LEGACY_GAUSSIANS:
            raise ValueError(f"LegacyParamLoss.mse computes MSE in the two-Gaussian legacy target space; got {pred_phys.shape[1]} Gaussian slots.")

        pred = LegacyParamLoss._scale(pred_phys, bounds_min, bounds_max)
        gt   = LegacyParamLoss._scale(gt_phys, bounds_min, bounds_max)

        return ((pred - gt) ** 2).mean()


class ParamLoss:
    DENOM_FLOOR = 1e-6
    FRAC_CLAMP  = 1e-3

    @staticmethod
    def presence_scale(active: torch.Tensor, balance: bool, active_weight: float, inactive_weight: float) -> torch.Tensor:
        if balance:
            frac            = active.mean(dim=(0, 2, 3, 4), keepdim=True).clamp(ParamLoss.FRAC_CLAMP, 1.0 - ParamLoss.FRAC_CLAMP)
            active_weight   = 0.5 / frac
            inactive_weight = 0.5 / (1.0 - frac)

        return active * active_weight + (1.0 - active) * inactive_weight

    @staticmethod
    def focal_scale(amp_pred: torch.Tensor, amp_gt: torch.Tensor, gamma: float, delta: float) -> torch.Tensor:
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
        diff = pred - gt
        val  = diff * diff

        return ParamLoss._reduce(weights * val, weights, active_norm)

    @staticmethod
    def _reduce(weighted: torch.Tensor, weights: torch.Tensor, active_norm: bool) -> torch.Tensor:
        if active_norm:
            return weighted.sum() / weights.sum().clamp(min=ParamLoss.DENOM_FLOOR)

        return weighted.mean()

    @staticmethod
    def tv(params: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(params[..., 1:, :] - params[..., :-1, :]).mean()
        dy = torch.abs(params[..., :, 1:] - params[..., :, :-1]).mean()

        return dx + dy
