from __future__ import annotations

import torch

from torch.utils.checkpoint import checkpoint


class PhysicalLoss:
    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def moment_sums(curves: torch.Tensor, x_axis: torch.Tensor, dx: float) -> tuple:
        x = x_axis.reshape(1, -1, 1, 1)

        s0 = curves.sum(dim=1) * dx
        s1 = (curves * x).sum(dim=1) * dx
        s2 = (curves * x * x).sum(dim=1) * dx

        return s0, s1, s2

    @staticmethod
    def total_power(pred: torch.Tensor, target: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        p0 = pred.sum(dim=1) * dx
        t0 = target.sum(dim=1) * dx

        mask = (t0 > floor).to(pred.dtype)
        rel  = (p0 - t0).abs() / t0.clamp(min=floor)

        return PhysicalLoss.masked_mean(rel, mask)

    @staticmethod
    def moments(pred: torch.Tensor, target: torch.Tensor, x_axis: torch.Tensor, dx: float, floor: float, weights: tuple) -> torch.Tensor:
        p0, p1, p2 = PhysicalLoss.moment_sums(pred,   x_axis, dx)
        t0, t1, t2 = PhysicalLoss.moment_sums(target, x_axis, dx)

        x_range = float(x_axis[-1] - x_axis[0])
        mask    = (t0 > floor).to(pred.dtype)

        p0c = p0.clamp(min=floor)
        t0c = t0.clamp(min=floor)

        p_mean = p1 / p0c
        t_mean = t1 / t0c

        p_var = (p2 / p0c - p_mean * p_mean).clamp(min=0.0)
        t_var = (t2 / t0c - t_mean * t_mean).clamp(min=0.0)

        p_spread = torch.sqrt(p_var + 1e-8)
        t_spread = torch.sqrt(t_var + 1e-8)

        mass_term   = (p0 - t0).abs() / t0c
        mean_term   = (p_mean - t_mean).abs() / x_range
        spread_term = (p_spread - t_spread).abs() / x_range

        w_mass, w_mean, w_spread = weights
        w_sum                    = max(w_mass + w_mean + w_spread, 1e-12)

        combined = (w_mass * mass_term + w_mean * mean_term + w_spread * spread_term) / w_sum

        return PhysicalLoss.masked_mean(combined, mask)

    @staticmethod
    def coherence_resynthesis(pred: torch.Tensor, target: torch.Tensor, steering: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        p0 = pred.sum(dim=1) * dx
        t0 = target.sum(dim=1) * dx

        mask = (t0 > floor).to(pred.dtype)

        gp = torch.einsum("nk,bkhw->bnhw", steering, pred.to(steering.dtype)) * dx
        gt = torch.einsum("nk,bkhw->bnhw", steering, target.to(steering.dtype)) * dx

        gp = gp / p0.clamp(min=floor).unsqueeze(1)
        gt = gt / t0.clamp(min=floor).unsqueeze(1)

        val = ((gp - gt).abs() ** 2).mean(dim=1)

        return PhysicalLoss.masked_mean(val, mask)

    @staticmethod
    def covariance_matching(pred: torch.Tensor, target: torch.Tensor, outer: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        delta = torch.einsum("ijk,bkhw->bijhw", outer, (pred - target).to(outer.dtype)) * dx
        ref   = torch.einsum("ijk,bkhw->bijhw", outer, target.to(outer.dtype)) * dx

        num = (delta.abs() ** 2).sum(dim=(1, 2))
        den = (ref.abs() ** 2).sum(dim=(1, 2)).clamp(min=1e-12)

        return PhysicalLoss.masked_mean(num / den, mask)

    @staticmethod
    def capon_cycle(pred: torch.Tensor, target: torch.Tensor, steering: torch.Tensor, outer: torch.Tensor, dx: float, loading: float, floor: float) -> torch.Tensor:
        n_tracks = steering.shape[0]

        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        cov   = torch.einsum("ijk,bkhw->bhwij", outer, pred.to(outer.dtype)) * dx
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).real / n_tracks
        eye   = torch.eye(n_tracks, dtype=cov.dtype, device=cov.device)
        cov   = cov + (loading * trace.clamp(min=floor)).unsqueeze(-1).unsqueeze(-1) * eye

        cov = 0.5 * (cov + cov.conj().transpose(-2, -1))
        cov = cov.to(torch.complex64)

        steer  = steering.to(torch.complex64)
        solved = torch.linalg.solve(cov, steer)
        denom  = torch.einsum("ik,bhwik->bhwk", steer.conj(), solved).real
        spec   = (1.0 / denom.clamp(min=1e-12)).permute(0, 3, 1, 2)

        p0     = spec.sum(dim=1) * dx
        spec_n = spec / p0.clamp(min=floor).unsqueeze(1)
        targ_n = target / t0.clamp(min=floor).unsqueeze(1)

        val = ((spec_n - targ_n) ** 2).mean(dim=1)

        return PhysicalLoss.masked_mean(val, mask)

    @staticmethod
    def _track_steering(kz_track: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
        phase = kz_track.unsqueeze(1) * x_axis.reshape(1, -1, 1, 1)

        return torch.polar(torch.ones_like(phase), phase)

    @staticmethod
    def synthesise_track(curves: torch.Tensor, kz_track: torch.Tensor, x_axis: torch.Tensor, dx: float) -> torch.Tensor:
        steering = PhysicalLoss._track_steering(kz_track, x_axis)

        return (steering * curves.to(steering.dtype)).sum(dim=1) * dx

    @staticmethod
    def coherence_resynthesis_pp_map(pred: torch.Tensor, target: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float, floor: float) -> tuple:
        n_tracks = kz_map.shape[1]

        p0   = pred.sum(dim=1) * dx
        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        p0c = p0.clamp(min=floor)
        t0c = t0.clamp(min=floor)

        val = torch.zeros_like(p0)

        for track in range(n_tracks):
            gp = PhysicalLoss.synthesise_track(pred,   kz_map[:, track], x_axis, dx) / p0c
            gt = PhysicalLoss.synthesise_track(target, kz_map[:, track], x_axis, dx) / t0c

            val = val + (gp - gt).abs() ** 2

        return val / n_tracks, mask

    @staticmethod
    def coherence_resynthesis_pp(pred: torch.Tensor, target: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        val, mask = PhysicalLoss.coherence_resynthesis_pp_map(pred, target, kz_map, x_axis, dx, floor)

        return PhysicalLoss.masked_mean(val, mask)

    @staticmethod
    def _synthesise_pair(curves: torch.Tensor, kz_difference: torch.Tensor, x_axis: torch.Tensor, dx: float) -> torch.Tensor:
        phase    = kz_difference.unsqueeze(1) * x_axis.reshape(1, -1, 1, 1)
        rotation = torch.polar(torch.ones_like(phase), phase)

        return (rotation * curves.to(rotation.dtype)).sum(dim=1) * dx

    @staticmethod
    def covariance_matching_pp_map(pred: torch.Tensor, target: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float, floor: float) -> tuple:
        n_tracks = kz_map.shape[1]

        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        diff = (pred - target)

        num = torch.zeros_like(t0)
        den = torch.zeros_like(t0)

        for i in range(n_tracks):
            for j in range(i, n_tracks):
                kz_difference = kz_map[:, i] - kz_map[:, j]
                weight        = 1.0 if i == j else 2.0

                delta = checkpoint(PhysicalLoss._synthesise_pair, diff,   kz_difference, x_axis, dx, use_reentrant=False)
                ref   = checkpoint(PhysicalLoss._synthesise_pair, target, kz_difference, x_axis, dx, use_reentrant=False)

                num = num + weight * delta.abs() ** 2
                den = den + weight * ref.abs() ** 2

        return num / den.clamp(min=1e-12), mask

    @staticmethod
    def covariance_matching_pp(pred: torch.Tensor, target: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        val, mask = PhysicalLoss.covariance_matching_pp_map(pred, target, kz_map, x_axis, dx, floor)

        return PhysicalLoss.masked_mean(val, mask)

    @staticmethod
    def _synthesise_covariance(curves: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float) -> torch.Tensor:
        B, n_tracks, H, W = kz_map.shape

        cov = torch.zeros((B, H, W, n_tracks, n_tracks), dtype=torch.complex64, device=kz_map.device)

        for i in range(n_tracks):
            for j in range(i, n_tracks):
                entry = PhysicalLoss._synthesise_pair(curves, kz_map[:, i] - kz_map[:, j], x_axis, dx).to(torch.complex64)

                cov[..., i, j] = entry
                if j != i:
                    cov[..., j, i] = entry.conj()

        return cov

    @staticmethod
    def capon_cycle_pp(pred: torch.Tensor, target: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float, loading: float, floor: float) -> torch.Tensor:
        B, n_tracks, H, W = kz_map.shape

        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        cov   = PhysicalLoss._synthesise_covariance(pred, kz_map, x_axis, dx)
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).real / n_tracks
        eye   = torch.eye(n_tracks, dtype=cov.dtype, device=cov.device)
        cov   = cov + (loading * trace.clamp(min=floor)).unsqueeze(-1).unsqueeze(-1) * eye

        cov = 0.5 * (cov + cov.conj().transpose(-2, -1))

        spec = torch.empty((B, x_axis.shape[0], H, W), dtype=pred.dtype, device=pred.device)

        for k in range(x_axis.shape[0]):
            phase  = kz_map * x_axis[k]
            steer  = torch.polar(torch.ones_like(phase), phase).to(torch.complex64).permute(0, 2, 3, 1).unsqueeze(-1)
            solved = torch.linalg.solve(cov, steer)
            denom  = (steer.conj() * solved).sum(dim=-2).squeeze(-1).real

            spec[:, k] = 1.0 / denom.clamp(min=1e-12)

        p0     = spec.sum(dim=1) * dx
        spec_n = spec / p0.clamp(min=floor).unsqueeze(1)
        targ_n = target / t0.clamp(min=floor).unsqueeze(1)

        val = ((spec_n - targ_n) ** 2).mean(dim=1)

        return PhysicalLoss.masked_mean(val, mask)
