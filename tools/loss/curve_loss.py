"""Elementwise and shape-aware penalties over volumetric reflectivity curves.

Collects the pointwise robust penalties applied to a prediction-minus-target
residual, the second difference used as a smoothness regulariser, and the
shape and energy terms comparing whole curves along the z axis.
"""

from __future__ import annotations

import torch


class CurveLoss:
    """Stateless penalties over curve residuals and curve pairs."""

    @staticmethod
    def mse_diff(diff: torch.Tensor) -> torch.Tensor:
        """Returns the mean squared value of a residual tensor of any shape."""
        return (diff * diff).mean()

    @staticmethod
    def l1_diff(diff: torch.Tensor) -> torch.Tensor:
        """Returns the mean absolute value of a residual tensor of any shape."""
        return diff.abs().mean()

    @staticmethod
    def huber_diff(diff: torch.Tensor, delta: float) -> torch.Tensor:
        """Returns the mean Huber penalty of a residual.

        Args:
            diff  : Residual tensor of any shape.
            delta : Magnitude at which the penalty switches from quadratic to linear.

        Returns:
            Scalar mean penalty.
        """
        abs_diff = diff.abs()
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
        return val.mean()

    @staticmethod
    def charbonnier_diff(diff: torch.Tensor, eps: float) -> torch.Tensor:
        """Returns the mean Charbonnier penalty of a residual.

        Args:
            diff : Residual tensor of any shape.
            eps  : Smoothing constant that keeps the square root differentiable at zero.

        Returns:
            Scalar mean penalty.
        """
        return torch.sqrt((diff * diff + eps * eps).clamp(min=eps * eps)).mean()

    @staticmethod
    def second_difference(curve: torch.Tensor) -> torch.Tensor:
        """Returns the discrete second difference along the z axis.

        Args:
            curve : Tensor of shape (batch, z, ...) sampled uniformly along dimension 1.

        Returns:
            Tensor of shape (batch, z - 2, ...) holding the second
            difference at each interior sample.

        Raises:
            ValueError : If fewer than three samples are present along dim 1.
        """
        if curve.shape[1] < 3:
            raise ValueError(f"Second difference needs at least 3 samples along dim 1, got {curve.shape[1]}.")

        return curve[:, 2:] - 2.0 * curve[:, 1:-1] + curve[:, :-2]

    @staticmethod
    def smooth_l1_diff(diff: torch.Tensor, beta: float) -> torch.Tensor:
        """Returns the mean smooth-L1 penalty of a residual.

        Args:
            diff : Residual tensor of any shape.
            beta : Magnitude below which the penalty is quadratic.

        Returns:
            Scalar mean penalty.
        """
        abs_diff = diff.abs()
        val      = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
        return val.mean()

    @staticmethod
    def cosine(pred: torch.Tensor, target: torch.Tensor, axis: int) -> torch.Tensor:
        """Returns the mean cosine distance between curve shapes.

        Curves whose target norm falls below 1e-3 are excluded, so the term
        measures shape agreement only where the target carries energy.

        Args:
            pred   : Predicted curves with the z samples along axis.
            target : Target curves of the same shape as pred.
            axis   : Dimension holding the z samples.

        Returns:
            Scalar mean of 1 - cosine similarity over the valid curves.
        """
        pred_norm   = torch.norm(pred,   dim=axis, keepdim=True)
        target_norm = torch.norm(target, dim=axis, keepdim=True)

        valid = (target_norm > 1e-3).squeeze(axis).float()

        p   = pred   / pred_norm.clamp(min=1e-3)
        t   = target / target_norm.clamp(min=1e-3)
        sim = (p * t).sum(dim=axis).clamp(-1.0, 1.0)
        n   = valid.sum().clamp(min=1.0)

        return ((1.0 - sim) * valid).sum() / n

    @staticmethod
    def log_energy(pred: torch.Tensor, target: torch.Tensor, axis: int) -> torch.Tensor:
        """Returns the mean absolute log-ratio of predicted to target curve energy.

        Curves whose target norm falls below 1e-3 are excluded.

        Args:
            pred   : Predicted curves with the z samples along axis.
            target : Target curves of the same shape as pred.
            axis   : Dimension holding the z samples.

        Returns:
            Scalar mean of |log(||pred|| / ||target||)| over the valid curves.
        """
        pred_norm   = torch.norm(pred,   dim=axis)
        target_norm = torch.norm(target, dim=axis)

        valid = (target_norm > 1e-3).float()

        ratio = torch.log((pred_norm + 1e-6) / (target_norm + 1e-6))
        n     = valid.sum().clamp(min=1.0)

        return (ratio.abs() * valid).sum() / n


class MomentLoss:
    """Moment-matching penalties comparing mass, mean and spread of profile pairs.

    Both penalties average only over the pixels whose integrated target power
    exceeds a floor, and return 0.0 for a batch where no pixel does.
    """

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns the mean of values over the pixels the mask selects, 0.0 when the mask selects none."""
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def moment_sums(curves: torch.Tensor, z_axis: torch.Tensor, dx: float) -> tuple:
        """Returns the zeroth, first and second raw moments of each profile.

        Args:
            curves : Profiles of shape (batch, profile_length, rows, cols).
            z_axis : The z sample positions, of shape (profile_length,).
            dx     : The z sample spacing, in z-axis units.

        Returns:
            Tuple of three tensors of shape (batch, rows, cols) holding the
            integrals over z of the profile, of z times the profile, and of z
            squared times the profile.
        """
        x = z_axis.reshape(1, -1, 1, 1)

        s0 = curves.sum(dim=1) * dx
        s1 = (curves * x).sum(dim=1) * dx
        s2 = (curves * x * x).sum(dim=1) * dx

        return s0, s1, s2

    @staticmethod
    def total_power(pred: torch.Tensor, target: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        """Returns the mean relative error of the integrated profile power.

        Args:
            pred   : Predicted profiles of shape (batch, profile_length, rows, cols).
            target : Target profiles of the same shape.
            dx     : The z sample spacing, in z-axis units.
            floor  : Integrated target power a pixel must exceed to enter the mean; also the lower bound of the denominator.

        Returns:
            Scalar mean of |P_pred - P_target| / P_target over the pixels above the floor, 0.0 when none is.
        """
        p0 = pred.sum(dim=1) * dx
        t0 = target.sum(dim=1) * dx

        mask = (t0 > floor).to(pred.dtype)
        rel  = (p0 - t0).abs() / t0.clamp(min=floor)

        return MomentLoss.masked_mean(rel, mask)

    @staticmethod
    def moments(pred: torch.Tensor, target: torch.Tensor, z_axis: torch.Tensor, dx: float, floor: float, weights: tuple) -> torch.Tensor:
        """Returns the weighted mismatch of profile mass, centroid and spread.

        The centroid and spread terms are expressed as fractions of the z axis
        span, so all three terms are dimensionless.

        Args:
            pred    : Predicted profiles of shape (batch, profile_length, rows, cols).
            target  : Target profiles of the same shape.
            z_axis  : The z sample positions, of shape (profile_length,).
            dx      : The z sample spacing, in z-axis units.
            floor   : Integrated target power a pixel must exceed to enter the mean; also the lower bound of the integrated-power denominators.
            weights : Tuple of (mass, mean, spread) weights, normalised by their sum.

        Returns:
            Scalar mean of the combined moment mismatch over the pixels above the floor, 0.0 when none is.
        """
        p0, p1, p2 = MomentLoss.moment_sums(pred,   z_axis, dx)
        t0, t1, t2 = MomentLoss.moment_sums(target, z_axis, dx)

        x_range = float(z_axis[-1] - z_axis[0])
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

        return MomentLoss.masked_mean(combined, mask)
