"""Gaussian-mixture z profiles in torch: reconstruction and clamping.

The torch half of the Gaussian toolbox, kept apart from the NumPy half so that
CPU-only consumers such as the dataset synthesis never pull torch into their
process. Both halves share the floors and ceilings declared on `GaussianMixture`.
"""
from __future__ import annotations

import torch

from tools.data.gaussians import GaussianMixture


class GaussianCurve:
    """Differentiable reconstruction of z curves from predicted parameter maps."""
    @staticmethod
    def reconstruct(params: torch.Tensor, z_axis: torch.Tensor, ppg: int = 3) -> torch.Tensor:
        """Reconstructs the z curve of every pixel from a parameter map.

        Args:
            params : Parameter map of shape (batch, ppg*n_gaussians, rows, cols), channel-interleaved as amplitude, mean in metres, width in metres.
            z_axis : Z positions in metres of shape (n_bins,).
            ppg    : Parameters per Gaussian.

        Returns:
            Reconstructed curves of shape (batch, n_bins, rows, cols).

        Raises:
            ValueError : If the channel count is not a multiple of ppg.
        """
        B, C, H, W = params.shape

        if C % ppg != 0:
            raise ValueError(f"Gaussian param channels ({C}) must be divisible by {ppg}")

        n_gaussians = C // ppg
        p           = params.reshape(B, n_gaussians, ppg, H, W)

        a   = p[:, :, 0, :, :]
        mu  = p[:, :, 1, :, :]
        sig = p[:, :, 2, :, :]
        x   = z_axis.reshape(1, -1, 1, 1)

        curves = torch.zeros((B, x.shape[1], H, W), dtype=params.dtype, device=params.device)

        for g in range(n_gaussians):
            a_g   = a[:, g:g+1, :, :]
            mu_g  = mu[:, g:g+1, :, :]
            sig_g = sig[:, g:g+1, :, :].clamp(min=GaussianMixture.SIGMA_FLOOR)

            sig2_g     = sig_g ** 2
            exponent_g = (-((x - mu_g) ** 2) / (2.0 * sig2_g)).clamp(GaussianMixture.EXPON_FLOOR, GaussianMixture.EXPON_CEIL)
            curves     = curves + a_g * torch.exp(exponent_g)

        return curves


class GaussianClamp:
    """Constrains predicted Gaussian parameters to the physical z axis."""
    @staticmethod
    def apply(
        params      : torch.Tensor,
        z_axis      : torch.Tensor,
        amp_max     : float,
        ppg         : int   = 3,
        leaky_slope : float = 0.0,
    ) -> torch.Tensor:
        """Clamps amplitudes, means and widths of a predicted parameter map.

        Means are held inside the z axis, widths between half a bin and half
        the axis span, and amplitudes between zero and amp_max.

        Args:
            params      : Parameter map of shape (batch, ppg*n_gaussians, rows, cols).
            z_axis      : Z positions in metres of shape (n_bins,).
            amp_max     : Largest permitted amplitude.
            ppg         : Parameters per Gaussian.
            leaky_slope : Slope kept outside the bounds; 0.0 gives a hard clamp.

        Returns:
            Clamped parameter map of the same shape as params.
        """
        x_min   = z_axis.min()
        x_max   = z_axis.max()
        x_step  = (x_max - x_min) / (z_axis.shape[0] - 1)
        x_range = x_max - x_min
        n_gauss = params.shape[1] // ppg
        slices  = []

        def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            if leaky_slope > 0.0:
                clamped = x.clamp(lo, hi)
                return clamped + leaky_slope * (x - clamped)
            return x.clamp(lo, hi)

        for _g in range(n_gauss):
            _b = _g * ppg
            slices.append(_clamp(params[:, _b + 0:_b + 1], 0.0,          amp_max      ))
            slices.append(_clamp(params[:, _b + 1:_b + 2], x_min,        x_max        ))
            slices.append(_clamp(params[:, _b + 2:_b + 3], x_step * 0.5, x_range * 0.5))

        return torch.cat(slices, dim=1)
