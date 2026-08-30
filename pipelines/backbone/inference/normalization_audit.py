"""Audits whether inference inputs and outputs sit inside their intended numerical ranges.

On the input side it checks that the normalised channels still have roughly zero mean and
unit standard deviation on this split; on the output side it counts how often predicted
Gaussians are pinned against the amplitude ceiling, the σ bounds or the z-axis edge.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from tools.loss.param_loss import ParamMatcher


class NormalizationAudit:
    """Reports input-normalization drift and output-clamp saturation for one inference run.

    Attributes:
        loaded         : Loaded run supplying the dataset, normalization statistics and z axis.
        result         : Inference result carrying the predicted parameter cube.
        active_amp_thr : Physical amplitude at or above which a predicted Gaussian slot counts as active.
    """

    OVERFLOW_Z    = 6.0
    EDGE_FRACTION = 1e-3

    def __init__(self, run, result, active_amp_thr: float = ParamMatcher.ACTIVE_AMP_THR) -> None:
        """Stores the loaded run, the inference result to audit and the active-amplitude threshold."""

        self.loaded         = run
        self.result         = result
        self.active_amp_thr = float(active_amp_thr)

    def input_health(self) -> Dict[str, float]:
        """Measures how far the normalised input channels drift from zero mean and unit variance.

        Returns:
            Per-channel mean, standard deviation and fraction beyond the overflow z-score,
            plus the worst absolute mean, worst |std − 1| and the pooled overflow fraction.

        Raises:
            ValueError : If any normalised channel holds no finite value.
        """

        normalized = self.loaded.dataset.assemble_window(self.loaded.complex_inputs, self.loaded.dataset.aux)

        out            = {}
        worst_abs_mean = 0.0
        worst_std_dev  = 0.0
        n_overflow     = 0
        n_total        = 0

        for c in range(normalized.shape[0]):
            channel = np.asarray(normalized[c], dtype=np.float64)
            finite  = channel[np.isfinite(channel)]

            if not finite.size:
                raise ValueError(f"Normalized input channel {c} holds no finite value; the input stack or normalization stats are degenerate")

            mean     = float(finite.mean())
            std      = float(finite.std())
            overflow = float((np.abs(finite) > self.OVERFLOW_Z).mean())

            out[f"norm_in_ch{c}_mean"]          = mean
            out[f"norm_in_ch{c}_std"]           = std
            out[f"norm_in_ch{c}_overflow_frac"] = overflow

            worst_abs_mean = max(worst_abs_mean, abs(mean))
            worst_std_dev  = max(worst_std_dev, abs(std - 1.0))
            n_overflow    += int((np.abs(finite) > self.OVERFLOW_Z).sum())
            n_total       += finite.size

        out["norm_in_worst_abs_mean"] = worst_abs_mean
        out["norm_in_worst_std_dev"]  = worst_std_dev
        out["norm_in_overflow_frac"]  = n_overflow / n_total

        return out

    def output_health(self) -> Dict[str, float]:
        """Measures how often active predicted Gaussians sit on their clamp boundaries.

        σ is bounded below by half an z step and above by half the axis span, and μ
        counts as edge-pinned within a thousandth of the axis span of either end.

        Returns:
            The active-Gaussian count and the fractions of active Gaussians at the amplitude
            ceiling, the σ floor, the σ ceiling and the μ axis edge; the fractions are NaN
            when no Gaussian is active.
        """

        params = np.asarray(self.result.params_pred, dtype=np.float64)
        clamp  = self.loaded.dataset.normalizer.stats.clamp
        n_k    = self.loaded.n_gaussians
        x_min  = float(self.loaded.z_axis[0])
        x_max  = float(self.loaded.z_axis[-1])
        x_step = float(self.loaded.z_axis[1]) - x_min

        sigma_floor = 0.5 * x_step
        sigma_ceil  = 0.5 * (x_max - x_min)
        mu_edge     = self.EDGE_FRACTION * (x_max - x_min)

        amps   = np.stack([params[3 * k]     for k in range(n_k)])
        mus    = np.stack([params[3 * k + 1] for k in range(n_k)])
        sigmas = np.stack([params[3 * k + 2] for k in range(n_k)])

        active   = ParamMatcher.is_active(amps, self.active_amp_thr)
        n_active = int(active.sum())

        out = {"clamp_n_active": float(n_active)}

        if n_active == 0:
            out["clamp_amp_ceil_frac"]    = float("nan")
            out["clamp_sigma_floor_frac"] = float("nan")
            out["clamp_sigma_ceil_frac"]  = float("nan")
            out["clamp_mu_edge_frac"]     = float("nan")
            return out

        amp_act   = amps[active]
        mu_act    = mus[active]
        sigma_act = sigmas[active]

        amp_cap = clamp.amp_max if clamp.enabled else None

        out["clamp_amp_ceil_frac"]    = float((amp_act >= amp_cap * (1.0 - self.EDGE_FRACTION)).mean()) if amp_cap else float("nan")
        out["clamp_sigma_floor_frac"] = float((sigma_act <= sigma_floor * (1.0 + self.EDGE_FRACTION)).mean())
        out["clamp_sigma_ceil_frac"]  = float((sigma_act >= sigma_ceil * (1.0 - self.EDGE_FRACTION)).mean())
        out["clamp_mu_edge_frac"]     = float(((mu_act <= x_min + mu_edge) | (mu_act >= x_max - mu_edge)).mean())

        return out

    def run(self) -> Dict[str, float]:
        """Runs the input audit, adding the output-clamp audit when a parameter cube exists.

        Returns:
            The merged scalar metrics of both audits.
        """

        out = self.input_health()

        if self.result.params_pred is not None:
            out.update(self.output_health())

        return out
