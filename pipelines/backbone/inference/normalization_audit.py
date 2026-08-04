from __future__ import annotations

from typing import Dict

import numpy as np

from tools.loss.param_loss import ParamMatcher


class NormalizationAudit:

    OVERFLOW_Z    = 6.0
    EDGE_FRACTION = 1e-3

    def __init__(self, run, result) -> None:
        self.loaded = run
        self.result = result

    def input_health(self) -> Dict[str, float]:
        normalized = self.loaded.dataset.assemble_window(self.loaded.complex_inputs, self.loaded.dataset.dem)

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
        params = np.asarray(self.result.params_pred, dtype=np.float64)
        clamp  = self.loaded.dataset.normalizer.stats.clamp
        n_k    = self.loaded.n_gaussians
        x_min  = float(self.loaded.x_axis[0])
        x_max  = float(self.loaded.x_axis[-1])
        x_step = float(self.loaded.x_axis[1]) - x_min

        sigma_floor = 0.5 * x_step
        sigma_ceil  = 0.5 * (x_max - x_min)
        mu_edge     = self.EDGE_FRACTION * (x_max - x_min)

        amps   = np.stack([params[3 * k]     for k in range(n_k)])
        mus    = np.stack([params[3 * k + 1] for k in range(n_k)])
        sigmas = np.stack([params[3 * k + 2] for k in range(n_k)])

        active   = ParamMatcher.is_active(amps)
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
        out = self.input_health()

        if self.result.params_pred is not None:
            out.update(self.output_health())

        return out
