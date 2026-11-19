
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from tools.data.gaussians import GaussianActivity


class NormalizationAudit:

    OVERFLOW_Z           = 6.0
    EDGE_FRACTION        = 1e-3
    DRIFT_WARN_ABS_MEAN  = 0.5
    DRIFT_WARN_STD_DEV   = 0.5
    SATURATION_WARN_FRAC = 0.01
    NONFINITE_KEY        = re.compile(r"^norm_in_ch(\d+)_nonfinite_frac$")

    def __init__(self, run, result) -> None:

        self.loaded         = run
        self.result         = result
        self.active_amp_thr = float(run.activity_threshold)

    def input_health(self) -> Dict[str, float]:

        normalized = self.loaded.dataset.assemble_window(self.loaded.complex_inputs, self.loaded.dataset.aux)

        out            = {}
        worst_abs_mean = 0.0
        worst_std_dev  = 0.0
        n_overflow     = 0
        n_finite       = 0
        n_nonfinite    = 0
        n_values       = 0

        for c in range(normalized.shape[0]):
            channel = np.asarray(normalized[c], dtype=np.float64)
            finite  = channel[np.isfinite(channel)]

            if not finite.size:
                raise ValueError(f"Normalized input channel {c} holds no finite value; the input stack or normalization stats are degenerate")

            mean      = float(finite.mean())
            std       = float(finite.std())
            overflow  = float((np.abs(finite) > self.OVERFLOW_Z).mean())
            nonfinite = channel.size - finite.size

            out[f"norm_in_ch{c}_mean"]           = mean
            out[f"norm_in_ch{c}_std"]            = std
            out[f"norm_in_ch{c}_overflow_frac"]  = overflow
            out[f"norm_in_ch{c}_nonfinite_frac"] = nonfinite / channel.size

            worst_abs_mean = max(worst_abs_mean, abs(mean))
            worst_std_dev  = max(worst_std_dev, abs(std - 1.0))
            n_overflow    += int((np.abs(finite) > self.OVERFLOW_Z).sum())
            n_finite      += finite.size
            n_nonfinite   += nonfinite
            n_values      += channel.size

        out["norm_in_worst_abs_mean"] = worst_abs_mean
        out["norm_in_worst_std_dev"]  = worst_std_dev
        out["norm_in_overflow_frac"]  = n_overflow / n_finite
        out["norm_in_nonfinite_frac"] = n_nonfinite / n_values

        return out

    def output_health(self) -> Dict[str, float]:

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

        active   = GaussianActivity.is_active(amps, self.active_amp_thr)
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

    @classmethod
    def warning_lines(cls, stats: Dict[str, float]) -> List[str]:

        lines = []

        for key, value in stats.items():
            match = cls.NONFINITE_KEY.match(key)
            if match and value > 0.0:
                lines.append(f"input channel {match.group(1)}: {value * 100.0:.4f}% of its normalised values are NaN or inf and were left out of the drift statistics; check the input stack and the stored normalization statistics of that channel")

        if stats["norm_in_worst_abs_mean"] > cls.DRIFT_WARN_ABS_MEAN:
            lines.append(f"input drift: worst channel |mean| {stats['norm_in_worst_abs_mean']:.3f} exceeds {cls.DRIFT_WARN_ABS_MEAN}; the stored normalization statistics do not centre this split")

        if stats["norm_in_worst_std_dev"] > cls.DRIFT_WARN_STD_DEV:
            lines.append(f"input drift: worst channel |std-1| {stats['norm_in_worst_std_dev']:.3f} exceeds {cls.DRIFT_WARN_STD_DEV}; the stored normalization statistics do not scale this split")

        if stats["norm_in_overflow_frac"] > cls.SATURATION_WARN_FRAC:
            lines.append(f"input overflow: {stats['norm_in_overflow_frac'] * 100.0:.2f}% of the normalised values lie beyond {cls.OVERFLOW_Z:g} standard deviations, above the {cls.SATURATION_WARN_FRAC * 100.0:g}% bound")

        for key, bound in (("clamp_amp_ceil_frac", "amplitude ceiling"), ("clamp_sigma_floor_frac", "sigma floor"), ("clamp_sigma_ceil_frac", "sigma ceiling"), ("clamp_mu_edge_frac", "z-axis edge")):
            if key in stats and np.isfinite(stats[key]) and stats[key] > cls.SATURATION_WARN_FRAC:
                lines.append(f"clamp saturation: {stats[key] * 100.0:.2f}% of the active predicted Gaussians sit on the {bound}, above the {cls.SATURATION_WARN_FRAC * 100.0:g}% bound")

        return lines

    def run(self) -> Dict[str, float]:

        out = self.input_health()

        if self.result.params_pred is not None:
            out.update(self.output_health())

        return out
