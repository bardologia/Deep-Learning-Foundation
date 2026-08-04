from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
import numpy             as np

from tools.reporting.plotting import PlotBase
from tools.data.preprocessing import ProfilePreprocessor
from tools.data.gaussians     import GaussianMixture
from tools.monitoring.logger  import Logger


class ExampleFitPlotter(PlotBase):
    def __init__(
        self,
        n_gaussians      : int,
        logger           : Logger,
        threshold_factor : float,
        truncation_index : int,
        n_fits_per_k     : int,
        amp_threshold    : float,
        fig_dpi          : int = 150,
        save_dpi         : int = 300,
    ) -> None:
        self.n_gaussians      = n_gaussians
        self.logger           = logger
        self.threshold_factor = threshold_factor
        self.truncation_index = truncation_index
        self.n_fits_per_k     = n_fits_per_k
        self.amp_threshold    = amp_threshold
        self.fig_dpi          = fig_dpi
        self.save_dpi         = save_dpi

    def _select_pixels_by_k(self, best_k_map : np.ndarray, r2_map : np.ndarray, seed : int = 42) -> Dict[int, np.ndarray]:
        rng    = np.random.default_rng(seed)
        flat_k = best_k_map.reshape(-1)
        flat_r = r2_map.reshape(-1)
        H, W   = best_k_map.shape
        finite = np.isfinite(flat_r)

        groups : Dict[int, np.ndarray] = {}

        for K in range(1, self.n_gaussians + 1):
            idx = np.where(finite & (flat_k == K))[0]

            if idx.size == 0:
                groups[K] = np.empty((0, 2), dtype=np.int32)
                continue

            chosen    = rng.choice(idx, size=min(self.n_fits_per_k, idx.size), replace=False)
            groups[K] = np.stack([(chosen // W).astype(np.int32), (chosen % W).astype(np.int32)], axis=1)

        return groups

    def _reconstruct_pixel(self, params : np.ndarray, height_axis : np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        return GaussianMixture.evaluate_pixel(params, height_axis, self.n_gaussians)

    def _extract_pixel_profiles(self, tomogram_path : Path, all_pixels : np.ndarray) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r")

        pixel_profiles : Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for az, rg in all_pixels.tolist():
            raw                      = np.abs(np.array(tomogram_mmap[:, az, rg])).astype(np.float32)
            processed                = ProfilePreprocessor.apply(raw, self.threshold_factor, self.truncation_index)
            pixel_profiles[(az, rg)] = (raw, processed)

        del tomogram_mmap
        gc.collect()

        return pixel_profiles

    def _plot_pixel_fit(self, height_axis, raw, profile, total, comps, params, comp_colors, k_color, k_label, az, rg, r2_val, k_dir) -> Path:
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.plot(height_axis, raw,     color="0.62",    lw=1.0, label="raw profile", zorder=3)
        ax.plot(height_axis, profile, color="black",   lw=1.5, label="fit target", zorder=4)
        ax.plot(height_axis, total,   color=k_color,   lw=1.4, ls="--", label="fit", zorder=5)

        for k, comp in enumerate(comps):
            if float(params[3 * k]) >= self.amp_threshold:
                ax.fill_between(height_axis, comp, alpha=0.20, color=comp_colors[k], zorder=2)
                ax.plot(height_axis, comp, color=comp_colors[k], lw=0.9, alpha=0.85, label=f"$g_{{{k + 1}}}$")

        ax.set_title(f"Example fit — {k_label}\naz={az},  rg={rg},  $R^2={r2_val:.3f}$", fontsize=10)
        ax.set_xlabel(r"height $h$ [m]")
        ax.set_ylabel(r"backscatter intensity")
        ax.grid(True, which="major", lw=0.25, alpha=0.40)
        ax.legend(fontsize=8, framealpha=0.90, ncol=2)
        fig.tight_layout()

        return self._save(fig, k_dir / f"az{az}_rg{rg}_fit.png")

    def _plot_pixel_residual(self, height_axis, residual, az, rg, r2_val, k_dir) -> Path:
        fig, ax = plt.subplots(figsize=(5.6, 2.8))
        ax.plot(height_axis, residual, color="0.35", lw=0.9, zorder=3)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.fill_between(height_axis, residual, 0.0, where=residual >= 0, color="#1f77b4", alpha=0.25, zorder=2)
        ax.fill_between(height_axis, residual, 0.0, where=residual < 0,  color="#d62728", alpha=0.25, zorder=2)

        ax.set_title(f"Fit residual — az={az},  rg={rg},  $R^2={r2_val:.3f}$", fontsize=10)
        ax.set_xlabel(r"height $h$ [m]")
        ax.set_ylabel(r"$\varepsilon = \mathrm{data} - \mathrm{fit}$")
        ax.grid(True, which="major", lw=0.25, alpha=0.40)
        fig.tight_layout()

        return self._save(fig, k_dir / f"az{az}_rg{rg}_residual.png")

    def _plot_example_fits(
        self,
        parameters_array : np.ndarray,
        pixel_profiles   : Dict[Tuple[int, int], np.ndarray],
        height_axis      : np.ndarray,
        pixels_by_k      : Dict[int, np.ndarray],
        r2_map           : np.ndarray,
        out_dir          : Path,
    ) -> Dict[str, Path]:
        comp_colors = [cm.tab10(i) for i in range(self.n_gaussians)]
        saved       : Dict[str, Path] = {}

        for K, pixels in pixels_by_k.items():
            if pixels.shape[0] == 0:
                continue

            k_color = cm.tab10((K - 1) % 10)
            k_label = rf"$K^*={K}$  ({K} Gaussian{'s' if K > 1 else ''})"
            k_dir   = out_dir / f"k{K}"
            k_dir.mkdir(parents=True, exist_ok=True)

            for az, rg in pixels.tolist():
                entry = pixel_profiles.get((az, rg))
                if entry is None:
                    continue

                raw, profile = entry
                raw          = raw.astype(np.float64)
                profile      = profile.astype(np.float64)
                params       = parameters_array[:, az, rg].astype(np.float64)
                total, comps = self._reconstruct_pixel(params, height_axis.astype(np.float64))
                residual = profile - total
                r2_val   = float(r2_map[az, rg]) if np.isfinite(r2_map[az, rg]) else float("nan")

                saved[f"k{K}_az{az}_rg{rg}_fit"]      = self._plot_pixel_fit(height_axis, raw, profile, total, comps, params, comp_colors, k_color, k_label, az, rg, r2_val, k_dir)
                saved[f"k{K}_az{az}_rg{rg}_residual"] = self._plot_pixel_residual(height_axis, residual, az, rg, r2_val, k_dir)

        return saved

    def run(
        self,
        parameters_array : np.ndarray,
        best_k_map       : np.ndarray,
        r2_map           : np.ndarray,
        height_axis      : np.ndarray,
        tomogram_path    : Path,
        out_dir          : Path,
    ) -> Dict[str, Path]:
        self.logger.subsection("Loading tomogram for example fit plots (memory-mapped)")

        pixels_by_k = self._select_pixels_by_k(best_k_map, r2_map)
        non_empty   = [px for px in pixels_by_k.values() if px.shape[0] > 0]
        all_pixels  = np.concatenate(non_empty, axis=0) if non_empty else np.empty((0, 2), dtype=np.int32)

        self.logger.subsection(f"Extracting {all_pixels.shape[0]} pixel profiles for example fits")
        pixel_profiles = self._extract_pixel_profiles(tomogram_path, all_pixels)

        self.logger.subsection(f"Plotting example fits  ({self.n_fits_per_k} pixels, up to {self.n_gaussians} K groups)")
        return self._plot_example_fits(parameters_array, pixel_profiles, height_axis, pixels_by_k, r2_map, out_dir)
