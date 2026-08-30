"""Per-pixel example fit figures for a parameter-extraction run.

Samples pixels grouped by their selected model order and, for each, plots the raw
profile, the fit target, the mixture and its components, together with the residual.
"""

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
    """Renders example z-profile fits sampled per selected model order.

    Attributes:
        n_gaussians      : Number of Gaussian slots in the parameter stack.
        logger           : Logger for the plotting report.
        threshold_factor : Peak fraction below which profile bins were zeroed by the fitter.
        truncation_index : Elevation bin above which the profile was truncated by the fitter.
        n_fits_per_k     : Number of example pixels sampled per selected model order.
        amp_threshold    : Amplitude above which a Gaussian component is drawn.
        fig_dpi          : Figure resolution used while rendering.
        save_dpi         : Resolution the PNGs are written at.
    """

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
        """Configures the example fit plotter to mirror the fitter's preprocessing.

        Args:
            n_gaussians      : Number of Gaussian slots in the parameter stack.
            logger           : Logger for the plotting report.
            threshold_factor : Peak fraction below which profile bins were zeroed.
            truncation_index : Elevation bin above which the profile was truncated.
            n_fits_per_k     : Number of example pixels sampled per selected model order.
            amp_threshold    : Amplitude above which a Gaussian component is drawn.
            fig_dpi          : Figure resolution used while rendering.
            save_dpi         : Resolution the PNGs are written at.
        """
        self.n_gaussians      = n_gaussians
        self.logger           = logger
        self.threshold_factor = threshold_factor
        self.truncation_index = truncation_index
        self.n_fits_per_k     = n_fits_per_k
        self.amp_threshold    = amp_threshold
        self.fig_dpi          = fig_dpi
        self.save_dpi         = save_dpi

    def _select_pixels_by_k(self, best_k_map : np.ndarray, r2_map : np.ndarray, seed : int = 42) -> Dict[int, np.ndarray]:
        """Samples example pixel coordinates for each selected model order.

        Args:
            best_k_map : Selected model order of shape (rows, cols).
            r2_map     : Per-pixel R-squared of the same shape; non-finite pixels are skipped.
            seed       : Seed of the sampling generator.

        Returns:
            Mapping from model order to an array of shape (n_selected, 2) holding the
            (rows, cols) indices of the sampled pixels.
        """
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

    def _reconstruct_pixel(self, params : np.ndarray, z_axis : np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Evaluates one pixel's mixture and its individual components.

        Args:
            params      : Parameter vector of shape (3 * n_gaussians,) for a single pixel.
            z_axis : Z axis of shape (z,), in metres.

        Returns:
            Tuple of the mixture of shape (z,) and the per-component profiles.
        """
        return GaussianMixture.evaluate_pixel(params, z_axis, self.n_gaussians)

    def _extract_pixel_profiles(self, cube_path : Path, all_pixels : np.ndarray) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        """Reads the raw and preprocessed z profile of each sampled pixel.

        Args:
            cube_path : Path of the cube of shape (z, row, col).
            all_pixels    : Pixel indices of shape (n_pixels, 2) as (row, col).

        Returns:
            Mapping from (row, col) to the raw amplitude profile and the
            thresholded/truncated fit target, both of shape (z,).
        """
        cube_mmap = np.load(str(cube_path), mmap_mode="r")

        pixel_profiles : Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for row, col in all_pixels.tolist():
            raw                      = np.abs(np.array(cube_mmap[:, row, col])).astype(np.float32)
            processed                = ProfilePreprocessor.apply(raw, self.threshold_factor, self.truncation_index)
            pixel_profiles[(row, col)] = (raw, processed)

        del cube_mmap
        gc.collect()

        return pixel_profiles

    def _plot_pixel_fit(self, z_axis, raw, profile, total, comps, params, comp_colors, k_color, k_label, row, col, r2_val, k_dir) -> Path:
        """Saves one pixel's fit figure with the raw profile, target, mixture and components.

        Args:
            z_axis : Z axis of shape (z,), in metres.
            raw         : Raw amplitude profile of shape (z,).
            profile     : Thresholded/truncated fit target of shape (z,).
            total       : Fitted mixture of shape (z,).
            comps       : Per-component profiles, each of shape (z,).
            params      : Parameter vector of shape (3 * n_gaussians,) for this pixel.
            comp_colors : Colour per Gaussian slot.
            k_color     : Colour used for the mixture curve.
            k_label     : Title fragment naming the selected model order.
            row          : Row index of the pixel.
            col          : Range index of the pixel.
            r2_val      : R-squared of this pixel's fit.
            k_dir       : Directory the figure is written to.

        Returns:
            The saved figure path.
        """
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.plot(z_axis, raw,     color="0.62",    lw=1.0, label="raw profile", zorder=3)
        ax.plot(z_axis, profile, color="black",   lw=1.5, label="fit target", zorder=4)
        ax.plot(z_axis, total,   color=k_color,   lw=1.4, ls="--", label="fit", zorder=5)

        for k, comp in enumerate(comps):
            if float(params[3 * k]) >= self.amp_threshold:
                ax.fill_between(z_axis, comp, alpha=0.20, color=comp_colors[k], zorder=2)
                ax.plot(z_axis, comp, color=comp_colors[k], lw=0.9, alpha=0.85, label=f"$g_{{{k + 1}}}$")

        ax.set_title(f"Example fit — {k_label}\naz={row},  col={col},  $R^2={r2_val:.3f}$", fontsize=10)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"intensity")
        ax.grid(True, which="major", lw=0.25, alpha=0.40)
        ax.legend(fontsize=8, framealpha=0.90, ncol=2)
        fig.tight_layout()

        return self._save(fig, k_dir / f"row{row}_col{col}_fit.png")

    def _plot_pixel_residual(self, z_axis, residual, row, col, r2_val, k_dir) -> Path:
        """Saves one pixel's signed fit residual against z.

        Args:
            z_axis : Z axis of shape (z,), in metres.
            residual    : Fit target minus mixture, of shape (z,).
            row          : Row index of the pixel.
            col          : Range index of the pixel.
            r2_val      : R-squared of this pixel's fit.
            k_dir       : Directory the figure is written to.

        Returns:
            The saved figure path.
        """
        fig, ax = plt.subplots(figsize=(5.6, 2.8))
        ax.plot(z_axis, residual, color="0.35", lw=0.9, zorder=3)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.fill_between(z_axis, residual, 0.0, where=residual >= 0, color="#1f77b4", alpha=0.25, zorder=2)
        ax.fill_between(z_axis, residual, 0.0, where=residual < 0,  color="#d62728", alpha=0.25, zorder=2)

        ax.set_title(f"Fit residual — row={row},  col={col},  $R^2={r2_val:.3f}$", fontsize=10)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\varepsilon = \mathrm{data} - \mathrm{fit}$")
        ax.grid(True, which="major", lw=0.25, alpha=0.40)
        fig.tight_layout()

        return self._save(fig, k_dir / f"row{row}_col{col}_residual.png")

    def _plot_example_fits(
        self,
        parameters_array : np.ndarray,
        pixel_profiles   : Dict[Tuple[int, int], np.ndarray],
        z_axis           : np.ndarray,
        pixels_by_k      : Dict[int, np.ndarray],
        r2_map           : np.ndarray,
        out_dir          : Path,
    ) -> Dict[str, Path]:
        """Saves the fit and residual figures of every sampled pixel, grouped by order.

        Args:
            parameters_array : Parameter stack of shape (3 * n_gaussians, rows, cols).
            pixel_profiles   : Mapping from (rows, cols) to the raw and target profiles.
            height_axis      : Elevation axis of shape (height,), in metres.
            pixels_by_k      : Sampled pixel indices per selected model order.
            r2_map           : Per-pixel R-squared of shape (rows, cols).
            out_dir          : Directory the per-order subdirectories are created under.

        Returns:
            Mapping from figure name to the saved PNG path.
        """
        comp_colors = [cm.tab10(i) for i in range(self.n_gaussians)]
        saved       : Dict[str, Path] = {}

        for K, pixels in pixels_by_k.items():
            if pixels.shape[0] == 0:
                continue

            k_color = cm.tab10((K - 1) % 10)
            k_label = rf"$K^*={K}$  ({K} Gaussian{'s' if K > 1 else ''})"
            k_dir   = out_dir / f"k{K}"
            k_dir.mkdir(parents=True, exist_ok=True)

            for row, col in pixels.tolist():
                entry = pixel_profiles.get((row, col))
                if entry is None:
                    continue

                raw, profile = entry
                raw          = raw.astype(np.float64)
                profile      = profile.astype(np.float64)
                params       = parameters_array[:, row, col].astype(np.float64)
                total, comps = self._reconstruct_pixel(params, z_axis.astype(np.float64))
                residual = profile - total
                r2_val   = float(r2_map[row, col]) if np.isfinite(r2_map[row, col]) else float("nan")

                saved[f"k{K}_row{row}_col{col}_fit"]      = self._plot_pixel_fit(z_axis, raw, profile, total, comps, params, comp_colors, k_color, k_label, row, col, r2_val, k_dir)
                saved[f"k{K}_row{row}_col{col}_residual"] = self._plot_pixel_residual(z_axis, residual, row, col, r2_val, k_dir)

        return saved

    def run(
        self,
        parameters_array : np.ndarray,
        best_k_map       : np.ndarray,
        r2_map           : np.ndarray,
        z_axis           : np.ndarray,
        cube_path        : Path,
        out_dir          : Path,
    ) -> Dict[str, Path]:
        """Samples pixels per model order and renders their fit and residual figures.

        Args:
            parameters_array : Parameter stack of shape (3 * n_gaussians, row, col).
            best_k_map       : Selected model order of shape (row, col).
            r2_map           : Per-pixel R-squared of the same shape.
            z_axis      : Z axis of shape (z,), in metres.
            cube_path    : Path of the cube of shape (z, row, col).
            out_dir          : Directory the per-order subdirectories are created under.

        Returns:
            Mapping from figure name to the saved PNG path.
        """
        self.logger.subsection("Loading cube for example fit plots (memory-mapped)")

        pixels_by_k = self._select_pixels_by_k(best_k_map, r2_map)
        non_empty   = [px for px in pixels_by_k.values() if px.shape[0] > 0]
        all_pixels  = np.concatenate(non_empty, axis=0) if non_empty else np.empty((0, 2), dtype=np.int32)

        self.logger.subsection(f"Extracting {all_pixels.shape[0]} pixel profiles for example fits")
        pixel_profiles = self._extract_pixel_profiles(cube_path, all_pixels)

        self.logger.subsection(f"Plotting example fits  ({self.n_fits_per_k} pixels, up to {self.n_gaussians} K groups)")
        return self._plot_example_fits(parameters_array, pixel_profiles, z_axis, pixels_by_k, r2_map, out_dir)
