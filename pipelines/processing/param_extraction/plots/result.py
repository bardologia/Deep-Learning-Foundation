from __future__ import annotations

from pathlib import Path
from typing  import Dict

import matplotlib

matplotlib.use("Agg")
import numpy as np

from tools.reporting.plotting                                  import PlotBase
from tools.monitoring.logger                                   import Logger
from pipelines.processing.param_extraction.plots.spatial       import SpatialMapPlotter
from pipelines.processing.param_extraction.plots.distributions import DistributionPlotter
from pipelines.processing.param_extraction.plots.metrics       import MetricsBarPlotter
from pipelines.processing.param_extraction.plots.examples      import ExampleFitPlotter


class FittingResultPlotter(PlotBase):
    def __init__(
        self,
        output_directory : Path,
        n_gaussians      : int,
        logger           : Logger,
        threshold_factor : float,
        truncation_index : int,
        fig_dpi          : int   = 150,
        save_dpi         : int   = 300,
        n_fits_per_k     : int   = 5,
        amp_threshold    : float = 1e-3,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.n_gaussians      = n_gaussians
        self.logger           = logger
        self.threshold_factor = threshold_factor
        self.truncation_index = truncation_index
        self.fig_dpi          = fig_dpi
        self.save_dpi         = save_dpi
        self.n_fits_per_k     = n_fits_per_k
        self.amp_threshold    = amp_threshold
        self._images_dir      = self.output_directory / "images"

        self.spatial_plotter      = SpatialMapPlotter(n_gaussians=n_gaussians, logger=logger, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.distribution_plotter = DistributionPlotter(n_gaussians=n_gaussians, logger=logger, amp_threshold=amp_threshold, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.metrics_bar_plotter  = MetricsBarPlotter(n_gaussians=n_gaussians, logger=logger, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.example_fit_plotter  = ExampleFitPlotter(
            n_gaussians      = n_gaussians,
            logger           = logger,
            threshold_factor = threshold_factor,
            truncation_index = truncation_index,
            n_fits_per_k     = n_fits_per_k,
            amp_threshold    = amp_threshold,
            fig_dpi          = fig_dpi,
            save_dpi         = save_dpi,
        )

    def _setup_output_dirs(self) -> Dict[str, Path]:
        dirs = {
            "colormaps"    : self._images_dir / "colormaps",
            "example_fits" : self._images_dir / "example_fits",
            "distributions": self._images_dir / "distributions",
            "metrics"      : self._images_dir / "metrics",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def _run_spatial_maps(self, metrics_dict : dict, r2_map : np.ndarray, activity_map : np.ndarray, dirs : Dict[str, Path]) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        self.logger.subsection("Plotting active-Gaussian count colormap")
        saved["n_gaussians_map"] = self.spatial_plotter._plot_discrete_k_map(
            activity_map,
            rf"Number of active Gaussians per pixel  ($A_k \geq {self.amp_threshold:g}$)",
            r"active Gaussians $K$",
            dirs["colormaps"] / "n_gaussians_map.png",
        )

        if "best_k_map" in metrics_dict:
            self.logger.subsection("Plotting selected model-order colormap")
            saved["best_k_map"] = self.spatial_plotter._plot_discrete_k_map(
                metrics_dict["best_k_map"],
                r"Selected model order $K^*$ per pixel  (penalised-MSE minimiser)",
                r"selected $K^*$",
                dirs["colormaps"] / "best_k_map.png",
            )

        self.logger.subsection("Plotting R² spatial map")
        saved["r2_spatial_map"] = self.spatial_plotter._plot_r2_spatial_map(r2_map, dirs["colormaps"] / "r2_map.png")

        self.logger.subsection("Plotting amplitude spatial maps")
        amp_keys   = [f"amp_{k}"   for k in range(self.n_gaussians)]
        amp_titles = [f"$A_{{{k + 1}}}$  amplitude  (inactive pixels masked)" for k in range(self.n_gaussians)]
        saved.update(self.spatial_plotter._plot_spatial_maps(
            metrics_dict, amp_keys, amp_titles,
            "Gaussian amplitude maps",
            r"$A_k$",
            dirs["colormaps"] / "amplitude_maps",
            cmap="plasma",
        ))

        self.logger.subsection("Plotting height-centroid (μ) spatial maps")
        mu_keys   = [f"mu_{k}"    for k in range(self.n_gaussians)]
        mu_titles = [rf"$\mu_{{{k + 1}}}$  centroid [m]  (inactive pixels masked)" for k in range(self.n_gaussians)]
        saved.update(self.spatial_plotter._plot_spatial_maps(
            metrics_dict, mu_keys, mu_titles,
            r"Gaussian centroid height maps",
            r"$\mu_k$ [m]",
            dirs["colormaps"] / "mu_maps",
            cmap="RdYlGn",
        ))

        self.logger.subsection("Plotting sigma spatial maps")
        sig_keys   = [f"sigma_{k}" for k in range(self.n_gaussians)]
        sig_titles = [rf"$\sigma_{{{k + 1}}}$  spread [m]  (inactive pixels masked)" for k in range(self.n_gaussians)]
        saved.update(self.spatial_plotter._plot_spatial_maps(
            metrics_dict, sig_keys, sig_titles,
            r"Gaussian spread maps",
            r"$\sigma_k$ [m]",
            dirs["colormaps"] / "sigma_maps",
            cmap="viridis",
        ))

        if self.n_gaussians >= 2:
            self.logger.subsection("Plotting μ-separation maps")
            sep_keys   = [f"mu_sep_{k}_{k + 1}" for k in range(self.n_gaussians - 1)]
            sep_titles = [rf"$|\mu_{{{k + 2}}} - \mu_{{{k + 1}}}|$  [m]  (both active)" for k in range(self.n_gaussians - 1)]
            saved.update(self.spatial_plotter._plot_spatial_maps(
                metrics_dict, sep_keys, sep_titles,
                r"Adjacent centroid separation maps",
                "separation [m]",
                dirs["colormaps"] / "mu_separation_maps",
                cmap="magma",
            ))

        return saved

    def _run_distributions(self, parameters_array : np.ndarray, r2_map : np.ndarray, summary : dict, dirs : Dict[str, Path]) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        self.logger.subsection("Plotting R² distribution and CDF")
        saved.update(self.distribution_plotter._plot_r2_distribution(r2_map, summary, dirs["distributions"]))

        self.logger.subsection("Plotting parameter distributions")
        saved.update(self.distribution_plotter._plot_parameter_distributions(parameters_array, dirs["distributions"]))

        self.logger.subsection("Plotting joint parameter distributions")
        saved.update(self.distribution_plotter._plot_param_joint_distributions(parameters_array, dirs["distributions"]))

        return saved

    def _run_metrics_and_snr(self, metrics_dict : dict, r2_map : np.ndarray, summary : dict, dirs : Dict[str, Path]) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        self.logger.subsection("Plotting global metrics summary")
        saved.update(self.metrics_bar_plotter._plot_global_metrics_summary(summary, dirs["metrics"]))

        snr_db_map    = metrics_dict["snr_db_map"]
        per_k_summary = metrics_dict["per_k_summary"]

        if snr_db_map is not None:
            self.logger.subsection("Plotting SNR spatial map")
            saved["snr_map"] = self.spatial_plotter._plot_snr_map(snr_db_map, dirs["colormaps"] / "snr_map.png")

        if "mse_per_k" in metrics_dict:
            self.logger.subsection("Plotting per-K MSE and penalty decomposition")
            saved.update(self.metrics_bar_plotter._plot_mse_penalty_per_k(metrics_dict["mse_per_k"], per_k_summary, dirs["metrics"]))

        if "k_relative_margin_map" in metrics_dict:
            self.logger.subsection("Plotting K-selection ambiguity maps and distribution")
            margin_keys   = ["k_margin_prev_map", "k_margin_next_map", "k_relative_margin_map"]
            margin_titles = [r"margin to $K^*-1$  (small = ambiguous choice)", r"margin to $K^*+1$  (small = ambiguous choice)", "relative margin to runner-up  (small = ambiguous choice)"]

            saved.update(self.spatial_plotter._plot_spatial_maps(
                metrics_dict, margin_keys, margin_titles,
                r"K-selection margin maps",
                "penalised-score margin",
                dirs["colormaps"] / "k_ambiguity_maps",
                cmap="cividis",
            ))
            saved.update(self.distribution_plotter._plot_k_ambiguity_distribution(metrics_dict, per_k_summary, dirs["distributions"]))

        if snr_db_map is not None:
            self.logger.subsection("Plotting SNR against fit quality and K-selection ambiguity")
            best_k_map     = metrics_dict["best_k_map"]            if "best_k_map"            in metrics_dict else None
            rel_margin_map = metrics_dict["k_relative_margin_map"] if "k_relative_margin_map" in metrics_dict else None
            saved.update(self.metrics_bar_plotter._plot_snr_vs_fit_quality(
                snr_db_map, r2_map,
                best_k_map,
                rel_margin_map,
                metrics_dict["snr_summary"],
                dirs["metrics"],
            ))

        return saved

    def _run_example_fits(self, parameters_array : np.ndarray, metrics_dict : dict, r2_map : np.ndarray, height_axis : np.ndarray, tomogram_path : Path, dirs : Dict[str, Path]) -> Dict[str, Path]:
        if "best_k_map" not in metrics_dict:
            self.logger.warning("Example fit plots skipped: best_k_map unavailable")
            return {}

        fits = self.example_fit_plotter.run(parameters_array, metrics_dict["best_k_map"], r2_map, height_axis, tomogram_path, dirs["example_fits"])

        saved : Dict[str, Path] = {}
        for key, path in fits.items():
            saved[f"example_fit_{key}"] = path

        return saved

    def run(self, parameters_array : np.ndarray, metrics_dict : dict, metadata : dict, tomogram_path : Path) -> Dict[str, Path]:
        self.logger.section("[Fitting Result Plots]")
        self._apply_style()
        dirs = self._setup_output_dirs()

        r2_map       = metrics_dict["r2_map"]
        activity_map = metrics_dict["activity_map"]
        height_axis  = metrics_dict["height_axis"]
        summary      = metrics_dict["global_summary"]

        saved : Dict[str, Path] = {}

        saved.update(self._run_spatial_maps(metrics_dict, r2_map, activity_map, dirs))
        saved.update(self._run_distributions(parameters_array, r2_map, summary, dirs))
        saved.update(self._run_metrics_and_snr(metrics_dict, r2_map, summary, dirs))
        saved.update(self._run_example_fits(parameters_array, metrics_dict, r2_map, height_axis, tomogram_path, dirs))

        self.logger.subsection(f"Saved {len(saved)} figures -> {self._images_dir}")
        return saved
