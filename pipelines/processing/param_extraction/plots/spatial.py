from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np

from tools.reporting.plotting import PlotBase
from tools.monitoring.logger  import Logger


class SpatialMapPlotter(PlotBase):
    def __init__(self, n_gaussians : int, logger : Logger, fig_dpi : int = 150, save_dpi : int = 300) -> None:
        self.n_gaussians = n_gaussians
        self.logger      = logger
        self.fig_dpi     = fig_dpi
        self.save_dpi    = save_dpi

    def _plot_discrete_k_map(self, k_map : np.ndarray, title : str, cbar_label : str, out_path : Path) -> Path:
        Az, R   = k_map.shape
        levels  = list(range(self.n_gaussians + 1))
        n_total = k_map.size

        text_lines   = ["share of all pixels:"] + [f"$K={k}$: {(k_map == k).sum() / n_total * 100:.1f}%" for k in levels if (k_map == k).sum() > 0]
        text_overlay = "\n".join(text_lines)

        return self._imshow_figure(
            k_map,
            x_label        = "range [px]",
            y_label        = "azimuth [px]",
            title          = title,
            cmap           = None,
            extent         = [0, R, Az, 0],
            colorbar_label = cbar_label,
            figsize        = (8, 6),
            discrete       = True,
            levels         = levels,
            text_overlay   = text_overlay,
            path           = out_path,
        )

    def _plot_spatial_maps(
        self,
        maps_dict  : Dict[str, np.ndarray],
        keys       : List[str],
        map_titles : List[str],
        group_name : str,
        col_label  : str,
        out_dir    : Path,
        cmap       : str = "plasma",
    ) -> Dict[str, Path]:
        valid_pairs = [(k, t) for k, t in zip(keys, map_titles) if k in maps_dict]
        if not valid_pairs:
            self.logger.warning(f"Spatial map group '{group_name}' skipped: no data")
            return {}

        cmap_obj = self._cmap_with_bad(cmap)
        saved    : Dict[str, Path] = {}

        for key, title in valid_pairs:
            data  = maps_dict[key].astype(np.float32)
            Az, R = data.shape

            if not np.isfinite(data).any():
                self.logger.warning(f"Spatial map '{key}' in group '{group_name}' skipped: field is entirely masked, no active pixels")
                continue

            vmin, vmax = self._shared_clim(data)

            saved[key] = self._imshow_figure(
                data,
                x_label        = "range [px]",
                y_label        = "azimuth [px]",
                title          = title,
                cmap           = cmap_obj,
                vmin           = vmin,
                vmax           = vmax,
                extent         = [0, R, Az, 0],
                colorbar_label = col_label,
                figsize        = (8, 6),
                path           = out_dir / f"{key}.png",
            )

        return saved

    def _plot_r2_spatial_map(self, r2_map : np.ndarray, out_path : Path) -> Path:
        Az, R    = r2_map.shape
        cmap_obj = self._cmap_with_bad("RdYlGn")
        vmin     = float(np.nanpercentile(r2_map, 1.0))
        vmax     = 1.0

        return self._imshow_figure(
            r2_map,
            x_label        = "range [px]",
            y_label        = "azimuth [px]",
            title          = rf"Per-pixel $R^2$ of Gaussian fit  (colour floor at $p_1={vmin:.2f}$)",
            cmap           = cmap_obj,
            vmin           = vmin,
            vmax           = vmax,
            extent         = [0, R, Az, 0],
            colorbar_label = r"$R^2$",
            figsize        = (8, 6),
            path           = out_path,
        )

    def _plot_snr_map(self, snr_db_map : np.ndarray, out_path : Path) -> Path:
        Az, R    = snr_db_map.shape
        cmap_obj = self._cmap_with_bad("viridis")
        vmin     = float(np.nanpercentile(snr_db_map, 1.0))
        vmax     = float(np.nanpercentile(snr_db_map, 99.0))

        return self._imshow_figure(
            snr_db_map,
            x_label        = "range [px]",
            y_label        = "azimuth [px]",
            title          = "Per-pixel peak-to-floor profile contrast  (uncalibrated proxy, not calibrated SNR)",
            cmap           = cmap_obj,
            vmin           = vmin,
            vmax           = vmax,
            extent         = [0, R, Az, 0],
            colorbar_label = "peak-to-floor contrast [dB]",
            figsize        = (8, 6),
            path           = out_path,
        )
