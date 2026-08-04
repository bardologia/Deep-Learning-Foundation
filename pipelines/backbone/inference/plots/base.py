from __future__ import annotations

from pathlib import Path
from typing  import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors    as mcolors
import numpy                as np

from tools.reporting.plotting import PlotBase
from tools.data.gaussians     import GaussianReconstructor


class PlotTools(PlotBase):
    PARAM_LABELS = (("a", "amplitude (a)"), ("mu", "mean (μ)"), ("sigma", "std-dev (σ)"))
    PARAM_SHORT  = ("a", "μ", "σ")

    def __init__(
        self,
        cmap      : str  = "jet",
        err_cmap  : str  = "magma",
        normalize : bool = False,
        fig_dpi   : int  = 150,
        save_dpi  : int  = 150,
    ) -> None:

        self.cmap      = cmap
        self.err_cmap  = err_cmap
        self.normalize = normalize
        self.fig_dpi   = fig_dpi
        self.save_dpi  = save_dpi
        self._apply_style()

    @staticmethod
    def _gaussian_components(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        return GaussianReconstructor.components(params, x_axis, n_gaussians)

    def _intensity_scale(self, reference: np.ndarray) -> float:
        if not self.normalize:
            return 1.0

        ref   = reference[np.isfinite(reference)]
        scale = float(ref.max()) if ref.size else 0.0

        return scale if scale > 1e-12 else 1.0

    @staticmethod
    def _rescale(arr: np.ndarray, scale: float) -> np.ndarray:
        return (arr / scale).astype(np.float32)

    @property
    def _int_label(self) -> str:
        return "intensity (GT-peak normalised)" if self.normalize else "intensity"

    @property
    def _err_label(self) -> str:
        return "|error| (GT-peak normalised)" if self.normalize else "|error| (intensity)"

    def _imshow_panel(
        self,
        data       : np.ndarray,
        title      : str,
        x_label    : str,
        y_label    : str,
        cbar_label : str,
        extent     : list,
        cmap       : str | mcolors.Colormap,
        vmin       : float,
        vmax       : float,
        origin     : str,
        path       : Path,
        figsize    : Tuple[float, float] = (6.2, 4.4),
    ) -> Path:

        return self._imshow_figure(
            data,
            x_label        = x_label,
            y_label        = y_label,
            title          = title,
            cmap           = cmap,
            vmin           = vmin,
            vmax           = vmax,
            extent         = extent,
            origin         = origin,
            colorbar_label = cbar_label,
            figsize        = figsize,
            path           = path,
        )
