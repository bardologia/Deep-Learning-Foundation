"""Shared drawing helpers every backbone-inference plotter inherits.

Carries the colormap and DPI settings, the Gaussian-parameter labels used across figures,
the optional GT-peak intensity rescaling and the single-panel imshow wrapper.
"""

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
    """Base class holding the plot style and the helpers shared by the inference plotters.

    Attributes:
        cmap      : Colormap for intensity panels.
        err_cmap  : Colormap for error panels.
        normalize : Whether intensity panels are rescaled by the GT peak.
        fig_dpi   : Figure DPI used while rendering.
        save_dpi  : DPI used when writing the file.
    """

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
        """Stores the plot style and applies it to matplotlib.

        Args:
            cmap      : Colormap for intensity panels.
            err_cmap  : Colormap for error panels.
            normalize : Whether intensity panels are rescaled by the GT peak.
            fig_dpi   : Figure DPI used while rendering.
            save_dpi  : DPI used when writing the file.
        """

        self.cmap      = cmap
        self.err_cmap  = err_cmap
        self.normalize = normalize
        self.fig_dpi   = fig_dpi
        self.save_dpi  = save_dpi
        self._apply_style()

    @staticmethod
    def _gaussian_components(params: np.ndarray, z_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        """Returns the individual Gaussian curves of one pixel, each of shape (n_z,).

        Args:
            params      : Parameter vector of length 3 * n_gaussians for a single pixel.
            z_axis      : Z axis in metres, of shape (n_z,).
            n_gaussians : Gaussian count K.

        Returns:
            One rendered curve per Gaussian.
        """

        return GaussianReconstructor.components(params, z_axis, n_gaussians)

    def _intensity_scale(self, reference: np.ndarray) -> float:
        """Returns the divisor rescaling intensities by the reference peak, or 1.0 when normalisation is off.

        Args:
            reference : Reference array whose finite maximum sets the scale.

        Returns:
            The scale factor, falling back to 1.0 for an empty or degenerate reference.
        """

        if not self.normalize:
            return 1.0

        ref   = reference[np.isfinite(reference)]
        scale = float(ref.max()) if ref.size else 0.0

        return scale if scale > 1e-12 else 1.0

    @staticmethod
    def _rescale(arr: np.ndarray, scale: float) -> np.ndarray:
        """Returns the array divided by `scale` as float32."""

        return (arr / scale).astype(np.float32)

    @property
    def _int_label(self) -> str:
        """Colorbar label for intensity panels, reflecting whether GT-peak normalisation is on."""

        return "intensity (GT-peak normalised)" if self.normalize else "intensity"

    @property
    def _err_label(self) -> str:
        """Colorbar label for error panels, reflecting whether GT-peak normalisation is on."""

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
        """Renders one labelled image panel with a colorbar and writes it to `path`.

        Args:
            data       : 2-D array to display.
            title      : Panel title.
            x_label    : Horizontal axis label.
            y_label    : Vertical axis label.
            cbar_label : Colorbar label.
            extent     : Axis extent as [x_min, x_max, y_min, y_max].
            cmap       : Colormap name or instance.
            vmin       : Lower colour limit.
            vmax       : Upper colour limit.
            origin     : Image origin, "upper" or "lower".
            path       : Destination file path.
            figsize    : Figure size in inches.

        Returns:
            The written file path.
        """

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
