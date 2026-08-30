"""Groups the per-family inference plotters behind one object sharing a single style."""

from __future__ import annotations

from pipelines.backbone.inference.plots.base         import PlotTools
from pipelines.backbone.inference.plots.slice        import SlicePlotter
from pipelines.backbone.inference.plots.param        import ParamPlotter
from pipelines.backbone.inference.plots.slot         import SlotPlotter
from pipelines.backbone.inference.plots.stratified   import StratifiedErrorPlotter
from pipelines.backbone.inference.plots.organization import SlotOrganizationPlotter


class Plotter(PlotTools):
    """Bundle exposing one plotter per figure family, all sharing the same colormaps and DPI.

    Attributes:
        slice        : Plotter for profiles, cube slices, pixel maps and metric curves.
        param        : Plotter for Gaussian-parameter distributions, scatters, error maps and histograms.
        slot         : Plotter for per-slot activity maps.
        organization : Plotter for slot-usage, μ-rank and GT-assignment matrices.
        stratified   : Plotter for covariate-stratified errors, failure modes and presence reliability.
    """

    def __init__(
        self,
        cmap      : str  = "jet",
        err_cmap  : str  = "magma",
        normalize : bool = False,
        fig_dpi   : int  = 150,
        save_dpi  : int  = 150,
    ) -> None:
        """Builds one plotter per figure family with a shared style.

        Args:
            cmap      : Colormap for intensity panels.
            err_cmap  : Colormap for error panels.
            normalize : Whether intensity panels are rescaled by the GT peak.
            fig_dpi   : Figure DPI used while rendering.
            save_dpi  : DPI used when writing the file.
        """

        super().__init__(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)

        self.slice = SlicePlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.param = ParamPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.slot  = SlotPlotter( cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.organization = SlotOrganizationPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.stratified   = StratifiedErrorPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
