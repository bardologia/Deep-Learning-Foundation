from __future__ import annotations

from pipelines.backbone.inference.plots.base         import PlotTools
from pipelines.backbone.inference.plots.slice        import SlicePlotter
from pipelines.backbone.inference.plots.param        import ParamPlotter
from pipelines.backbone.inference.plots.slot         import SlotPlotter
from pipelines.backbone.inference.plots.stratified   import StratifiedErrorPlotter
from pipelines.backbone.inference.plots.track        import TrackPlotter
from pipelines.backbone.inference.plots.organization import SlotOrganizationPlotter


class Plotter(PlotTools):
    def __init__(
        self,
        cmap      : str  = "jet",
        err_cmap  : str  = "magma",
        normalize : bool = False,
        fig_dpi   : int  = 150,
        save_dpi  : int  = 150,
    ) -> None:

        super().__init__(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)

        self.slice = SlicePlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.param = ParamPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.slot  = SlotPlotter( cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.track = TrackPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.organization = SlotOrganizationPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.stratified   = StratifiedErrorPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
