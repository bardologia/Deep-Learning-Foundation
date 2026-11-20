from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.inference  import InferenceConfig
from tools.reporting.plotting import PlotBase


class TargetMapFigures(PlotBase):

    def __init__(self, config: InferenceConfig) -> None:
        self.config   = config
        self.fig_dpi  = config.fig_dpi
        self.save_dpi = config.save_dpi

    def _map(self, data: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, label: str, path: Path) -> Path:
        return self._imshow_figure(data, x_label="col (pixel)", y_label="row (pixel)", title=title, cmap=self._cmap_with_bad(cmap), vmin=vmin, vmax=vmax, colorbar_label=label, path=path)

    def channel(self, name: str, prediction: np.ndarray, target: np.ndarray, out_dir: Path) -> dict[str, Path]:
        vmin, vmax = self._shared_clim(target, prediction)
        error      = np.abs(prediction - target)

        return {
            "target"     : self._map(target,     f"{name}: target",           self.config.cmap,       vmin, vmax,                             name,             out_dir / "target.png"),
            "prediction" : self._map(prediction, f"{name}: prediction",       self.config.cmap,       vmin, vmax,                             name,             out_dir / "prediction.png"),
            "error"      : self._map(error,      f"{name}: absolute error",   self.config.cmap_error, 0.0,  float(np.nanpercentile(error, 99.0)), f"|error| {name}", out_dir / "error.png"),
        }

    def compose(self, predictions: np.ndarray, targets: np.ndarray, names: list[str], figures_dir: Path) -> dict[str, dict[str, Path]]:
        return {name: self.channel(name, predictions[index], targets[index], figures_dir / name) for index, name in enumerate(names)}
