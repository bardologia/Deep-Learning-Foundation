from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.reporting.plotting import PlotBase


class ActivationXrayPlots(PlotBase):

    def _depth_series(self, values: list[float], ylabel: str, title: str, path: Path, log: bool = False) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(range(len(values)), values, marker=".", linewidth=1.0)
        ax.set_xlabel("Leaf layer index (forward order)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)

    def zero_frac_by_depth(self, stats: list[dict], path: Path) -> Path:
        return self._depth_series([s["zero_frac"] for s in stats], "Zero fraction", "Exact-zero activation fraction by depth", path)

    def std_by_depth(self, stats: list[dict], path: Path) -> Path:
        values = [max(s["std"], 1e-12) for s in stats]
        return self._depth_series(values, "Activation std", "Activation standard deviation by depth", path, log=True)

    def dead_channels_by_depth(self, stats: list[dict], path: Path) -> Path:
        values = [s["dead_channel_frac"] if s["dead_channel_frac"] is not None else 0.0 for s in stats]
        return self._depth_series(values, "Dead channel fraction", "Dead-channel fraction by depth", path)

    def max_abs_by_depth(self, stats: list[dict], path: Path) -> Path:
        values = [max(s["max_abs"], 1e-12) for s in stats]
        return self._depth_series(values, "max |activation|", "Peak activation magnitude by depth", path, log=True)

    def layer_histogram(self, stats: dict, edges: np.ndarray, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        counts  = np.asarray(stats["hist_counts"], dtype=np.float64)[1:-1]
        centers = np.sqrt(edges[:-1] * edges[1:])

        ax.step(centers, counts, where="mid")
        ax.set_xscale("log")
        ax.set_xlabel("|activation|")
        ax.set_ylabel("Count")
        ax.set_title(f"{stats['name']} ({stats['module_type']})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)
