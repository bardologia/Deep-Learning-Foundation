"""Publication-quality figures for the activation x-ray diagnostic.

Draws depth-ordered series of per-layer activation statistics, colour-coded by
module family and ringed where the issue detector flagged a layer, plus the
magnitude histogram of an individual layer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.reporting.plotting import PlotBase


class ModuleFamilies:
    """Maps torch module class names to plotting families and their colours."""

    FAMILIES = (
        ("linear",     ("Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Linear")),
        ("norm",       ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "LayerNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d")),
        ("activation", ("ReLU", "LeakyReLU", "GELU", "SiLU", "ELU", "PReLU", "Sigmoid", "Tanh", "Hardswish", "Softplus", "Mish")),
        ("resample",   ("MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d", "AdaptiveAvgPool2d", "Upsample", "PixelShuffle", "PixelUnshuffle")),
    )

    COLORS = {
        "linear"     : PlotBase.OKABE_ITO[0],
        "norm"       : PlotBase.OKABE_ITO[2],
        "activation" : PlotBase.OKABE_ITO[3],
        "resample"   : PlotBase.OKABE_ITO[5],
        "other"      : "0.45",
    }

    @classmethod
    def of(cls, module_type: str) -> str:
        """Returns the plotting family of a module class name, or "other" when unlisted."""
        for family, types in cls.FAMILIES:
            if module_type in types:
                return family
        return "other"


class ActivationXrayPlots(PlotBase):
    """Renders the activation x-ray figure set for one capture.

    Attributes:
        config : Activation x-ray configuration supplying the threshold values drawn as reference lines.
    """

    SEVERITY_EDGE = {"critical": "#B22222", "warning": "#E69F00"}

    def __init__(self, config) -> None:
        """Initialises the plotter.

        Args:
            config : Configuration exposing the detector thresholds.
        """
        self.config = config

    def _annotate_extreme(self, ax, depth: np.ndarray, values: np.ndarray, stats: list[dict], take_min: bool) -> None:
        """Labels the layer holding the smallest or largest finite value of the series."""
        finite = np.isfinite(values)
        if not finite.any():
            return

        candidates = np.where(finite)[0]
        pick       = candidates[np.argmin(values[candidates])] if take_min else candidates[np.argmax(values[candidates])]
        short      = ".".join(stats[pick]["name"].split(".")[-2:])

        ax.annotate(short, xy=(depth[pick], values[pick]), xytext=(4, 6), textcoords="offset points", fontsize=7, color="0.25")

    def _depth_series(
        self,
        stats      : list[dict],
        values     : list[float | None],
        ylabel     : str,
        title      : str,
        path       : Path,
        log        : bool = False,
        references : tuple = (),
        flagged    : dict[str, str] | None = None,
        annotate_min : bool = False,
    ) -> Path:
        """Renders one per-layer quantity against forward-call depth.

        Args:
            stats        : Per-layer statistics in forward-call order.
            values       : One value per layer; None entries are drawn as gaps.
            ylabel       : Axis label for the plotted quantity.
            title        : Figure title.
            path         : Destination file for the figure.
            log          : Whether to use a logarithmic y axis.
            references   : Tuples of (value, label, colour) drawn as horizontal threshold lines.
            flagged      : Severity keyed by layer name; flagged points get a ring.
            annotate_min : Whether to annotate the minimum instead of the maximum.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        series = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
        depth  = np.arange(series.size)

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.55))
        ax.plot(depth, series, color="0.78", linewidth=0.9, zorder=1)

        families = [ModuleFamilies.of(s["module_type"]) for s in stats]
        for family in dict.fromkeys(families):
            sel = [i for i, f in enumerate(families) if f == family and np.isfinite(series[i])]
            if sel:
                ax.scatter(depth[sel], series[sel], s=18, color=ModuleFamilies.COLORS[family], label=family, zorder=3)

        flagged = flagged or {}
        for i, s in enumerate(stats):
            severity = flagged.get(s["name"])
            if severity is not None and np.isfinite(series[i]):
                ax.scatter(depth[i], series[i], s=70, facecolors="none", edgecolors=self.SEVERITY_EDGE[severity], linewidths=1.4, zorder=4, label=None)

        for value, label, color in references:
            ax.axhline(value, color=color, linestyle="--", linewidth=1.0, label=label, zorder=2)

        self._annotate_extreme(ax, depth, series, stats, take_min=annotate_min)

        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Leaf layer index (forward-call order)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)

    def zero_frac_by_depth(self, stats: list[dict], flagged: dict[str, str], path: Path) -> Path:
        """Plots the exact-zero activation fraction against depth with dead-layer thresholds."""
        references = (
            (self.config.dead_zero_frac_warn,     f"warn ({self.config.dead_zero_frac_warn:.2f})",     self.SEVERITY_EDGE["warning"]),
            (self.config.dead_zero_frac_critical, f"critical ({self.config.dead_zero_frac_critical:.2f})", self.SEVERITY_EDGE["critical"]),
        )

        return self._depth_series(stats, [s["zero_frac"] for s in stats], "Exact-zero fraction", "Exact-zero activation fraction by depth", path, references=references, flagged=flagged)

    def std_by_depth(self, stats: list[dict], flagged: dict[str, str], path: Path) -> Path:
        """Plots the activation standard deviation against depth on a log axis."""
        references = ((self.config.constant_std_threshold, f"constant ({self.config.constant_std_threshold:.0e})", self.SEVERITY_EDGE["warning"]),)
        values     = [max(s["std"], 1e-12) for s in stats]

        return self._depth_series(stats, values, "Activation standard deviation", "Activation spread by depth", path, log=True, references=references, flagged=flagged, annotate_min=True)

    def dead_channels_by_depth(self, stats: list[dict], flagged: dict[str, str], path: Path) -> Path:
        """Plots the dead-channel fraction against depth with its warning threshold."""
        references = ((self.config.dead_channel_frac_warn, f"warn ({self.config.dead_channel_frac_warn:.2f})", self.SEVERITY_EDGE["warning"]),)

        return self._depth_series(stats, [s["dead_channel_frac"] for s in stats], "Dead-channel fraction", "Dead-channel fraction by depth", path, references=references, flagged=flagged)

    def max_abs_by_depth(self, stats: list[dict], flagged: dict[str, str], path: Path) -> Path:
        """Plots the peak absolute activation against depth on a log axis."""
        references = ((self.config.explode_abs_threshold, f"explode ({self.config.explode_abs_threshold:.0e})", self.SEVERITY_EDGE["critical"]),)
        values     = [max(s["max_abs"], 1e-12) for s in stats]

        return self._depth_series(stats, values, "Peak |activation|", "Peak activation magnitude by depth", path, log=True, references=references, flagged=flagged)

    def effective_channels_by_depth(self, stats: list[dict], flagged: dict[str, str], path: Path) -> Path:
        """Plots the entropy-effective channel fraction against depth."""
        references = ((self.config.channel_collapse_frac_warn, f"collapse ({self.config.channel_collapse_frac_warn:.2f})", self.SEVERITY_EDGE["warning"]),)

        return self._depth_series(stats, [s["effective_channel_frac"] for s in stats], "Effective channel fraction", "Channel utilisation by depth (entropy-effective channels / channels)", path, references=references, flagged=flagged, annotate_min=True)

    def dynamic_range_by_depth(self, stats: list[dict], flagged: dict[str, str], path: Path) -> Path:
        """Plots the p99/p50 ratio of |activation| against depth on a log axis."""
        return self._depth_series(stats, [s["dynamic_range"] for s in stats], "p99 / p50 of |activation|", "Activation dynamic range by depth", path, log=True, flagged=flagged)

    def layer_histogram(self, stats: dict, edges: np.ndarray, path: Path) -> Path:
        """Renders the |activation| histogram of one layer with its percentile markers.

        Args:
            stats : Finalised statistics of the layer to plot.
            edges : Log-spaced histogram bin edges, of length one more than the number of interior bins.
            path  : Destination file for the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.55))
        counts  = np.asarray(stats["hist_counts"], dtype=np.float64)[1:-1]
        centers = np.sqrt(edges[:-1] * edges[1:])

        ax.step(centers, counts, where="mid", color=self.OKABE_ITO[0], linewidth=1.2)
        ax.fill_between(centers, counts, step="mid", color=self.OKABE_ITO[0], alpha=0.25)

        ax.axvline(stats["abs_p50"], color=self.OKABE_ITO[1], linestyle="--", linewidth=1.0, label=f"p50 = {stats['abs_p50']:.3g}")
        ax.axvline(stats["abs_p99"], color=self.OKABE_ITO[2], linestyle="--", linewidth=1.0, label=f"p99 = {stats['abs_p99']:.3g}")
        ax.axvline(max(stats["max_abs"], 1e-12), color="0.3", linestyle=":", linewidth=1.0, label=f"max = {stats['max_abs']:.3g}")

        dead = f"{stats['dead_channels']}/{stats['n_channels']}" if stats["n_channels"] is not None else "n/a"
        box  = f"zero {stats['zero_frac'] * 100.0:.1f}%\ndead ch {dead}\nstd {stats['std']:.3g}"
        ax.text(0.02, 0.97, box, transform=ax.transAxes, fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        shape = "x".join(str(v) for v in stats["shape"]) if stats["shape"] else "scalar"
        ax.set_xscale("log")
        ax.set_xlabel("|activation| (log-spaced bins)")
        ax.set_ylabel("Count")
        ax.set_title(f"{stats['name']} ({stats['module_type']}, {shape})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)
