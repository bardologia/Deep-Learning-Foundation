"""Publication-quality figures for the model weight x-ray.

Draws per-tensor series of norm, dispersion, sparsity and effective rank in
state-dict order, the pooled weight-value distribution, and one histogram per
flagged weight tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing  import List, Optional

import matplotlib.pyplot as plt
import numpy             as np
import torch

from matplotlib.ticker import MaxNLocator

from tools.reporting.plotting               import PlotBase
from tools.diagnostics.weight_xray_analysis import SEVERITY_RANK

from configuration.diagnostics import WeightXrayConfig


class WeightXrayPlots(PlotBase):
    """Renders the weight x-ray figure set for one checkpoint.

    Attributes:
        config     : Weight x-ray configuration supplying thresholds and sampling limits.
        output_dir : Directory the figures are written to.
    """

    save_dpi = 300

    SERIES_COLOR    = "#1f77b4"
    THRESHOLD_COLOR = "#d62728"
    TITLE_WRAP      = 58

    def __init__(self, config: WeightXrayConfig) -> None:
        """Initialises the plotter.

        Args:
            config : Weight x-ray configuration for the checkpoint.
        """
        self.config     = config
        self.output_dir = config.plots_directory

    def _wrap_title(self, title: str) -> str:
        """Returns the title broken across lines at dots when it exceeds the wrap width."""
        if len(title) <= self.TITLE_WRAP:
            return title

        lines, current = [], ""
        for part in title.split("."):
            candidate = f"{current}.{part}" if current else part
            if len(candidate) > self.TITLE_WRAP and current:
                lines.append(current + ".")
                current = part
            else:
                current = candidate

        lines.append(current)
        return "\n".join(lines)

    def _decorate(self, ax) -> None:
        """Applies the grid styling shared by every weight x-ray figure."""
        ax.grid(which="major", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)

    def _series_plot(self, values: List[float], y_label: str, title: str, filename: str, log_y: bool = False, threshold: Optional[float] = None) -> Optional[Path]:
        """Renders one per-tensor quantity against state-dict order.

        Args:
            values    : One value per tensor; None entries are dropped.
            y_label   : Axis label for the plotted quantity.
            title     : Figure title.
            filename  : Output file name relative to the plots directory.
            log_y     : Whether to use a logarithmic y axis.
            threshold : Optional value drawn as a horizontal reference line.

        Returns:
            Path of the saved figure, or None when no value is present.
        """
        points = [(index, value) for index, value in enumerate(values) if value is not None]
        if not points:
            return None

        xs = [index for index, _ in points]
        ys = [value for _, value in points]

        self._apply_style()
        fig, ax = plt.subplots(figsize=(7.4, 3.8))

        ax.plot(xs, ys, color=self.SERIES_COLOR, linewidth=1.2, marker="o", markersize=2.6, markerfacecolor=self.SERIES_COLOR, markeredgecolor="none")

        if threshold is not None:
            ax.axhline(threshold, color=self.THRESHOLD_COLOR, linewidth=1.0, linestyle="--", label=f"Threshold = {threshold:g}")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        if log_y:
            ax.set_yscale("log")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._decorate(ax)

        ax.set_xlabel("Tensor index (forward order)")
        ax.set_ylabel(y_label)
        ax.set_title(self._wrap_title(title))
        ax.margins(x=0.01)

        fig.tight_layout()
        return self._save(fig, self.output_dir / filename)

    def _histogram_plot(self, values: np.ndarray, title: str, filename: str) -> Optional[Path]:
        """Renders a log-count histogram of weight values.

        Args:
            values   : Flat array of weight values.
            title    : Figure title.
            filename : Output file name relative to the plots directory.

        Returns:
            Path of the saved figure, or None when the array is empty.
        """
        if values.size == 0:
            return None

        self._apply_style()
        fig, ax = plt.subplots(figsize=(6.6, 4.0))

        ax.hist(values, bins=160, color=self.SERIES_COLOR, edgecolor="none")
        ax.set_yscale("log")
        self._decorate(ax)

        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.set_title(self._wrap_title(title))

        fig.tight_layout()
        return self._save(fig, self.output_dir / filename)

    @staticmethod
    def _even_subsample(values: np.ndarray, keep: int) -> np.ndarray:
        """Returns at most keep values drawn at evenly spaced positions of the array."""
        if values.size <= keep:
            return values

        index = np.linspace(0, values.size - 1, keep).astype(np.int64)

        return values[index]

    def _global_sample(self, reports, state_dict: dict) -> np.ndarray:
        """Returns a size-proportional sample of all finite weight values.

        Args:
            reports    : Per-tensor reports; only those with role "weight" are pooled.
            state_dict : Mapping the reports were derived from.

        Returns:
            Flat array of at most histogram_sample_max weight values, drawn from
            each tensor in proportion to its element count.
        """
        weights = [report for report in reports if report.role == "weight"]
        total   = sum(state_dict[report.name].numel() for report in weights)

        if total == 0:
            return np.empty(0, dtype=np.float32)

        budget = int(self.config.histogram_sample_max)
        chunks = []

        for report in weights:
            tensor = state_dict[report.name]
            flat   = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
            finite = flat[np.isfinite(flat)]
            keep   = max(1, round(budget * tensor.numel() / total)) if total > budget else finite.size

            chunks.append(self._even_subsample(finite, keep))

        return self._even_subsample(np.concatenate(chunks), budget)

    def _flagged_histograms(self, reports, state_dict: dict) -> List[Path]:
        """Renders one value histogram per flagged weight tensor, worst severity first.

        Args:
            reports    : Per-tensor reports carrying their issues.
            state_dict : Mapping the reports were derived from.

        Returns:
            Paths of the histograms written, capped at max_layer_histograms.
        """
        flagged = [report for report in reports if report.role == "weight" and report.severity in ("warning", "critical")]
        flagged = sorted(flagged, key=lambda report: -SEVERITY_RANK[report.severity])[: self.config.max_layer_histograms]

        paths = []
        for report in flagged:
            flat   = state_dict[report.name].detach().to(torch.float32).cpu().numpy().reshape(-1)
            finite = flat[np.isfinite(flat)]
            safe   = report.name.replace(".", "_").replace("/", "_")

            path = self._histogram_plot(finite, f"{report.name} ({report.severity})", str(Path("layer_histograms") / f"{safe}.png"))
            if path is not None:
                paths.append(path)

        return paths

    def render(self, reports, state_dict: dict) -> List[Path]:
        """Renders the complete weight x-ray figure set.

        Args:
            reports    : Per-tensor reports carrying statistics and issues.
            state_dict : Mapping the reports were derived from.

        Returns:
            Paths of every figure written: the four per-tensor series, the
            pooled weight histogram, and the per-tensor flagged histograms.
        """
        weight_reports = [report for report in reports if report.role == "weight"]
        thresholds     = self.config.thresholds

        produced = [
            self._series_plot([report.l2_norm    for report in weight_reports], "L2 norm",              "Per-tensor weight L2 norm",        "layer_l2_norm.png",    log_y=True),
            self._series_plot([report.std        for report in weight_reports], "Standard deviation",   "Per-tensor weight dispersion",     "layer_std.png",        log_y=True),
            self._series_plot([report.frac_dead  for report in weight_reports], "Dead fraction",        "Per-tensor dead-weight fraction",  "layer_sparsity.png",   threshold=thresholds.dead_fraction_warn),
            self._series_plot([report.rank_ratio for report in weight_reports], "Effective rank ratio", "Per-tensor effective rank ratio",  "layer_rank_ratio.png", threshold=thresholds.rank_ratio_warn),
            self._histogram_plot(self._global_sample(reports, state_dict), "Pooled weight distribution", "weight_histogram.png"),
        ]

        produced = [path for path in produced if path is not None]
        produced.extend(self._flagged_histograms(reports, state_dict))
        return produced
