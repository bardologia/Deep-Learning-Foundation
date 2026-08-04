from __future__ import annotations

from pathlib import Path
from typing  import List, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy             as np

matplotlib.use("Agg")

from tools.reporting.colormaps import PhaseColormap


class PlotBase:
    PHASE_TICKS  = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    PHASE_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    FULL_WIDTH : float = 5.5
    HALF_WIDTH : float = 2.65

    OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

    SCIENTIFIC_RC: dict = {
        "font.family"         : "serif",
        "font.serif"          : ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth"      : 0.8,
        "xtick.direction"     : "in",
        "ytick.direction"     : "in",
        "xtick.top"           : True,
        "ytick.right"         : True,
        "xtick.minor.visible" : True,
        "ytick.minor.visible" : True,
        "image.interpolation" : "nearest",
        "savefig.bbox"        : "tight",
        "pdf.fonttype"        : 42,
        "ps.fonttype"         : 42,
    }

    REPORT_RC: dict = {
        "font.size"        : 11,
        "axes.titlesize"   : 12,
        "axes.labelsize"   : 11,
        "xtick.labelsize"  : 10,
        "ytick.labelsize"  : 10,
        "legend.fontsize"  : 9,
        "mathtext.fontset" : "dejavuserif",
        "lines.linewidth"  : 1.5,
        "lines.markersize" : 6.0,
    }

    PAPER_RC: dict = {
        "font.size"        : 9,
        "axes.titlesize"   : 9,
        "axes.labelsize"   : 9,
        "xtick.labelsize"  : 8,
        "ytick.labelsize"  : 8,
        "legend.fontsize"  : 8,
        "mathtext.fontset" : "stix",
        "lines.linewidth"  : 1.2,
        "lines.markersize" : 4.0,
    }

    STYLE_RC = {"report": REPORT_RC, "paper": PAPER_RC}

    PAPER_DPI: int = 300

    style: str = "report"

    fig_dpi  : int = 150
    save_dpi : int = 150

    @classmethod
    def use_style(cls, style: str) -> None:
        if style not in PlotBase.STYLE_RC:
            raise ValueError(f"unknown figure style '{style}', expected one of {sorted(PlotBase.STYLE_RC)}")
        PlotBase.style = style

    @classmethod
    def figsize(cls, width: float, aspect: float = 0.62) -> Tuple[float, float]:
        return (width, width * aspect)

    def _apply_style(self) -> None:
        plt.rcParams.update(self.SCIENTIFIC_RC)
        plt.rcParams.update(self.STYLE_RC[PlotBase.style])

        if PlotBase.style == "paper":
            plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.OKABE_ITO)
            plt.rcParams["figure.dpi"]      = self.PAPER_DPI
            plt.rcParams["savefig.dpi"]     = self.PAPER_DPI
        else:
            plt.rcParams["axes.prop_cycle"] = plt.rcParamsDefault["axes.prop_cycle"]
            plt.rcParams["figure.dpi"]      = self.fig_dpi
            plt.rcParams["savefig.dpi"]     = self.save_dpi

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        if PlotBase.style == "paper" and path.suffix == ".png" and not any(ax.images for ax in fig.axes):
            path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _shared_clim(*arrays: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
        flat = np.concatenate([a.reshape(-1) for a in arrays])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise ValueError("_shared_clim received no finite values; cannot derive a colour scale from an empty or all-NaN field")
        return float(np.percentile(flat, q_low)), float(np.percentile(flat, q_high))

    @staticmethod
    def _amplitude_clim(*arrays: np.ndarray) -> Tuple[float, float]:
        flat = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise ValueError("_amplitude_clim received no finite values; cannot derive an amplitude scale from an empty or all-NaN field")

        vmax = 3.0 * float(flat.mean())
        if vmax <= 0.0:
            raise ValueError("_amplitude_clim requires a positive mean amplitude; got a non-positive field")

        return 0.0, vmax

    @staticmethod
    def _cmap_with_bad(name: str, bad_color: str = "0.88") -> mcolors.Colormap:
        cmap = plt.get_cmap(name).copy()
        cmap.set_bad(color=bad_color)
        return cmap

    @staticmethod
    def _phase_cmap() -> mcolors.Colormap:
        return PhaseColormap.colormap()

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo < 1e-12:
            raise ValueError("_normalize_01 received a constant or degenerate field; refusing to flatten it silently to a single colour")
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    def _imshow_figure(
        self,
        data           : np.ndarray,
        *,
        x_label        : str,
        y_label        : str,
        title          : str,
        cmap,
        vmin           : float | None        = None,
        vmax           : float | None        = None,
        extent         : list | None         = None,
        origin         : str                 = "upper",
        colorbar_label : str                 = "",
        interpolation  : str | None          = "nearest",
        aspect         : str                 = "auto",
        figsize        : Tuple[float, float] = (6.2, 4.4),
        discrete       : bool                = False,
        levels         : List[int] | None    = None,
        text_overlay   : str | None          = None,
        path           : Path | None         = None,
    ):
        self._apply_style()

        if not discrete and np.issubdtype(np.asarray(data).dtype, np.floating):
            data = np.asarray(data, dtype=np.float64)

        fig, ax = plt.subplots(figsize=figsize)

        if discrete:
            level_list = list(levels) if levels is not None else list(range(int(np.nanmax(data)) + 1))
            palette    = plt.get_cmap("tab10", len(level_list))
            disc_cmap  = mcolors.ListedColormap([palette(i) for i in range(len(level_list))])
            bounds     = [value - 0.5 for value in level_list] + [level_list[-1] + 0.5]
            norm       = mcolors.BoundaryNorm(bounds, disc_cmap.N)

            im = ax.imshow(data, cmap=disc_cmap, norm=norm, extent=extent, aspect=aspect, origin=origin, interpolation=interpolation)

            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=level_list, boundaries=bounds)
            cbar.set_label(colorbar_label)
            cbar.ax.set_yticklabels([str(value) for value in level_list])
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect=aspect, origin=origin, interpolation=interpolation)

            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(colorbar_label)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if text_overlay is not None:
            ax.text(0.02, 0.98, text_overlay, transform=ax.transAxes, fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        fig.tight_layout()

        if path is not None:
            return self._save(fig, path)

        return fig

    @staticmethod
    def _subsample(values: np.ndarray, n_max: int, seed: int = 0) -> np.ndarray:
        vals = values[np.isfinite(values)]
        if vals.size > n_max:
            rng  = np.random.default_rng(seed)
            vals = rng.choice(vals, size=n_max, replace=False)
        return vals

    @staticmethod
    def _paired_subsample(arrays: List[np.ndarray], n_max: int, seed: int = 0) -> List[np.ndarray]:
        flats = [a.reshape(-1) for a in arrays]
        ok    = np.ones(flats[0].size, dtype=bool)

        for flat in flats:
            ok &= np.isfinite(flat)

        idx = np.where(ok)[0]
        if idx.size > n_max:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=n_max, replace=False)

        return [flat[idx] for flat in flats]

    @staticmethod
    def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 30, min_count: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        edges   = np.linspace(float(x.min()), float(x.max()), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        medians = np.full(n_bins, np.nan)
        which   = np.digitize(x, edges[1:-1])

        for b in range(n_bins):
            sel = which == b
            if int(sel.sum()) >= min_count:
                medians[b] = float(np.median(y[sel]))

        return centers, medians

    @staticmethod
    def _violin_with_iqr(ax, data_by_slot: List[np.ndarray], palette: List) -> None:
        drawn_positions = [i + 1 for i, d in enumerate(data_by_slot) if d.size > 0]
        drawn_data      = [d     for    d in data_by_slot           if d.size > 0]
        drawn_colors    = [palette[i] for i, d in enumerate(data_by_slot) if d.size > 0]

        if drawn_data:
            parts = ax.violinplot(drawn_data, positions=drawn_positions, showmedians=True, showextrema=False, widths=0.7)

            for body, color in zip(parts["bodies"], drawn_colors):
                body.set_facecolor(color)
                body.set_alpha(0.60)

            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.8)

        y_lo, y_hi = ax.get_ylim()
        for i, (slot_data, color) in enumerate(zip(data_by_slot, palette)):
            if slot_data.size == 0:
                ax.text(i + 1, y_lo + 0.02 * (y_hi - y_lo), "n=0", ha="center", va="bottom", fontsize=8, color="0.45", rotation=90)
                continue
            if slot_data.size < 4:
                continue
            q25, q50, q75 = np.percentile(slot_data, [25, 50, 75])
            ax.vlines(i + 1, q25, q75, color=color, lw=3.0, zorder=3)
            ax.scatter(i + 1, q50, color="white", s=22, zorder=5, edgecolors=color, linewidths=1.2)

    @staticmethod
    def _render_panels(
        fig,
        axes,
        panels         : List[Tuple[np.ndarray, str, str, float, float]],
        *,
        x_label        : str,
        extent         : list,
        origin         : str,
        interpolation  : str | None = None,
        title_size     : int | None = None,
        label_size     : int | None = None,
        colorbar_label = None,
    ) -> None:

        for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes, panels):
            im = ax_i.imshow(data, cmap=cm_used, vmin=vlo, vmax=vhi, extent=extent, aspect="auto", origin=origin, interpolation=interpolation)

            ax_i.set_title(label, fontsize=title_size)
            ax_i.set_xlabel(x_label, fontsize=label_size)

            cbar = fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label(cm_used))

    @staticmethod
    def _triple_panel(
        fig,
        axes,
        panels    : List[Tuple[np.ndarray, str, str, float, float]],
        x_label   : str,
        int_label : str,
        extent    : list,
        origin    : str,
    ) -> None:

        PlotBase._render_panels(
            fig, axes, panels,
            x_label        = x_label,
            extent         = extent,
            origin         = origin,
            colorbar_label = lambda cm_used: int_label if cm_used == panels[0][2] else "|error|",
        )
