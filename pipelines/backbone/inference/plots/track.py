from __future__ import annotations

from pathlib import Path
from typing  import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools


class TrackPlotter(PlotTools):
    def plot_track_geometry(self, baselines, out_path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(5.4, 4.6))

        for index, label in enumerate(baselines.labels):
            color  = "black" if index == 0 else f"C{(index - 1) % 10}"
            marker = "*" if index == 0 else "o"
            size   = 140 if index == 0 else 60
            ax.scatter(baselines.horizontal[index], baselines.vertical[index], s=size, marker=marker, color=color, zorder=3)
            ax.annotate(label, (baselines.horizontal[index], baselines.vertical[index]), textcoords="offset points", xytext=(7, 5), fontsize=9)
            ax.errorbar(baselines.horizontal[index], baselines.vertical[index], xerr=baselines.horizontal_std[index], yerr=baselines.vertical_std[index], fmt="none", ecolor=color, elinewidth=0.7, capsize=2, alpha=0.6)

        ax.set_xlabel(r"horizontal baseline $b_{\perp,\mathrm{h}}$ [m]")
        ax.set_ylabel(r"vertical baseline $b_{\perp,\mathrm{v}}$ [m]")
        ax.set_title(f"Passes used  (reference {baselines.reference}, mean over azimuth window)")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_residual_spectroscopy(self, rows: List[dict], out_path: Path) -> Path:
        self._apply_style()

        kz     = [row["kz_abs_mean"] for row in rows]
        errors = [row["coherence_error"] for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(kz, errors, marker="o", color="#D55E00", linewidth=1.4)
        for row in rows:
            ax.annotate(row["label"], (row["kz_abs_mean"], row["coherence_error"]), textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_xlabel("Mean |kz| of the track [rad/m]")
        ax.set_ylabel("Coherence-resynthesis error")
        ax.set_yscale("log")
        ax.set_title("Physics residual by baseline")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_phase_spread_spectrum(self, rows: List[dict], out_path: Path) -> Path:
        self._apply_style()

        kz = [row["kz_abs_mean"] for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(kz, [row["spread_gt"] for row in rows],   marker="s", color="#0072B2", linewidth=1.2, label="GT labels")
        ax.plot(kz, [row["spread_pred"] for row in rows], marker="o", color="#D55E00", linewidth=1.4, label="Prediction")
        for row in rows:
            ax.annotate(row["label"], (row["kz_abs_mean"], row["spread_pred"]), textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_xlabel("Mean |kz| of the track [rad/m]")
        ax.set_ylabel("Random phase spread [rad]")
        ax.set_title("Random residual phase after removing the systematic offset")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_phase_agreement(self, labels: List[str], aligned: List[float], flipped: List[float], source: str, out_path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))

        positions = np.arange(len(labels), dtype=np.float64)
        width     = 0.38

        ax.bar(positions - width / 2, aligned, width, color="C0", label="aligned kz")
        ax.bar(positions + width / 2, flipped, width, color="C3", label="flipped kz")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("secondary pass")
        ax.set_ylabel("mean resultant length")
        ax.set_title(f"Measured-interferogram phase agreement ({source} curves)")
        ax.legend(framealpha=0.9, fontsize=8)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_track_profiles(self, profiles, out_dir: Path, split_azimuth: Optional[Tuple[int, int]] = None) -> List[Path]:
        azimuth = profiles.azimuth_axis
        paths   = []

        for component, data, symbol in (
            ("horizontal", profiles.relative_to_reference("horizontal"), r"$b_{\perp,\mathrm{h}}$"),
            ("vertical",   profiles.relative_to_reference("vertical"),   r"$b_{\perp,\mathrm{v}}$"),
        ):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))

            for index, label in enumerate(profiles.labels):
                color = "black" if index == 0 else f"C{(index - 1) % 10}"
                ax.plot(azimuth, data[index], color=color, linewidth=1.0, label=label)

            if split_azimuth is not None:
                ax.axvspan(split_azimuth[0], split_azimuth[1], color="C7", alpha=0.18, label="inference split")

            ax.set_xlabel("azimuth sample index")
            ax.set_ylabel(f"{symbol} relative to {profiles.labels[0]} [m]")
            ax.set_title(f"Per-azimuth {component} baselines of the passes used")
            ax.legend(framealpha=0.9, fontsize=8, ncol=2)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"baseline_profiles_{component}.png"))

        return paths

    def plot_track_flight_3d(self, profiles, out_path: Path, elev: float = 28.0, azim: float = -55.0) -> Path:
        azimuth = profiles.azimuth_axis
        radii   = profiles.deviation_radii()
        h_mean  = np.nanmean(profiles.horizontal, axis=1)
        v_mean  = np.nanmean(profiles.vertical,   axis=1)

        step  = max(1, len(azimuth) // 200)
        theta = np.linspace(0.0, 2.0 * np.pi, 36)
        az_grid, th_grid = np.meshgrid(azimuth[::step], theta)

        fig = plt.figure(figsize=(12.0, 7.0))
        ax  = fig.add_axes([0.12, 0.08, 0.70, 0.84], projection="3d")

        for index, label in enumerate(profiles.labels):
            color = "black" if index == 0 else f"C{(index - 1) % 10}"

            ax.plot(azimuth, profiles.horizontal[index], profiles.vertical[index], color=color, linewidth=1.3, label=f"{label}  (RMS dev {radii[index]:.2f} m)", zorder=3)
            ax.plot_surface(az_grid, h_mean[index] + radii[index] * np.cos(th_grid), v_mean[index] + radii[index] * np.sin(th_grid), color=color, alpha=0.16, linewidth=0, antialiased=False, shade=False)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("azimuth sample index", labelpad=14)
        ax.set_ylabel(r"horizontal baseline $b_{\perp,\mathrm{h}}$ [m]", labelpad=14)
        ax.set_zlabel(r"vertical baseline $b_{\perp,\mathrm{v}}$ [m]",   labelpad=18)
        ax.tick_params(axis="z", pad=8)
        ax.set_title("Flight tracks of the passes used  (tube radius = RMS planar deviation over azimuth)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

        return self._save(fig, out_path)
