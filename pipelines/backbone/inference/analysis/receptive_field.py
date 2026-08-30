"""Effective receptive field (ERF) measurement for trained backbones.

The ERF is the input-gradient magnitude of the centre-pixel output with respect
to the surrounding input window, averaged over probe pixels. It is summarised by
Gaussian-equivalent sigmas, cumulative gradient-mass curves and marginal axis
profiles, all in input pixels.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.patches import Ellipse

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice, ProbeWindows
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class ErfComputation:
    """Computes effective-receptive-field gradient maps and their summaries.

    Attributes:
        model        : Backbone module evaluated at the probe windows.
        window       : Side of the square probe window, in input pixels.
        mass_windows : Window sides at which cumulative gradient mass is reported.
    """

    FAMILIES = ("amp", "mu", "sigma")

    def __init__(self, model, window: int, mass_windows: list[int]) -> None:
        """Binds the model and validates the requested mass windows.

        Args:
            model        : Backbone module taking an input window and predicting Gaussian parameters interleaved as (amp, mu, sigma) per slot.
            window       : Side of the square probe window, in input pixels.
            mass_windows : Window sides (pixels) at which to report gradient mass.

        Raises:
            ValueError : If any requested mass window exceeds the probe window.
        """
        self.model        = model
        self.window       = int(window)
        self.mass_windows = [int(w) for w in mass_windows]

        bad = [w for w in self.mass_windows if w > self.window]
        if bad:
            raise ValueError(f"mass_windows {bad} exceed the probe window {self.window}; shrink them or enlarge window")

    def gradient_maps(self, windows: torch.Tensor) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
        """Accumulates per-family input-gradient magnitude maps over probe windows.

        Args:
            windows : Probe input windows of shape (probes, channels, window, window).

        Returns:
            A tuple of the gradient maps and the per-probe sigmas. The maps are
            keyed by output family plus "combined", each a (window, window) array
            of mean absolute input gradient, or None for a family the output does
            not depend on. The sigmas array has shape (valid_probes, 2), holding
            (azimuth, range) Gaussian-equivalent sigmas in pixels.

        Raises:
            ValueError : If the combined gradient map is identically zero.
        """
        half   = self.window // 2
        device = ModelDevice.of(self.model)

        family_totals = {family: torch.zeros(self.window, self.window) for family in self.FAMILIES}
        probe_sigmas  = []

        for p in range(windows.shape[0]):
            x = windows[p:p + 1].clone().to(device).requires_grad_(True)
            y = self.model(x)

            probe_map = torch.zeros(self.window, self.window)
            for f, family in enumerate(self.FAMILIES):
                target       = y[0, f::3, half, half].abs().sum()
                grad         = torch.autograd.grad(target, x, retain_graph=True)[0]
                contribution = grad.abs().sum(dim=(0, 1)).cpu()

                family_totals[family] += contribution
                probe_map             += contribution

            if float(probe_map.sum()) > 0.0:
                probe_sigmas.append(self.sigma(probe_map.numpy()))

        combined = np.zeros((self.window, self.window))
        maps     = {}
        for family in self.FAMILIES:
            arr          = (family_totals[family] / windows.shape[0]).numpy()
            combined    += arr
            maps[family] = arr if arr.sum() > 0.0 else None

        if combined.sum() <= 0.0:
            raise ValueError("Receptive-field gradient map is all zero; the model output at the probe centre does not depend on the input window")

        maps["combined"] = combined
        return maps, np.asarray(probe_sigmas)

    def sigma(self, grad: np.ndarray) -> tuple[float, float]:
        """Returns the mass-weighted spread of a gradient map about the window centre.

        Args:
            grad : Gradient map of shape (window, window).

        Returns:
            The (azimuth, range) standard deviations in input pixels.
        """
        center = self.window // 2
        coords = np.arange(self.window, dtype=np.float64)
        rr, cc = np.meshgrid(coords, coords, indexing="ij")
        weight = grad / grad.sum()

        var_row = float((weight * (rr - center) ** 2).sum())
        var_col = float((weight * (cc - center) ** 2).sum())

        return math.sqrt(var_row), math.sqrt(var_col)

    def mass_profile(self, grad: np.ndarray) -> dict[int, float]:
        """Returns the gradient-mass fraction inside each configured window side.

        Args:
            grad : Gradient map of shape (window, window).

        Returns:
            Mapping from window side in pixels to the fraction of total gradient
            mass inside the centred box of that side.
        """
        center = self.window // 2
        total  = grad.sum()

        masses = {}
        for w in self.mass_windows:
            half      = w // 2
            masses[w] = float(grad[center - half:center + half, center - half:center + half].sum() / total)

        return masses

    def fine_mass_curve(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the cumulative gradient mass on a dense grid of window sides.

        Args:
            grad : Gradient map of shape (window, window).

        Returns:
            A tuple of the even window sides in pixels, shape (n_sides,), and the
            enclosed mass fraction at each side, same shape.
        """
        center = self.window // 2
        total  = grad.sum()
        sides  = np.arange(2, self.window + 1, 2)
        masses = np.array([grad[center - s // 2:center + s // 2, center - s // 2:center + s // 2].sum() / total for s in sides])

        return sides, masses

    @staticmethod
    def side_at_mass(sides: np.ndarray, masses: np.ndarray, q: float) -> float:
        """Interpolates the window side enclosing a target gradient-mass fraction.

        Args:
            sides  : Monotonically increasing window sides in pixels, shape (n_sides,).
            masses : Enclosed mass fraction at each side, shape (n_sides,).
            q      : Target mass fraction in [0, 1].

        Returns:
            The interpolated window side in pixels, clamped to the sampled range.
        """
        idx = int(np.searchsorted(masses, q))

        if idx == 0:
            return float(sides[0])
        if idx >= masses.size:
            return float(sides[-1])

        m0, m1 = masses[idx - 1], masses[idx]
        s0, s1 = sides[idx - 1], sides[idx]

        return float(s0 + (q - m0) / (m1 - m0) * (s1 - s0))

    def mass_within_2sigma(self, grad: np.ndarray, sigma_az: float, sigma_rg: float) -> float:
        """Returns the gradient-mass fraction inside the ±2σ box.

        A separable Gaussian ERF would place 0.911 of its mass there, so a smaller
        value indicates heavier-than-Gaussian tails.

        Args:
            grad     : Gradient map of shape (window, window).
            sigma_az : Azimuth spread in pixels.
            sigma_rg : Range spread in pixels.

        Returns:
            The enclosed mass fraction in [0, 1].
        """
        center  = self.window // 2
        half_az = min(int(round(2.0 * sigma_az)), center)
        half_rg = min(int(round(2.0 * sigma_rg)), center)

        box = grad[center - half_az:center + half_az + 1, center - half_rg:center + half_rg + 1]
        return float(box.sum() / grad.sum())

    def axis_profiles(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the normalized marginal gradient profiles along both axes.

        Args:
            grad : Gradient map of shape (window, window).

        Returns:
            A tuple of the azimuth marginal, the range marginal and the pixel
            offsets relative to the window centre, each of shape (window,).
        """
        total   = grad.sum()
        offsets = np.arange(self.window) - self.window // 2

        return grad.sum(axis=1) / total, grad.sum(axis=0) / total, offsets

    @staticmethod
    def gaussian_mass(sides: np.ndarray, sigma_az: float, sigma_rg: float) -> np.ndarray:
        """Returns the mass a separable Gaussian ERF places inside each window side.

        Args:
            sides    : Window sides in pixels, shape (n_sides,).
            sigma_az : Azimuth spread in pixels.
            sigma_rg : Range spread in pixels.

        Returns:
            The reference enclosed mass fraction per side, shape (n_sides,).

        Raises:
            ValueError : If either sigma is not positive.
        """
        if sigma_az <= 0.0 or sigma_rg <= 0.0:
            raise ValueError(f"Gaussian mass reference needs positive sigmas, got ({sigma_az}, {sigma_rg})")

        scale = 2.0 * math.sqrt(2.0)
        return np.array([math.erf(s / (scale * sigma_az)) * math.erf(s / (scale * sigma_rg)) for s in sides])


class ReceptiveFieldPlots(PlotBase):
    """Publication figures for effective-receptive-field maps and profiles."""

    @staticmethod
    def _mass_levels(grad: np.ndarray, quantiles: tuple[float, ...]) -> dict[float, str]:
        """Returns log10 contour levels enclosing the given cumulative mass quantiles."""
        flat       = np.sort(grad.reshape(-1))[::-1]
        cumulative = np.cumsum(flat) / flat.sum()

        levels = {}
        for q in quantiles:
            value = flat[min(int(np.searchsorted(cumulative, q)), flat.size - 1)]
            level = float(np.log10(value / grad.max() + 1e-12))
            levels[level] = f"{q * 100.0:.0f}%"

        return levels

    def _overlay_geometry(self, ax, grad: np.ndarray, sigma_az: float, sigma_rg: float) -> None:
        """Draws mass contours, 1σ/2σ ellipses and centre lines onto an ERF axis."""
        half    = grad.shape[0] // 2
        offsets = np.arange(grad.shape[0]) - half
        display = np.log10(grad / grad.max() + 1e-12)

        levels = self._mass_levels(grad, (0.5, 0.9, 0.99))
        if len(levels) > 1:
            ordered = sorted(levels)
            contour = ax.contour(offsets, offsets, display, levels=ordered, colors="white", linewidths=0.8, linestyles="dashed")
            ax.clabel(contour, fmt={level: levels[level] for level in ordered}, fontsize=7)

        for k, style in ((1.0, "-"), (2.0, ":")):
            ax.add_patch(Ellipse((0.0, 0.0), width=2.0 * k * sigma_col, z=2.0 * k * sigma_row, fill=False, edgecolor="#56B4E9", linewidth=1.0, linestyle=style))

        ax.axhline(0.0, color="white", linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color="white", linewidth=0.5, alpha=0.4)

    def erf_map(self, grad: np.ndarray, sigma_row: float, sigma_col: float, anisotropy: float, side_90: float, path: Path) -> Path:
        """Saves the combined ERF map on a log scale with sigma geometry overlaid.

        Args:
            grad       : Combined gradient map of shape (window, window).
            sigma_row   : Row spread in pixels.
            sigma_col   : Range spread in pixels.
            anisotropy : Ratio sigma_row / sigma_col.
            side_90    : Window side enclosing 90% of the mass, in pixels.
            path       : Destination image path.

        Returns:
            The written figure path.
        """
        half    = grad.shape[0] // 2
        display = np.log10(grad / grad.max() + 1e-12)

        fig = self._imshow_figure(
            display,
            x_label        = "Col offset [px]",
            y_label        = "Row offset [px]",
            title          = "Effective receptive field (combined output families)",
            cmap           = self._cmap_with_bad("inferno"),
            vmin           = -6.0,
            vmax           = 0.0,
            extent         = [-half, half, half, -half],
            colorbar_label = "log10 normalized |∂y/∂x|",
            text_overlay   = f"σ_az = {sigma_az:.2f} px\nσ_rg = {sigma_rg:.2f} px\nσ_az/σ_rg = {anisotropy:.2f}\ns90 = {side_90:.1f} px",
            path           = None,
        )

        self._overlay_geometry(fig.axes[0], grad, sigma_az, sigma_rg)
        return self._save(fig, path)

    def family_map(self, grad: np.ndarray, family: str, sigma_az: float, sigma_rg: float, path: Path) -> Path:
        """Saves the ERF map of one output family with its 90% mass contour.

        Args:
            grad     : Family gradient map of shape (window, window).
            family   : Output family name ("amp", "mu" or "sigma").
            sigma_az : Azimuth spread in pixels.
            sigma_rg : Range spread in pixels.
            path     : Destination image path.

        Returns:
            The written figure path.
        """
        half    = grad.shape[0] // 2
        display = np.log10(grad / grad.max() + 1e-12)

        fig = self._imshow_figure(
            display,
            x_label        = "Range offset [px]",
            y_label        = "Azimuth offset [px]",
            title          = f"Effective receptive field of predicted {family}",
            cmap           = self._cmap_with_bad("inferno"),
            vmin           = -6.0,
            vmax           = 0.0,
            extent         = [-half, half, half, -half],
            colorbar_label = "log10 normalized |∂y/∂x|",
            text_overlay   = f"σ_az = {sigma_az:.2f} px\nσ_rg = {sigma_rg:.2f} px",
            path           = None,
        )

        levels = self._mass_levels(grad, (0.9,))
        fig.axes[0].contour(np.arange(grad.shape[0]) - half, np.arange(grad.shape[0]) - half, display, levels=sorted(levels), colors="white", linewidths=0.8, linestyles="dashed")

        return self._save(fig, path)

    def mass_curve(self, sides: np.ndarray, fine_masses: np.ndarray, gaussian_masses: np.ndarray, marker_masses: dict[int, float], radii: dict[float, float], path: Path) -> Path:
        """Saves the cumulative gradient-mass curve against its Gaussian reference.

        Args:
            sides           : Window sides in pixels, shape (n_sides,).
            fine_masses     : Measured enclosed mass fraction per side, shape (n_sides,).
            gaussian_masses : Matched-Gaussian mass fraction per side, same shape.
            marker_masses   : Mass fraction at each configured window side.
            radii           : Mapping from mass quantile to the interpolated side in pixels.
            path            : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        ax.plot(sides, fine_masses * 100.0, color=self.OKABE_ITO[0], linewidth=1.4, label="measured")
        ax.plot(sides, gaussian_masses * 100.0, color=self.OKABE_ITO[1], linewidth=1.2, linestyle="--", label="matched Gaussian")
        ax.scatter(sorted(marker_masses), [marker_masses[w] * 100.0 for w in sorted(marker_masses)], s=26, color=self.OKABE_ITO[0], edgecolors="white", linewidths=0.6, zorder=4, label="configured windows")

        for index, (q, side) in enumerate(sorted(radii.items())):
            ax.axvline(side, color="0.45", linestyle=":", linewidth=0.9)
            ax.annotate(f"s{q * 100.0:.0f} = {side:.1f}", xy=(side, 4.0 + 14.0 * index), xytext=(3, 0), textcoords="offset points", fontsize=7, color="0.25", rotation=90)

        ax.set_xlabel("Window side [px]")
        ax.set_ylabel("Gradient mass inside window [%]")
        ax.set_title("Cumulative gradient mass by window")
        ax.set_ylim(0.0, 102.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)

    def axis_profile(self, offsets: np.ndarray, profile: np.ndarray, sigma: float, axis_name: str, path: Path) -> Path:
        """Saves a log-scale marginal ERF profile with ±1σ and ±2σ markers.

        Args:
            offsets   : Pixel offsets from the window centre, shape (window,).
            profile   : Normalized marginal gradient mass, shape (window,).
            sigma     : Spread along this axis in pixels.
            axis_name : Axis label, "azimuth" or "range".
            path      : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        floor = max(float(profile[profile > 0.0].min()) if (profile > 0.0).any() else 1e-12, 1e-12)
        ax.plot(offsets, np.maximum(profile, floor), color=self.OKABE_ITO[0], linewidth=1.4)

        for k, style in ((1.0, "--"), (2.0, ":")):
            ax.axvline(+k * sigma, color=self.OKABE_ITO[1], linestyle=style, linewidth=1.0, label=f"±{k:.0f}σ")
            ax.axvline(-k * sigma, color=self.OKABE_ITO[1], linestyle=style, linewidth=1.0)

        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        ax.set_yscale("log")
        ax.set_xlabel(f"{axis_name.capitalize()} offset [px]")
        ax.set_ylabel("Normalized gradient mass")
        ax.set_title(f"ERF marginal profile along {axis_name} (σ = {sigma:.2f} px)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class RunReceptiveField(AnalysisRun):
    """Measures and reports the effective receptive field of one trained run."""

    GRAD_FILENAME    = "erf_gradients.npz"
    SUMMARY_FILENAME = "receptive_field.json"
    REPORT_FILENAME  = "receptive_field.md"

    def _batch_size(self) -> int:
        """Returns the loader batch size, fixed at one window per gradient pass."""
        return 1

    def _write_summary(self, run, centers, measures: dict) -> Path:
        """Writes the ERF measures and probe centres as JSON.

        Args:
            run      : Loaded run providing the backbone name.
            centers  : Probe pixel centres as (row, col) index pairs.
            measures : ERF summary scalars and per-family entries.

        Returns:
            The written JSON path.
        """
        payload = {
            "backbone"   : run.backbone_name,
            "checkpoint" : self.config.checkpoint_name,
            "split"      : self.config.split,
            "window"     : self.config.window,
            "probes"     : [[row, col] for row, col in centers],
            **measures,
        }

        return FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

    def _write_report(self, run, measures: dict, masses: dict[int, float], figure_paths: dict[str, Path]) -> Path:
        """Writes the markdown ERF report with summary tables and figures.

        Args:
            run          : Loaded run providing the backbone name.
            measures     : ERF summary scalars and per-family entries.
            masses       : Enclosed mass fraction per configured window side.
            figure_paths : Mapping from figure name to its written path.

        Returns:
            The written report path.
        """
        doc = MarkdownDoc(title=f"Effective receptive field: {run.backbone_name}")
        doc.paragraph(
            f"Measured on the '{self.config.split}' split with {self.config.n_row_probes * self.config.n_col_probes} probe pixels and a {self.config.window}px window: "
            "gradient of the centre-pixel output magnitude with respect to the input window, averaged over probes, split by output family (amplitude, z, width). "
            "sX is the interpolated window side containing X% of the gradient mass. The Gaussian shape check compares the mass inside the ±2σ box "
            "against the 0.911 a separable Gaussian ERF would place there; a smaller value means heavier tails than Gaussian."
        )

        sigma_spread = measures["probe_sigma_spread"]
        doc.kv_table({
            "ERF sigma row"     : f"{measures['sigma_az_px']:.2f} px (± {sigma_spread['azimuth_std']:.2f} across probes)",
            "ERF sigma col"       : f"{measures['sigma_rg_px']:.2f} px (± {sigma_spread['range_std']:.2f} across probes)",
            "Anisotropy σ_az/σ_rg"  : f"{measures['anisotropy']:.2f}",
            "s50 / s90 / s99"       : " / ".join(f"{measures['mass_sides'][key]:.1f} px" for key in ("50", "90", "99")),
            "Mass in ±2σ box"       : f"{measures['mass_within_2sigma'] * 100.0:.1f}% (Gaussian reference 91.1%)",
        })

        doc.heading("Per-family ERF", level=2)
        family_table = MarkdownTable(("Output family", "σ azimuth", "σ range"))
        for family in ErfComputation.FAMILIES:
            entry = measures["families"][family]
            if entry is None:
                family_table.add_row(family, "no dependence", "no dependence")
            else:
                family_table.add_row(family, f"{entry['sigma_az_px']:.2f} px", f"{entry['sigma_rg_px']:.2f} px")
        doc.table(family_table)

        doc.heading("Gradient mass by window", level=2)
        table    = MarkdownTable(("Window", "Mass inside", "Marginal"))
        previous = 0.0
        for w in sorted(masses):
            table.add_row(f"{w}x{w}", f"{masses[w] * 100.0:.2f}%", f"{(masses[w] - previous) * 100.0:.2f}%")
            previous = masses[w]
        doc.table(table)

        doc.heading("Figures", level=2)
        for name, path in figure_paths.items():
            doc.image(name, path.name)

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        """Measures the ERF, writes the gradient maps, figures, JSON and report.

        Returns:
            The ERF measures plus the "report" path of the written markdown.
        """
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run     = self._load_run()
        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_row_probes, self.config.n_col_probes)
        windows = probes.assemble(centers)

        erf                = ErfComputation(run.model.module, self.config.window, self.config.mass_windows)
        maps, probe_sigmas = erf.gradient_maps(windows)
        grad               = maps["combined"]

        sigma_row, sigma_col = erf.sigma(grad)
        masses             = erf.mass_profile(grad)
        sides, fine        = erf.fine_mass_curve(grad)
        radii              = {q: erf.side_at_mass(sides, fine, q) for q in (0.50, 0.90, 0.99)}
        anisotropy         = sigma_az / sigma_rg
        within_2sigma      = erf.mass_within_2sigma(grad, sigma_az, sigma_rg)
        az_prof, rg_prof, offsets = erf.axis_profiles(grad)

        families = {}
        for family in ErfComputation.FAMILIES:
            if maps[family] is None:
                families[family] = None
            else:
                fam_az, fam_rg   = erf.sigma(maps[family])
                families[family] = {"sigma_az_px": fam_az, "sigma_rg_px": fam_rg}

        measures = {
            "sigma_az_px"        : sigma_az,
            "sigma_rg_px"        : sigma_rg,
            "anisotropy"         : anisotropy,
            "mass_by_window"     : {str(w): mass for w, mass in sorted(masses.items())},
            "mass_sides"         : {f"{q * 100.0:.0f}": side for q, side in radii.items()},
            "mass_within_2sigma" : within_2sigma,
            "families"           : families,
            "probe_sigma_spread" : {
                "azimuth_std" : float(probe_sigmas[:, 0].std()),
                "range_std"   : float(probe_sigmas[:, 1].std()),
            },
        }

        np.savez_compressed(self.output_dir / self.GRAD_FILENAME, **{name: arr.astype(np.float32) for name, arr in maps.items() if arr is not None})

        plots        = ReceptiveFieldPlots()
        figure_paths = {
            "erf_map"          : plots.erf_map(grad, sigma_row, sigma_col, anisotropy, radii[0.90], self.output_dir / "erf_map.png"),
            "mass_curve"       : plots.mass_curve(sides, fine, erf.gaussian_mass(sides, sigma_row, sigma_col), masses, radii, self.output_dir / "mass_curve.png"),
            "profile_azimuth"  : plots.axis_profile(offsets, az_prof, sigma_row, "row", self.output_dir / "profile_azimuth.png"),
            "profile_range"    : plots.axis_profile(offsets, rg_prof, sigma_col, "col", self.output_dir / "profile_range.png"),
        }
        for family in ErfComputation.FAMILIES:
            if maps[family] is not None:
                figure_paths[f"erf_{family}"] = plots.family_map(maps[family], family, families[family]["sigma_az_px"], families[family]["sigma_rg_px"], self.output_dir / f"erf_{family}.png")

        self._write_summary(run, centers, measures)
        report_path = self._write_report(run, measures, masses, figure_paths)

        self.logger.ok(f"{self.run_dir.name}: ERF σ row {sigma_row:.2f} px, col {sigma_col:.2f} px, s90 {radii[0.90]:.1f} px -> {report_path}")

        return {**measures, "report": report_path}


class ReceptiveFieldBatch(RunBatch):
    """Measures the effective receptive field of every selected run."""

    SELECTOR_ACTION = "measure"
    SECTION_TITLE   = "Receptive-field measurement"
    RUN_CLASS       = RunReceptiveField
