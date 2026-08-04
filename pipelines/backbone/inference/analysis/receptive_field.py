from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice, ProbeWindows
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class ErfComputation:

    def __init__(self, model, window: int, mass_windows: list[int]) -> None:
        self.model        = model
        self.window       = int(window)
        self.mass_windows = [int(w) for w in mass_windows]

        bad = [w for w in self.mass_windows if w > self.window]
        if bad:
            raise ValueError(f"mass_windows {bad} exceed the probe window {self.window}; shrink them or enlarge window")

    def gradient_map(self, windows: torch.Tensor) -> np.ndarray:
        half        = self.window // 2
        device      = ModelDevice.of(self.model)
        accumulated = torch.zeros(self.window, self.window)

        for p in range(windows.shape[0]):
            x = windows[p:p + 1].clone().to(device).requires_grad_(True)
            y = self.model(x)
            y[0, :, half, half].abs().sum().backward()
            accumulated += x.grad.abs().sum(dim=(0, 1)).cpu()

        grad  = accumulated / windows.shape[0]
        total = float(grad.sum())
        if total <= 0.0:
            raise ValueError("Receptive-field gradient map is all zero; the model output at the probe centre does not depend on the input window")

        return grad.numpy()

    def sigma(self, grad: np.ndarray) -> tuple[float, float]:
        center = self.window // 2
        coords = np.arange(self.window, dtype=np.float64)
        rr, cc = np.meshgrid(coords, coords, indexing="ij")
        weight = grad / grad.sum()

        var_row = float((weight * (rr - center) ** 2).sum())
        var_col = float((weight * (cc - center) ** 2).sum())

        return math.sqrt(var_row), math.sqrt(var_col)

    def mass_profile(self, grad: np.ndarray) -> dict[int, float]:
        center = self.window // 2
        total  = grad.sum()

        masses = {}
        for w in self.mass_windows:
            half      = w // 2
            masses[w] = float(grad[center - half:center + half, center - half:center + half].sum() / total)

        return masses


class ReceptiveFieldPlots(PlotBase):

    def erf_map(self, grad: np.ndarray, sigma_row: float, sigma_col: float, path: Path) -> Path:
        display = np.log10(grad / grad.max() + 1e-12)

        return self._imshow_figure(
            display,
            x_label        = "Col offset [px]",
            y_label        = "Row offset [px]",
            title          = f"Effective receptive field (log10, σ_row {sigma_row:.1f} px, σ_col {sigma_col:.1f} px)",
            cmap           = self._cmap_with_bad("inferno"),
            vmin           = -6.0,
            vmax           = 0.0,
            colorbar_label = "log10 normalized |∂y/∂x|",
            path           = path,
        )

    def mass_curve(self, masses: dict[int, float], path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        windows = sorted(masses)
        ax.plot(windows, [masses[w] * 100.0 for w in windows], marker="o")
        ax.set_xlabel("Window side [px]")
        ax.set_ylabel("Gradient mass inside window [%]")
        ax.set_title("Cumulative gradient mass by window")
        ax.set_ylim(0.0, 102.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class RunReceptiveField(AnalysisRun):

    GRAD_FILENAME    = "erf_gradient.npy"
    SUMMARY_FILENAME = "receptive_field.json"
    REPORT_FILENAME  = "receptive_field.md"

    def _batch_size(self) -> int:
        return 1

    def _write_summary(self, run, centers, sigma_row: float, sigma_col: float, masses: dict[int, float]) -> Path:
        payload = {
            "backbone"        : run.backbone_name,
            "checkpoint"      : self.config.checkpoint_name,
            "split"           : self.config.split,
            "window"          : self.config.window,
            "probes"          : [[row, col] for row, col in centers],
            "sigma_az_px"     : sigma_row,
            "sigma_rg_px"     : sigma_col,
            "mass_by_window"  : {str(w): mass for w, mass in sorted(masses.items())},
        }

        return FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

    def _write_report(self, run, sigma_row: float, sigma_col: float, masses: dict[int, float], figure_paths: dict[str, Path]) -> Path:
        doc = MarkdownDoc(title=f"Effective receptive field: {run.backbone_name}")
        doc.paragraph(f"Measured on the '{self.config.split}' split with {self.config.n_row_probes * self.config.n_col_probes} probe pixels and a {self.config.window}px window: gradient of the centre-pixel output magnitude with respect to the input window, averaged over probes.")

        doc.kv_table({
            "ERF sigma row" : f"{sigma_row:.2f} px",
            "ERF sigma col" : f"{sigma_col:.2f} px",
        })

        table    = MarkdownTable(("Window", "Mass inside", "Marginal"))
        previous = 0.0
        for w in sorted(masses):
            table.add_row(f"{w}x{w}", f"{masses[w] * 100.0:.2f}%", f"{(masses[w] - previous) * 100.0:.2f}%")
            previous = masses[w]
        doc.table(table)

        for name, path in figure_paths.items():
            doc.image(name, path.name)

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run     = self._load_run()
        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_row_probes, self.config.n_col_probes)
        windows = probes.assemble(centers)

        erf  = ErfComputation(run.model.module, self.config.window, self.config.mass_windows)
        grad = erf.gradient_map(windows)

        sigma_row, sigma_col = erf.sigma(grad)
        masses             = erf.mass_profile(grad)

        np.save(self.output_dir / self.GRAD_FILENAME, grad.astype(np.float32))

        plots        = ReceptiveFieldPlots()
        figure_paths = {
            "erf_map"    : plots.erf_map(grad, sigma_row, sigma_col, self.output_dir / "erf_map.png"),
            "mass_curve" : plots.mass_curve(masses, self.output_dir / "mass_curve.png"),
        }

        self._write_summary(run, centers, sigma_row, sigma_col, masses)
        report_path = self._write_report(run, sigma_row, sigma_col, masses, figure_paths)

        self.logger.ok(f"{self.run_dir.name}: ERF σ row {sigma_row:.2f} px, col {sigma_col:.2f} px -> {report_path}")

        return {"sigma_row": sigma_row, "sigma_col": sigma_col, "masses": masses, "report": report_path}


class ReceptiveFieldBatch(RunBatch):

    SELECTOR_ACTION = "measure"
    SECTION_TITLE   = "Receptive-field measurement"
    RUN_CLASS       = RunReceptiveField
