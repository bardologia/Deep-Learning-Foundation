
from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisOutcome, AnalysisRun, RunBatch
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class FilterNormalizedDirection:

    @staticmethod
    def build(model: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        direction = {}

        for name, param in model.named_parameters():
            if param.ndim <= 1:
                direction[name] = torch.zeros_like(param)
                continue

            random = torch.randn(param.shape, generator=generator).to(param.device)
            flat_w = param.detach().reshape(param.shape[0], -1)
            flat_d = random.reshape(param.shape[0], -1)

            scale  = flat_w.norm(dim=1, keepdim=True) / flat_d.norm(dim=1, keepdim=True).clamp_min(1e-10)
            direction[name] = (flat_d * scale).reshape(param.shape)

        return direction


class LandscapeEvaluator:

    def __init__(self, model: torch.nn.Module, loss_fn) -> None:
        self.model   = model
        self.loss_fn = loss_fn

        self.origin = {name: param.detach().clone() for name, param in model.named_parameters()}

    def _apply(self, alpha: float, beta: float, dir1: dict, dir2: dict) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.origin[name] + alpha * dir1[name].to(param.device) + beta * dir2[name].to(param.device))

    def restore(self) -> None:
        self._apply(0.0, 0.0, {name: torch.zeros_like(t) for name, t in self.origin.items()}, {name: torch.zeros_like(t) for name, t in self.origin.items()})

    def grid(self, alphas: np.ndarray, betas: np.ndarray, dir1: dict, dir2: dict, on_point) -> np.ndarray:
        losses = np.zeros((alphas.size, betas.size))

        try:
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    self._apply(float(alpha), float(beta), dir1, dir2)
                    losses[i, j] = self.loss_fn()
                    on_point()
        finally:
            self.restore()

        return losses


class CutDiagnostics:

    SHARPNESS_RADIUS = 0.1
    CURVATURE_RADIUS = 0.2
    RADIUS_FACTOR    = 2.0

    @classmethod
    def sharpness(cls, alphas: np.ndarray, cut: np.ndarray) -> float:
        center = cut[np.argmin(np.abs(alphas))]
        near   = np.abs(alphas) <= cls.SHARPNESS_RADIUS + 1e-9

        return float(cut[near].max() / max(center, 1e-12) - 1.0)

    @classmethod
    def curvature(cls, alphas: np.ndarray, cut: np.ndarray) -> float:
        center = cut[np.argmin(np.abs(alphas))]
        near   = np.abs(alphas) <= cls.CURVATURE_RADIUS + 1e-9

        design       = np.stack([np.ones(int(near.sum())), alphas[near] ** 2], axis=1)
        coefficients = np.linalg.lstsq(design, cut[near], rcond=None)[0]

        return float(coefficients[1] / max(center, 1e-12))

    @classmethod
    def flatness_radius(cls, alphas: np.ndarray, cut: np.ndarray) -> dict:
        center    = cut[np.argmin(np.abs(alphas))]
        threshold = cls.RADIUS_FACTOR * max(center, 1e-12)
        span      = float(np.abs(alphas).max())

        sides = []
        for sign in (1.0, -1.0):
            side_alphas = alphas * sign
            keep        = side_alphas >= 0.0
            ordered     = np.argsort(side_alphas[keep])
            a           = side_alphas[keep][ordered]
            l           = cut[keep][ordered]

            above = np.where(l > threshold)[0]
            if above.size == 0 or above[0] == 0:
                sides.append(span if above.size == 0 else 0.0)
                continue

            k        = above[0]
            fraction = (threshold - l[k - 1]) / max(l[k] - l[k - 1], 1e-300)
            sides.append(float(a[k - 1] + fraction * (a[k] - a[k - 1])))

        radius = min(sides)
        return {"radius": float(radius), "censored": bool(radius >= span)}

    @classmethod
    def min_offset(cls, alphas: np.ndarray, cut: np.ndarray) -> float:
        return float(alphas[int(np.argmin(cut))])


class LossLandscapePlots(PlotBase):

    def cuts_overlay(self, alphas: np.ndarray, cuts: list[np.ndarray], curvatures: list[float], center: float, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.55))

        for index, (cut, curvature) in enumerate(zip(cuts, curvatures)):
            ax.plot(alphas, cut, marker=".", markersize=4, linewidth=1.2, color=self.OKABE_ITO[index % len(self.OKABE_ITO)], label=f"direction {index} (curv. {curvature:.2g})")

        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.axhline(CutDiagnostics.RADIUS_FACTOR * max(center, 1e-12), color="0.4", linestyle=":", linewidth=1.0, label=f"{CutDiagnostics.RADIUS_FACTOR:.0f}× centre loss")

        ax.set_xlabel("Step along the direction (filter-normalized)")
        ax.set_ylabel("Weighted MSE (normalized target units)")
        ax.set_yscale("log")
        ax.set_title("Loss cuts along random filter-normalized directions")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)

    def contour_2d(self, alphas: np.ndarray, betas: np.ndarray, losses: np.ndarray, path: Path, annotation: str | None = None) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.85))
        levels  = np.logspace(np.log10(max(losses.min(), 1e-12)), np.log10(losses.max()), 20)
        filled  = ax.contourf(betas, alphas, losses, levels=levels, cmap="magma", norm=plt.matplotlib.colors.LogNorm())
        lines   = ax.contour(betas, alphas, losses, levels=levels[::4], colors="white", linewidths=0.5, alpha=0.7)
        ax.clabel(lines, fontsize=6, fmt="%.2g")

        ax.plot(0.0, 0.0, marker="*", color="white", markersize=11, markeredgecolor="black", markeredgewidth=0.5)

        if annotation is not None:
            ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        ax.set_xlabel("Direction 1 (filter-normalized)")
        ax.set_ylabel("Direction 0 (filter-normalized)")
        ax.set_title("Loss landscape around the trained weights")
        fig.colorbar(filled, ax=ax, fraction=0.045, pad=0.02).set_label("Weighted MSE (normalized target units)")
        fig.tight_layout()

        return self._save(fig, path)


class LossLandscapeRun(AnalysisRun):

    SUMMARY_FILENAME = "loss_landscape.json"
    REPORT_FILENAME  = "loss_landscape.md"

    def _loss_fn(self, run):
        module  = run.model.module
        device  = run.model.device
        batches = [batch for _, batch in zip(range(self.config.max_batches), run.loader)]

        def evaluate() -> float:
            total, mass = 0.0, 0.0

            with torch.no_grad():
                for inputs, targets, weights in batches:
                    error  = (module(inputs.to(device)) - targets.to(device)) ** 2
                    pixel  = weights.to(device)
                    total += float((error * pixel).sum())
                    mass  += float(pixel.sum()) * error.shape[1]

            return total / mass

        return evaluate, len(batches)

    def _diagnose(self, alphas: np.ndarray, cuts: list[np.ndarray]) -> list[dict]:
        diagnostics = []
        for index, cut in enumerate(cuts):
            nonfinite = alphas[~np.isfinite(cut)]

            if nonfinite.size:
                self.logger.warning(f"{self.run_dir.name}: direction {index}: {nonfinite.size}/{cut.size} cut losses are non-finite at alpha {', '.join(f'{alpha:+.3f}' for alpha in nonfinite)}; the cut is left out of the diagnostics")
                diagnostics.append({"valid": False, "nonfinite_alphas": nonfinite.tolist()})
                continue

            diagnostics.append({
                "valid"            : True,
                "nonfinite_alphas" : [],
                "sharpness"        : CutDiagnostics.sharpness(alphas, cut),
                "curvature"        : CutDiagnostics.curvature(alphas, cut),
                "flatness_radius"  : CutDiagnostics.flatness_radius(alphas, cut),
                "min_offset"       : CutDiagnostics.min_offset(alphas, cut),
            })

        return diagnostics

    def _write_report(self, run, center: float, diagnostics: list[dict], figures: dict[str, Path], n_batches: int) -> Path:
        valid      = [entry for entry in diagnostics if entry["valid"]]
        sharpness  = np.array([entry["sharpness"] for entry in valid])
        curvature  = np.array([entry["curvature"] for entry in valid])
        radii      = np.array([entry["flatness_radius"]["radius"] for entry in valid])

        doc = MarkdownDoc(title=f"Loss landscape: {run.backbone_name}")
        doc.paragraph(
            f"Weighted MSE landscape around the trained weights along {len(diagnostics)} filter-normalized random directions (Li et al. 2018 convention), "
            f"evaluated on {n_batches} '{self.config.split}' batches. Sharpness is the relative loss increase within ±{CutDiagnostics.SHARPNESS_RADIUS} of the centre; "
            f"curvature is the normalized quadratic coefficient fitted within ±{CutDiagnostics.CURVATURE_RADIUS}; the flatness radius is the interpolated step at which the loss "
            f"doubles (censored at the sampled span when it never does); the minimum offset flags cuts whose sampled minimum is not the trained point. "
            "The objective is the pixel-weighted mean squared error of the normalized prediction over every target channel, not the training loss, so runs with different loss configurations stay comparable. "
            "A cut holding a non-finite loss is left out of the summary and listed with the steps where the perturbed model diverged."
        )

        doc.kv_table({
            "Centre loss"        : f"{center:.4g}",
            "Valid directions"   : f"{len(valid)} of {len(diagnostics)}",
            "Sharpness"          : f"{sharpness.mean():.3f} ± {sharpness.std():.3f}",
            "Curvature"          : f"{curvature.mean():.3g} ± {curvature.std():.3g}",
            "Flatness radius"    : f"{radii.mean():.3f} ± {radii.std():.3f}",
            "Sharpest direction" : diagnostics.index(valid[int(np.argmax(sharpness))]),
        })

        table = MarkdownTable(("Direction", "Sharpness", "Curvature", "Flatness radius", "Min offset"))
        for index, entry in enumerate(diagnostics):
            if not entry["valid"]:
                note = "non-finite at alpha " + ", ".join(f"{alpha:+.3f}" for alpha in entry["nonfinite_alphas"])
                table.add_row(str(index), note, "—", "—", "—")
                continue
            radius = entry["flatness_radius"]
            table.add_row(str(index), f"{entry['sharpness']:.3f}", f"{entry['curvature']:.3g}", f"{radius['radius']:.3f}{' (≥ span)' if radius['censored'] else ''}", f"{entry['min_offset']:+.3f}")
        doc.table(table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, path.name)

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def _analyse(self, run) -> AnalysisOutcome:
        loss_fn, n_batches = self._loss_fn(run)
        module             = run.model.module

        directions = [FilterNormalizedDirection.build(module, seed=self.config.direction_seed + index) for index in range(self.config.n_directions)]
        evaluator  = LandscapeEvaluator(module, loss_fn)

        alphas_1d = np.linspace(-self.config.span, self.config.span, self.config.n_points_1d)
        alphas_2d = np.linspace(-self.config.span, self.config.span, self.config.n_points_2d)
        n_cut     = len(directions) * alphas_1d.size
        n_plane   = alphas_2d.size ** 2

        self.logger.info(f"{self.run_dir.name}: {n_cut + n_plane} loss evaluations planned ({len(directions)} cuts x {alphas_1d.size} points + {alphas_2d.size}x{alphas_2d.size} grid)")

        with self.logger.track() as progress:
            task     = progress.add_task("[section]Loss landscape[/section]", total=n_cut + n_plane)
            on_point = functools.partial(progress.advance, task)

            with self.logger.timer(f"Loss landscape: 1-D cuts ({n_cut} points)"):
                cuts = [evaluator.grid(alphas_1d, np.zeros(1), direction, directions[0], on_point)[:, 0] for direction in directions]

            with self.logger.timer(f"Loss landscape: 2-D grid ({n_plane} points)"):
                grid = evaluator.grid(alphas_2d, alphas_2d, directions[0], directions[1], on_point)

        center = float(loss_fn())

        diagnostics = self._diagnose(alphas_1d, cuts)
        valid       = [entry for entry in diagnostics if entry["valid"]]
        sharpness   = np.array([entry["sharpness"] for entry in valid])
        radii       = np.array([entry["flatness_radius"]["radius"] for entry in valid])

        plots   = LossLandscapePlots()
        figures = {
            "cuts"    : plots.cuts_overlay(alphas_1d, cuts, [entry["curvature"] if entry["valid"] else float("nan") for entry in diagnostics], center, self.output_dir / "cuts.png"),
            "contour" : plots.contour_2d(alphas_2d, alphas_2d, grid, self.output_dir / "contour.png", annotation=f"sharpness = {sharpness.mean():.2f} ± {sharpness.std():.2f}\nflatness radius = {radii.mean():.2f}"),
        }

        payload = {
            "backbone"    : run.backbone_name,
            "split"       : self.config.split,
            "n_batches"   : n_batches,
            "span"        : self.config.span,
            "center_loss" : center,
            "alphas_1d"   : alphas_1d.tolist(),
            "cuts"        : [cut.tolist() for cut in cuts],
            "alphas_2d"   : alphas_2d.tolist(),
            "grid"        : grid.tolist(),
            "directions"  : diagnostics,
        }
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, center, diagnostics, figures, n_batches)

        return AnalysisOutcome(payload, f"sharpness {sharpness.mean():.3f} ± {sharpness.std():.3f} over {len(valid)} of {len(cuts)} directions", report_path)


class LossLandscapeBatch(RunBatch):

    SELECTOR_ACTION = "map"
    SECTION_TITLE   = "Loss landscape"
    RUN_CLASS       = LossLandscapeRun
