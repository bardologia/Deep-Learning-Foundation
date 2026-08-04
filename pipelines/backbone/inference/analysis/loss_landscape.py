from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import PredictionCurves
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc
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

        if all(tensor.abs().sum() == 0 for tensor in direction.values()):
            raise ValueError("Filter-normalized direction is all zero; the model holds no weight tensor with 2 or more dimensions")

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

    def grid(self, alphas: np.ndarray, betas: np.ndarray, dir1: dict, dir2: dict) -> np.ndarray:
        losses = np.zeros((alphas.size, betas.size))

        try:
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    self._apply(float(alpha), float(beta), dir1, dir2)
                    losses[i, j] = self.loss_fn()
        finally:
            self.restore()

        return losses


class LossLandscapePlots(PlotBase):

    def cut_1d(self, alphas: np.ndarray, losses: np.ndarray, label: str, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(alphas, losses, marker=".", color="#0072B2", linewidth=1.4)
        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"Step along {label} (filter-normalized)")
        ax.set_ylabel("Curve MSE")
        ax.set_yscale("log")
        ax.set_title(f"Loss cut along {label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)

    def contour_2d(self, alphas: np.ndarray, betas: np.ndarray, losses: np.ndarray, path: Path) -> Path:
        if not losses.max() > losses.min():
            raise ValueError(f"The sampled landscape is flat at {float(losses.min()):.6g} over the whole grid; a filled contour needs a loss range, so widen span or check that the loss depends on the perturbed weights")

        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.85))
        levels  = np.logspace(np.log10(max(losses.min(), 1e-12)), np.log10(losses.max()), 20)
        contour = ax.contourf(betas, alphas, losses, levels=levels, cmap="magma", norm=plt.matplotlib.colors.LogNorm())

        ax.plot(0.0, 0.0, marker="*", color="white", markersize=10)
        ax.set_xlabel("Direction 2")
        ax.set_ylabel("Direction 1")
        ax.set_title("Loss landscape around the trained weights")
        fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.02).set_label("Curve MSE")
        fig.tight_layout()

        return self._save(fig, path)


class LossLandscapeRun(AnalysisRun):

    SUMMARY_FILENAME = "loss_landscape.json"
    REPORT_FILENAME  = "loss_landscape.md"

    def _loss_fn(self, run):
        renderer = PredictionCurves(run.n_gaussians, run.x_axis)
        batches  = []

        for index, batch in enumerate(run.loader):
            if index >= self.config.max_batches:
                break
            gt_phys = run.dataset.normalizer.denormalize_output(batch[1].float()).numpy()
            batches.append((batch[0], renderer.render(gt_phys)))

        if not batches:
            raise ValueError("The loader yielded no batches to evaluate the landscape on")

        def evaluate() -> float:
            total = 0.0
            for images, gt_curves in batches:
                pred_curves = renderer.render(run.model(images))
                total      += float(((pred_curves - gt_curves) ** 2).mean())
            return total / len(batches)

        return evaluate

    @staticmethod
    def _sharpness(alphas: np.ndarray, cut: np.ndarray) -> float:
        center = cut[np.argmin(np.abs(alphas))]
        near   = np.abs(alphas) <= 0.1 + 1e-9

        return float(cut[near].max() / max(center, 1e-12) - 1.0)

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run     = self._load_run()
        loss_fn = self._loss_fn(run)
        module  = run.model.module

        dir1 = FilterNormalizedDirection.build(module, seed=self.config.direction_seed)
        dir2 = FilterNormalizedDirection.build(module, seed=self.config.direction_seed + 1)

        evaluator = LandscapeEvaluator(module, loss_fn)

        alphas_1d = np.linspace(-self.config.span, self.config.span, self.config.n_points_1d)
        cut1      = evaluator.grid(alphas_1d, np.zeros(1), dir1, dir2)[:, 0]
        cut2      = evaluator.grid(np.zeros(1), alphas_1d, dir1, dir2)[0, :]

        alphas_2d = np.linspace(-self.config.span, self.config.span, self.config.n_points_2d)
        grid      = evaluator.grid(alphas_2d, alphas_2d, dir1, dir2)

        plots   = LossLandscapePlots()
        figures = {
            "cut_dir1" : plots.cut_1d(alphas_1d, cut1, "direction 1", self.output_dir / "cut_dir1.png"),
            "cut_dir2" : plots.cut_1d(alphas_1d, cut2, "direction 2", self.output_dir / "cut_dir2.png"),
            "contour"  : plots.contour_2d(alphas_2d, alphas_2d, grid, self.output_dir / "contour.png"),
        }

        sharpness = {
            "sharpness_dir1" : self._sharpness(alphas_1d, cut1),
            "sharpness_dir2" : self._sharpness(alphas_1d, cut2),
        }

        payload = {
            "backbone"  : run.backbone_name,
            "split"     : self.config.split,
            "span"      : self.config.span,
            "alphas_1d" : alphas_1d.tolist(),
            "cut_dir1"  : cut1.tolist(),
            "cut_dir2"  : cut2.tolist(),
            "alphas_2d" : alphas_2d.tolist(),
            "grid"      : grid.tolist(),
            **sharpness,
        }
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        doc = MarkdownDoc(title=f"Loss landscape: {run.backbone_name}")
        doc.paragraph(
            f"Curve-MSE landscape around the trained weights along two filter-normalized random directions (Li et al. 2018 convention), "
            f"evaluated on {self.config.max_batches} '{self.config.split}' batches. Sharpness is the relative loss increase within ±0.1 of the minimum: "
            f"direction 1 {sharpness['sharpness_dir1']:.3f}, direction 2 {sharpness['sharpness_dir2']:.3f}. "
            "The objective is the physical curve MSE, not the training loss, so runs with different loss configurations stay comparable."
        )
        for name, path in figures.items():
            doc.image(name, path.name)
        report_path = doc.save(self.output_dir / self.REPORT_FILENAME)

        self.logger.ok(f"{self.run_dir.name}: sharpness {sharpness['sharpness_dir1']:.3f}/{sharpness['sharpness_dir2']:.3f} -> {report_path}")

        return payload


class LossLandscapeBatch(RunBatch):

    SELECTOR_ACTION = "map"
    SECTION_TITLE   = "Loss landscape"
    RUN_CLASS       = LossLandscapeRun
