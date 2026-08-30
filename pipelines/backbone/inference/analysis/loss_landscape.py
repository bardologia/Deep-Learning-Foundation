"""Loss landscape of a trained backbone around its converged weights.

Perturbs the weights along filter-normalized random directions in the Li et al. (2018)
convention, evaluates the physical curve MSE on real batches, and reports sharpness,
curvature and flatness radius alongside 1-D cuts and a 2-D contour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import PredictionCurves
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class FilterNormalizedDirection:
    """Random weight-space directions rescaled filter by filter."""

    @staticmethod
    def build(model: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
        """Draws a random direction whose filters carry the norm of the trained filters.

        Parameters with one dimension or fewer, such as biases and norm scales, get a zero
        direction so they are left untouched.

        Args:
            model : Model whose parameter shapes and norms define the direction.
            seed  : Seed of the direction generator.

        Returns:
            One perturbation tensor per parameter name, matching that parameter's shape.

        Raises:
            ValueError : If the model holds no parameter with two or more dimensions, which would leave the direction all zero.
        """
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
    """Evaluates a loss on a grid of weight perturbations around the trained point.

    Attributes:
        model   : Model whose weights are perturbed in place.
        loss_fn : Callable returning the scalar loss at the current weights.
        origin  : Clone of the trained weights, restored after every sweep.
    """

    def __init__(self, model: torch.nn.Module, loss_fn) -> None:
        """Snapshots the trained weights as the landscape origin.

        Args:
            model   : Model to perturb.
            loss_fn : Zero-argument callable returning the loss at the current weights.
        """
        self.model   = model
        self.loss_fn = loss_fn

        self.origin = {name: param.detach().clone() for name, param in model.named_parameters()}

    def _apply(self, alpha: float, beta: float, dir1: dict, dir2: dict) -> None:
        """Sets the weights to the origin displaced by alpha along dir1 and beta along dir2.

        Args:
            alpha : Step along the first direction.
            beta  : Step along the second direction.
            dir1  : Perturbation tensors of the first direction, keyed by parameter name.
            dir2  : Perturbation tensors of the second direction, keyed by parameter name.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.origin[name] + alpha * dir1[name].to(param.device) + beta * dir2[name].to(param.device))

    def restore(self) -> None:
        """Puts the trained weights back in place."""
        self._apply(0.0, 0.0, {name: torch.zeros_like(t) for name, t in self.origin.items()}, {name: torch.zeros_like(t) for name, t in self.origin.items()})

    def grid(self, alphas: np.ndarray, betas: np.ndarray, dir1: dict, dir2: dict) -> np.ndarray:
        """Evaluates the loss on the Cartesian product of the two step grids.

        The trained weights are restored even when the evaluation raises.

        Args:
            alphas : Steps along the first direction, shape (n_alpha,).
            betas  : Steps along the second direction, shape (n_beta,).
            dir1   : Perturbation tensors of the first direction.
            dir2   : Perturbation tensors of the second direction.

        Returns:
            Loss array of shape (n_alpha, n_beta).
        """
        losses = np.zeros((alphas.size, betas.size))

        try:
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    self._apply(float(alpha), float(beta), dir1, dir2)
                    losses[i, j] = self.loss_fn()
        finally:
            self.restore()

        return losses


class CutDiagnostics:
    """Scalar descriptors of a 1-D loss cut through the trained point."""

    SHARPNESS_RADIUS = 0.1
    CURVATURE_RADIUS = 0.2
    RADIUS_FACTOR    = 2.0

    @classmethod
    def sharpness(cls, alphas: np.ndarray, cut: np.ndarray) -> float:
        """Returns the relative loss increase within SHARPNESS_RADIUS of the trained point.

        Args:
            alphas : Steps along the direction, shape (n_points,).
            cut    : Loss at each step, shape (n_points,).

        Returns:
            The largest nearby loss over the centre loss, minus one.
        """
        center = cut[np.argmin(np.abs(alphas))]
        near   = np.abs(alphas) <= cls.SHARPNESS_RADIUS + 1e-9

        return float(cut[near].max() / max(center, 1e-12) - 1.0)

    @classmethod
    def curvature(cls, alphas: np.ndarray, cut: np.ndarray) -> float:
        """Fits a quadratic near the trained point and returns its normalized coefficient.

        Args:
            alphas : Steps along the direction, shape (n_points,).
            cut    : Loss at each step, shape (n_points,).

        Returns:
            The quadratic coefficient of the least-squares fit within CURVATURE_RADIUS,
            divided by the centre loss.

        Raises:
            ValueError : If fewer than three samples fall inside the fitting radius.
        """
        center = cut[np.argmin(np.abs(alphas))]
        near   = np.abs(alphas) <= cls.CURVATURE_RADIUS + 1e-9

        if near.sum() < 3:
            raise ValueError(f"Curvature fit needs at least 3 samples within |alpha| <= {cls.CURVATURE_RADIUS}; refine the 1-D grid")

        design       = np.stack([np.ones(int(near.sum())), alphas[near] ** 2], axis=1)
        coefficients = np.linalg.lstsq(design, cut[near], rcond=None)[0]

        return float(coefficients[1] / max(center, 1e-12))

    @classmethod
    def flatness_radius(cls, alphas: np.ndarray, cut: np.ndarray) -> dict:
        """Finds the step at which the loss first reaches RADIUS_FACTOR times its centre value.

        Both sides of the cut are scanned and the smaller radius is kept; the crossing is
        interpolated linearly between the bracketing samples.

        Args:
            alphas : Steps along the direction, shape (n_points,).
            cut    : Loss at each step, shape (n_points,).

        Returns:
            Dictionary with the "radius" in direction steps and a "censored" flag set when
            the loss never crosses within the sampled span.
        """
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

    @staticmethod
    def min_offset(alphas: np.ndarray, cut: np.ndarray) -> float:
        """Returns the step at which the cut attains its sampled minimum, zero when at the trained point."""
        return float(alphas[int(np.argmin(cut))])


class LossLandscapePlots(PlotBase):
    """Figures of the 1-D loss cuts and the 2-D landscape contour."""

    def cuts_overlay(self, alphas: np.ndarray, cuts: list[np.ndarray], curvatures: list[float], center: float, path: Path) -> Path:
        """Overlays the loss cuts of every sampled direction on a log axis.

        Args:
            alphas     : Steps along the directions, shape (n_points,).
            cuts       : Loss of each direction, each of shape (n_points,).
            curvatures : Normalized curvature of each direction, shown in the legend.
            center     : Loss at the trained point, setting the doubling reference line.
            path       : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.55))

        for index, (cut, curvature) in enumerate(zip(cuts, curvatures)):
            ax.plot(alphas, cut, marker=".", markersize=4, linewidth=1.2, color=self.OKABE_ITO[index % len(self.OKABE_ITO)], label=f"direction {index} (curv. {curvature:.2g})")

        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.axhline(CutDiagnostics.RADIUS_FACTOR * max(center, 1e-12), color="0.4", linestyle=":", linewidth=1.0, label=f"{CutDiagnostics.RADIUS_FACTOR:.0f}× centre loss")

        ax.set_xlabel("Step along the direction (filter-normalized)")
        ax.set_ylabel("Curve MSE")
        ax.set_yscale("log")
        ax.set_title("Loss cuts along random filter-normalized directions")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)

    def contour_2d(self, alphas: np.ndarray, betas: np.ndarray, losses: np.ndarray, path: Path, annotation: str | None = None) -> Path:
        """Draws the log-spaced filled contour of the 2-D landscape around the trained point.

        Args:
            alphas     : Steps along the first direction, shape (n_alpha,), drawn on the y axis.
            betas      : Steps along the second direction, shape (n_beta,), drawn on the x axis.
            losses     : Loss grid of shape (n_alpha, n_beta).
            path       : Output path of the figure.
            annotation : Optional text box drawn in the top-left corner.

        Returns:
            Path of the saved figure.

        Raises:
            ValueError : If the sampled landscape is flat, leaving no range to contour.
        """
        if not losses.max() > losses.min():
            raise ValueError(f"The sampled landscape is flat at {float(losses.min()):.6g} over the whole grid; a filled contour needs a loss range, so widen span or check that the loss depends on the perturbed weights")

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
        fig.colorbar(filled, ax=ax, fraction=0.045, pad=0.02).set_label("Curve MSE")
        fig.tight_layout()

        return self._save(fig, path)


class LossLandscapeRun(AnalysisRun):
    """Maps the loss landscape of a single training run."""

    SUMMARY_FILENAME = "loss_landscape.json"
    REPORT_FILENAME  = "loss_landscape.md"

    def _loss_fn(self, run):
        """Builds the objective evaluated at every landscape point.

        The batches and their ground-truth profiles are materialized once, so only the
        forward chan is repeated as the weights move. The objective is the physical curve
        MSE rather than the training loss, keeping runs with different loss configurations
        comparable.

        Args:
            run : Loaded run bundling the model, the dataloader and the normalizer.

        Returns:
            A zero-argument callable returning the mean curve MSE over the cached batches.

        Raises:
            ValueError : If the loader yields no batch.
        """
        renderer = PredictionCurves(run.n_gaussians, run.z_axis)
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

    def _write_report(self, run, center: float, diagnostics: list[dict], figures: dict[str, Path]) -> Path:
        """Writes the markdown report of the landscape.

        Args:
            run         : Loaded run, used for the backbone name in the title.
            center      : Curve MSE at the trained weights.
            diagnostics : Per-direction sharpness, curvature, flatness radius and minimum offset.
            figures     : Figure paths keyed by figure name.

        Returns:
            Path of the written report.
        """
        sharpness  = np.array([entry["sharpness"] for entry in diagnostics])
        curvature  = np.array([entry["curvature"] for entry in diagnostics])
        radii      = np.array([entry["flatness_radius"]["radius"] for entry in diagnostics])
        censored   = [entry["flatness_radius"]["censored"] for entry in diagnostics]

        doc = MarkdownDoc(title=f"Loss landscape: {run.backbone_name}")
        doc.paragraph(
            f"Curve-MSE landscape around the trained weights along {len(diagnostics)} filter-normalized random directions (Li et al. 2018 convention), "
            f"evaluated on {self.config.max_batches} '{self.config.split}' batches. Sharpness is the relative loss increase within ±{CutDiagnostics.SHARPNESS_RADIUS} of the centre; "
            f"curvature is the normalized quadratic coefficient fitted within ±{CutDiagnostics.CURVATURE_RADIUS}; the flatness radius is the interpolated step at which the loss "
            f"doubles (censored at the sampled span when it never does); the minimum offset flags cuts whose sampled minimum is not the trained point. "
            "The objective is the physical curve MSE, not the training loss, so runs with different loss configurations stay comparable."
        )

        radius_cells = [f"{r:.3f}{' (≥ span)' if c else ''}" for r, c in zip(radii, censored)]
        doc.kv_table({
            "Centre loss"      : f"{center:.4g}",
            "Sharpness"        : f"{sharpness.mean():.3f} ± {sharpness.std():.3f}",
            "Curvature"        : f"{curvature.mean():.3g} ± {curvature.std():.3g}",
            "Flatness radius"  : f"{radii.mean():.3f} ± {radii.std():.3f}",
            "Sharpest direction": int(np.argmax(sharpness)),
        })

        table = MarkdownTable(("Direction", "Sharpness", "Curvature", "Flatness radius", "Min offset"))
        for index, entry in enumerate(diagnostics):
            table.add_row(str(index), f"{entry['sharpness']:.3f}", f"{entry['curvature']:.3g}", radius_cells[index], f"{entry['min_offset']:+.3f}")
        doc.table(table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, path.name)

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        """Samples the 1-D cuts and the 2-D grid, diagnoses them and writes the report.

        Returns:
            Payload with the backbone name, the split, the sampled span, the centre loss,
            the 1-D steps and cuts, the 2-D steps and loss grid, and the per-direction
            diagnostics.

        Raises:
            ValueError : If fewer than two directions are configured, leaving no plane for the 2-D grid.
        """
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        if self.config.n_directions < 2:
            raise ValueError(f"The landscape needs at least 2 directions for the 2-D grid, got n_directions={self.config.n_directions}")

        run     = self._load_run()
        loss_fn = self._loss_fn(run)
        module  = run.model.module

        directions = [FilterNormalizedDirection.build(module, seed=self.config.direction_seed + index) for index in range(self.config.n_directions)]
        evaluator  = LandscapeEvaluator(module, loss_fn)

        alphas_1d = np.linspace(-self.config.span, self.config.span, self.config.n_points_1d)
        cuts      = [evaluator.grid(alphas_1d, np.zeros(1), direction, directions[0])[:, 0] for direction in directions]
        center    = float(loss_fn())

        diagnostics = [
            {
                "sharpness"       : CutDiagnostics.sharpness(alphas_1d, cut),
                "curvature"       : CutDiagnostics.curvature(alphas_1d, cut),
                "flatness_radius" : CutDiagnostics.flatness_radius(alphas_1d, cut),
                "min_offset"      : CutDiagnostics.min_offset(alphas_1d, cut),
            }
            for cut in cuts
        ]

        alphas_2d = np.linspace(-self.config.span, self.config.span, self.config.n_points_2d)
        grid      = evaluator.grid(alphas_2d, alphas_2d, directions[0], directions[1])

        sharpness = np.array([entry["sharpness"] for entry in diagnostics])
        radii     = np.array([entry["flatness_radius"]["radius"] for entry in diagnostics])

        plots   = LossLandscapePlots()
        figures = {
            "cuts"    : plots.cuts_overlay(alphas_1d, cuts, [entry["curvature"] for entry in diagnostics], center, self.output_dir / "cuts.png"),
            "contour" : plots.contour_2d(alphas_2d, alphas_2d, grid, self.output_dir / "contour.png", annotation=f"sharpness = {sharpness.mean():.2f} ± {sharpness.std():.2f}\nflatness radius = {radii.mean():.2f}"),
        }

        payload = {
            "backbone"    : run.backbone_name,
            "split"       : self.config.split,
            "span"        : self.config.span,
            "center_loss" : center,
            "alphas_1d"   : alphas_1d.tolist(),
            "cuts"        : [cut.tolist() for cut in cuts],
            "alphas_2d"   : alphas_2d.tolist(),
            "grid"        : grid.tolist(),
            "directions"  : diagnostics,
        }
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, center, diagnostics, figures)

        self.logger.ok(f"{self.run_dir.name}: sharpness {sharpness.mean():.3f} ± {sharpness.std():.3f} over {len(cuts)} directions -> {report_path}")

        return payload


class LossLandscapeBatch(RunBatch):
    """Maps the loss landscape of every run matching the configured filter."""

    SELECTOR_ACTION = "map"
    SECTION_TITLE   = "Loss landscape"
    RUN_CLASS       = LossLandscapeRun
