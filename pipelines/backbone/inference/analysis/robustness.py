from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.input_attribution import TrackChannels
from pipelines.backbone.inference.analysis.run_batch         import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes                     import PredictionCurves
from tools.data.io                                           import FileIO
from tools.reporting.markdown                                import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                                import PlotBase


class RobustnessCore:

    def __init__(self, model, renderer: PredictionCurves, seed: int = 0) -> None:
        self.model    = model
        self.renderer = renderer
        self.rng      = np.random.default_rng(seed)

    def _mse(self, batches: list[tuple[torch.Tensor, np.ndarray]], transform) -> float:
        total = 0.0
        for images, gt_curves in batches:
            perturbed = transform(images)
            with torch.no_grad():
                curves = self.renderer.render(self.model(perturbed))
            total += float(((curves - gt_curves) ** 2).mean())

        return total / len(batches)

    def noise_curve(self, batches: list, sigmas: list[float]) -> list[dict]:
        rows = []
        for sigma in sigmas:
            def transform(images, sigma=sigma):
                if sigma == 0.0:
                    return images
                jitter = torch.from_numpy(self.rng.normal(0.0, sigma, size=tuple(images.shape)).astype(np.float32))
                return images + jitter

            rows.append({"sigma": float(sigma), "mse": self._mse(batches, transform)})

        return rows

    def drop_curve(self, batches: list, per_track: list[list[int]], draws: int) -> list[dict]:
        n_tracks = len(per_track)
        rows     = []

        for n_dropped in range(n_tracks + 1):
            if n_dropped == 0:
                rows.append({"dropped": 0, "mse": self._mse(batches, lambda images: images)})
                continue

            n_draws = min(draws, math.comb(n_tracks, n_dropped))
            values  = []

            for _draw in range(n_draws):
                chosen   = self.rng.choice(n_tracks, size=n_dropped, replace=False)
                channels = [index for track in chosen for index in per_track[track]]

                def transform(images, channels=tuple(channels)):
                    occluded = images.clone()
                    occluded[:, list(channels)] = 0.0
                    return occluded

                values.append(self._mse(batches, transform))

            rows.append({"dropped": n_dropped, "mse": float(np.mean(values)), "mse_std": float(np.std(values)) if n_draws > 1 else None})

        return rows


class RobustnessPlots(PlotBase):

    def severity_curve(self, xs: list[float], ys: list[float], x_label: str, title: str, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(xs, ys, marker="o", color="#D55E00", linewidth=1.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Curve MSE")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class RobustnessRun(AnalysisRun):

    SUMMARY_FILENAME = "robustness.json"
    REPORT_FILENAME  = "robustness.md"

    def _batches(self, run, renderer: PredictionCurves) -> list[tuple[torch.Tensor, np.ndarray]]:
        batches = []
        for index, batch in enumerate(run.loader):
            if index >= self.config.max_batches:
                break
            gt_phys = run.dataset.normalizer.denormalize_output(batch[1].float()).numpy()
            batches.append((batch[0], renderer.render(gt_phys)))

        if not batches:
            raise ValueError("The loader yielded no batches for the robustness sweep")

        return batches

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run      = self._load_run()
        renderer = PredictionCurves(run.n_gaussians, run.x_axis, self.config.render_amp_floor)
        batches  = self._batches(run, renderer)
        core     = RobustnessCore(run.model, renderer, seed=self.config.seed)

        noise_rows = core.noise_curve(batches, list(self.config.noise_sigmas))
        drop_rows  = core.drop_curve(batches, TrackChannels.build(run), self.config.draws_per_count)

        plots   = RobustnessPlots()
        figures = {
            "noise" : plots.severity_curve([row["sigma"] for row in noise_rows], [row["mse"] for row in noise_rows], "Added input noise sigma (normalized units)", "Degradation under input noise", self.output_dir / "noise.png"),
            "drop"  : plots.severity_curve([row["dropped"] for row in drop_rows], [row["mse"] for row in drop_rows], "Tracks dropped (secondary + interferogram zeroed)", "Degradation under track dropout", self.output_dir / "drop.png"),
        }

        payload = {"backbone": run.backbone_name, "split": self.config.split, "noise": noise_rows, "drop": drop_rows}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        doc = MarkdownDoc(title=f"Robustness: {run.backbone_name}")
        doc.paragraph(
            f"Curve-MSE degradation on {self.config.max_batches} '{self.config.split}' batches under two controlled stresses: "
            "gaussian noise added to the normalized inputs, and whole tracks zeroed (their secondary and interferogram channels), "
            f"averaged over up to {self.config.draws_per_count} random track subsets per count, capped at the number of distinct subsets that exist; counts with a single possible subset report no spread."
        )

        noise_table = MarkdownTable(("Noise sigma", "Curve MSE"))
        for row in noise_rows:
            noise_table.add_row(f"{row['sigma']:.2f}", f"{row['mse']:.4g}")
        doc.table(noise_table)

        drop_table = MarkdownTable(("Tracks dropped", "Curve MSE", "± across draws"))
        for row in drop_rows:
            spread = row.get("mse_std")
            drop_table.add_row(str(row["dropped"]), f"{row['mse']:.4g}", f"{spread:.2g}" if spread is not None else "n/a")
        doc.table(drop_table)

        for name, path in figures.items():
            doc.image(name, path.name)

        report_path = doc.save(self.output_dir / self.REPORT_FILENAME)

        clean = noise_rows[0]["mse"]
        worst = noise_rows[-1]["mse"]
        self.logger.ok(f"{self.run_dir.name}: noise degrades MSE {clean:.3g} -> {worst:.3g} at sigma {noise_rows[-1]['sigma']} -> {report_path}")

        return payload


class RobustnessBatch(RunBatch):

    SELECTOR_ACTION = "stress"
    SECTION_TITLE   = "Robustness sweep"
    RUN_CLASS       = RobustnessRun
