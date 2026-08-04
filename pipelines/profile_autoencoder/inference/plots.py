from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from pipelines.autoencoder_common.inference.plots import AePlotsBase


class ProfileAePlots(AePlotsBase):
    ERROR_XLABEL = r"$\log_{10}$ per-curve MSE"

    def _overlay(self, x_axis: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        ax.plot(x_axis, gt,   color="#1f4e79", linewidth=1.6, label="ground truth")
        ax.plot(x_axis, pred, color="#c0392b", linewidth=1.6, linestyle="--", label="reconstruction")

        ax.set_xlabel("elevation (m)")
        ax.set_ylabel("intensity")
        ax.set_title(title)
        ax.legend(frameon=False)

        return self._save(fig, path)

    def _reconstructions(self, x_axis, gt, pred, mse, ranked, label, figures_dir) -> List[Path]:
        paths = []
        for rank, idx in enumerate(ranked):
            title = f"{label} reconstruction #{rank + 1}  (MSE = {mse[idx]:.4g})"
            path  = figures_dir / "reconstructions" / f"{label}_{rank + 1:02d}.png"
            paths.append(self._overlay(x_axis, gt[idx], pred[idx], title, path))

        return paths

    def _mean_profile(self, x_axis, gt, pred, figures_dir) -> Path:
        return self._overlay(x_axis, gt.mean(axis=0), pred.mean(axis=0), "Mean profile over split", figures_dir / "mean_profile.png")

    def _power_scatter(self, x_axis, gt, pred, active, n_points, seed, figures_dir) -> Path:
        power_gt   = np.trapezoid(gt[active],   x_axis, axis=1)
        power_pred = np.trapezoid(pred[active], x_axis, axis=1)

        idx = self._subsample(np.arange(power_gt.shape[0]), n_points, seed=seed)
        pg  = power_gt[idx]
        pp  = power_pred[idx]

        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.scatter(pg, pp, s=3, alpha=0.25, color="#1f4e79", edgecolors="none")

        lim = float(max(pg.max(initial=0.0), pp.max(initial=0.0)))
        ax.plot([0.0, lim], [0.0, lim], color="#c0392b", linewidth=1.0, linestyle="--")

        ax.set_xlabel("integrated power (ground truth)")
        ax.set_ylabel("integrated power (reconstruction)")
        ax.set_title("Integrated profile power")
        ax.set_xlim(0.0, lim)
        ax.set_ylim(0.0, lim)

        return self._save(fig, figures_dir / "power_scatter.png")

    def compose(self, result, x_axis: np.ndarray, mse: np.ndarray, cfg, figures_dir: Path, amp_zero_thr: float) -> Dict[str, List[Path]]:
        gt   = result.gt
        pred = result.pred

        active     = np.flatnonzero(gt.max(axis=1) > amp_zero_thr)
        order      = active[np.argsort(mse[active])]
        rng        = np.random.default_rng(cfg.curve_seed)

        best   = order[: cfg.n_best_curves]
        worst  = order[::-1][: cfg.n_worst_curves]
        random = rng.choice(active, size=min(cfg.n_random_curves, active.shape[0]), replace=False) if active.size else np.empty(0, dtype=np.int64)

        figures = {
            "best"            : self._reconstructions(x_axis, gt, pred, mse, best,   "best",   figures_dir),
            "worst"           : self._reconstructions(x_axis, gt, pred, mse, worst,  "worst",  figures_dir),
            "random"          : self._reconstructions(x_axis, gt, pred, mse, random, "random", figures_dir),
            "mean_profile"    : [self._mean_profile(x_axis, gt, pred, figures_dir)],
            "error_histogram" : [self._error_histogram(mse, figures_dir)],
            "power_scatter"   : [self._power_scatter(x_axis, gt, pred, active, cfg.n_scatter_points, cfg.curve_seed, figures_dir)],
            "embedding_norm"  : [self._embedding_norm(result.embeddings, figures_dir)],
        }

        return figures
