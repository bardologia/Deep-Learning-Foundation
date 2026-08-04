from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from pipelines.autoencoder_common.inference.plots import AePlotsBase


class ImageAePlots(AePlotsBase):
    INTENSITY_CMAP = "viridis"
    ERROR_CMAP     = "magma"
    ERROR_XLABEL   = r"$\log_{10}$ per-patch MSE"

    def _patch_triplet(self, gt: np.ndarray, pred: np.ndarray, mse: float, label: str, rank: int, figures_dir: Path) -> List[Path]:
        stem = figures_dir / "reconstructions" / f"{label}_{rank + 1:02d}"

        vmin, vmax = self._shared_clim(gt, pred)
        error      = np.abs(pred - gt)
        cmap       = self._cmap_with_bad(self.INTENSITY_CMAP)
        err_cmap   = self._cmap_with_bad(self.ERROR_CMAP)

        gt_path   = self._imshow_figure(gt,    x_label="range (px)", y_label="azimuth (px)", title=f"{label} #{rank + 1}  ground truth (ch 0)", cmap=cmap,     vmin=vmin, vmax=vmax, colorbar_label="intensity",     path=Path(f"{stem}_gt.png"))
        pred_path = self._imshow_figure(pred,  x_label="range (px)", y_label="azimuth (px)", title=f"{label} #{rank + 1}  reconstruction (ch 0)", cmap=cmap,   vmin=vmin, vmax=vmax, colorbar_label="intensity",     path=Path(f"{stem}_recon.png"))
        err_path  = self._imshow_figure(error, x_label="range (px)", y_label="azimuth (px)", title=f"{label} #{rank + 1}  abs error  (MSE = {mse:.4g})", cmap=err_cmap, colorbar_label="absolute error", path=Path(f"{stem}_error.png"))

        return [gt_path, pred_path, err_path]

    def _reconstructions(self, gt, pred, mse, ranked, label, figures_dir) -> List[Path]:
        paths = []
        for rank, idx in enumerate(ranked):
            paths.extend(self._patch_triplet(gt[idx, 0], pred[idx, 0], float(mse[idx]), label, rank, figures_dir))

        return paths

    def _channel_mse(self, channel_mse: List[float], figures_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        channels = np.arange(len(channel_mse))
        ax.bar(channels, channel_mse, color="#1f4e79", edgecolor="white", linewidth=0.3)

        ax.set_xlabel("input channel")
        ax.set_ylabel("mean squared error")
        ax.set_title("Per-channel reconstruction error")

        return self._save(fig, figures_dir / "channel_mse.png")

    def _intensity_scatter(self, gt, pred, n_points, seed, figures_dir) -> Path:
        mean_gt   = gt.mean(axis=(1, 2, 3))
        mean_pred = pred.mean(axis=(1, 2, 3))

        idx = self._subsample(np.arange(mean_gt.shape[0]), n_points, seed=seed)
        pg  = mean_gt[idx]
        pp  = mean_pred[idx]

        if pg.size == 0:
            raise ValueError("The intensity scatter needs at least one reconstructed patch; the split produced none")

        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.scatter(pg, pp, s=3, alpha=0.25, color="#1f4e79", edgecolors="none")

        lo = float(min(pg.min(), pp.min()))
        hi = float(max(pg.max(), pp.max()))
        ax.plot([lo, hi], [lo, hi], color="#c0392b", linewidth=1.0, linestyle="--")

        ax.set_xlabel("mean patch intensity (ground truth)")
        ax.set_ylabel("mean patch intensity (reconstruction)")
        ax.set_title("Mean patch intensity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        return self._save(fig, figures_dir / "intensity_scatter.png")

    def compose(self, result, channel_mse: List[float], mse: np.ndarray, cfg, figures_dir: Path) -> Dict[str, List[Path]]:
        gt   = result.gt
        pred = result.pred

        order = np.argsort(mse)
        rng   = np.random.default_rng(cfg.patch_seed)

        best   = order[: cfg.n_best_patches]
        worst  = order[::-1][: cfg.n_worst_patches]
        random = rng.choice(mse.shape[0], size=min(cfg.n_random_patches, mse.shape[0]), replace=False) if mse.shape[0] else np.empty(0, dtype=np.int64)

        figures = {
            "best"              : self._reconstructions(gt, pred, mse, best,   "best",   figures_dir),
            "worst"             : self._reconstructions(gt, pred, mse, worst,  "worst",  figures_dir),
            "random"            : self._reconstructions(gt, pred, mse, random, "random", figures_dir),
            "error_histogram"   : [self._error_histogram(mse, figures_dir)],
            "channel_mse"       : [self._channel_mse(channel_mse, figures_dir)],
            "intensity_scatter" : [self._intensity_scatter(gt, pred, cfg.n_scatter_points, cfg.patch_seed, figures_dir)],
            "embedding_norm"    : [self._embedding_norm(result.embeddings, figures_dir)],
        }

        return figures
