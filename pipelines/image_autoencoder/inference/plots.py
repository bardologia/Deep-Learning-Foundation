"""Figures summarising image-autoencoder reconstruction quality on a split.

Renders per-patch ground truth, reconstruction and absolute-error maps for the
best, worst and randomly drawn patches, plus the per-channel error bars, the
mean-intensity scatter and the latent-norm figure.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from pipelines.autoencoder_common.inference.plots import AePlotsBase


class ImageAePlots(AePlotsBase):
    """Publication-quality figures for image-autoencoder inference."""

    INTENSITY_CMAP = "viridis"
    ERROR_CMAP     = "magma"
    ERROR_XLABEL   = r"$\log_{10}$ per-patch MSE"

    def _patch_triplet(self, gt: np.ndarray, pred: np.ndarray, mse: float, label: str, rank: int, figures_dir: Path) -> List[Path]:
        """Saves the ground-truth, reconstruction and absolute-error maps of one patch.

        Args:
            gt          : Ground-truth channel of shape (rows, cols), in intensity units.
            pred        : Reconstruction of the same shape and units.
            mse         : Mean squared error of the whole patch, shown in the error title.
            label       : Selection the patch comes from, `best`, `worst` or `random`.
            rank        : Zero-based position of the patch inside that selection.
            figures_dir : Root directory the `reconstructions` subdirectory is written under.

        Returns:
            The three saved figure paths, ground truth first.
        """
        stem = figures_dir / "reconstructions" / f"{label}_{rank + 1:02d}"

        vmin, vmax = self._shared_clim(gt, pred)
        error      = np.abs(pred - gt)
        cmap       = self._cmap_with_bad(self.INTENSITY_CMAP)
        err_cmap   = self._cmap_with_bad(self.ERROR_CMAP)

        gt_path   = self._imshow_figure(gt,    x_label="col (px)", y_label="row (px)", title=f"{label} #{rank + 1}  ground truth (ch 0)", cmap=cmap,     vmin=vmin, vmax=vmax, colorbar_label="intensity",     path=Path(f"{stem}_gt.png"))
        pred_path = self._imshow_figure(pred,  x_label="col (px)", y_label="row (px)", title=f"{label} #{rank + 1}  reconstruction (ch 0)", cmap=cmap,   vmin=vmin, vmax=vmax, colorbar_label="intensity",     path=Path(f"{stem}_recon.png"))
        err_path  = self._imshow_figure(error, x_label="col (px)", y_label="row (px)", title=f"{label} #{rank + 1}  abs error  (MSE = {mse:.4g})", cmap=err_cmap, colorbar_label="absolute error", path=Path(f"{stem}_error.png"))

        return [gt_path, pred_path, err_path]

    def _reconstructions(self, gt, pred, mse, ranked, label, figures_dir) -> List[Path]:
        """Saves the first-channel triplet of every selected patch.

        Args:
            gt          : Ground-truth patches of shape (n_patches, channels, rows, cols).
            pred        : Reconstructions of the same shape.
            mse         : Per-patch mean squared error of shape (n_patches,).
            ranked      : Patch indices to render, in the order they are ranked.
            label       : Selection name used in filenames and titles.
            figures_dir : Root directory of the figure tree.

        Returns:
            All saved figure paths, three per selected patch.
        """
        paths = []
        for rank, idx in enumerate(ranked):
            paths.extend(self._patch_triplet(gt[idx, 0], pred[idx, 0], float(mse[idx]), label, rank, figures_dir))

        return paths

    def _channel_mse(self, channel_mse: List[float], figures_dir: Path) -> Path:
        """Saves the per-channel reconstruction error as a bar chart.

        Args:
            channel_mse : Mean squared error of each input channel, in stack order.
            figures_dir : Directory the figure is written to.

        Returns:
            Path of the saved figure.
        """
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        channels = np.arange(len(channel_mse))
        ax.bar(channels, channel_mse, color="#1f4e79", edgecolor="white", linewidth=0.3)

        ax.set_xlabel("input channel")
        ax.set_ylabel("mean squared error")
        ax.set_title("Per-channel reconstruction error")

        return self._save(fig, figures_dir / "channel_mse.png")

    def _intensity_scatter(self, gt, pred, n_points, seed, figures_dir) -> Path:
        """Saves reconstructed against true mean patch intensity, with the identity line.

        Args:
            gt          : Ground-truth patches of shape (n_patches, channels, rows, cols).
            pred        : Reconstructions of the same shape.
            n_points    : Maximum number of patches drawn into the scatter.
            seed        : Seed of the subsampling generator.
            figures_dir : Directory the figure is written to.

        Returns:
            Path of the saved figure.

        Raises:
            ValueError : If the split produced no patches to plot.
        """
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
        """Renders every inference figure and returns their paths grouped by kind.

        Args:
            result      : Predictor result holding ground truth, reconstructions and embeddings.
            channel_mse : Mean squared error of each input channel, in stack order.
            mse         : Per-patch mean squared error of shape (n_patches,), used to rank the best and worst patches.
            cfg         : Inference configuration giving the patch counts, the scatter point budget and the patch selection seed.
            figures_dir : Root directory of the figure tree.

        Returns:
            Figure paths keyed by figure kind.
        """
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
