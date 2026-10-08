"""Publication figures for profile-autoencoder inference.

Draws reconstruction overlays, error and power diagnostics, and latent RGB maps
that place the sampled embeddings back onto the row-col grid of each split
region.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from pipelines.autoencoder_common.inference.plots import AePlotsBase
from tools.data.regions                           import CropRegion


class ProfileAePlots(AePlotsBase):
    """Renders the figure set for a profile-autoencoder inference run."""

    ERROR_XLABEL = r"$\log_{10}$ per-curve MSE"

    def _overlay(self, z_axis: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, path: Path) -> Path:
        """Plots one ground-truth curve against its reconstruction.

        Args:
            z_axis : Z axis of shape (H,), in metres.
            gt     : Ground-truth profile of shape (H,).
            pred   : Reconstructed profile of shape (H,).
            title  : Axis title.
            path   : Destination file.

        Returns:
            The written figure path.
        """
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        ax.plot(z_axis, gt,   color="#1f4e79", linewidth=1.6, label="ground truth")
        ax.plot(z_axis, pred, color="#c0392b", linewidth=1.6, linestyle="--", label="reconstruction")

        ax.set_xlabel("z (m)")
        ax.set_ylabel("intensity")
        ax.set_title(title)
        ax.legend(frameon=False)

        return self._save(fig, path)

    def _reconstructions(self, z_axis, gt, pred, mse, ranked, label, figures_dir) -> List[Path]:
        """Plots one overlay per selected curve.

        Args:
            z_axis      : Z axis of shape (H,), in metres.
            gt          : Ground-truth curves of shape (n_curves, H).
            pred        : Reconstructed curves of shape (n_curves, H).
            mse         : Per-curve MSE of shape (n_curves,).
            ranked      : Row indices to plot, in the order they are numbered.
            label       : Group name used in titles and filenames.
            figures_dir : Root figures directory.

        Returns:
            The written figure paths.
        """
        paths = []
        for rank, idx in enumerate(ranked):
            title = f"{label} reconstruction #{rank + 1}  (MSE = {mse[idx]:.4g})"
            path  = figures_dir / "reconstructions" / f"{label}_{rank + 1:02d}.png"
            paths.append(self._overlay(z_axis, gt[idx], pred[idx], title, path))

        return paths

    def _mean_profile(self, z_axis, gt, pred, figures_dir) -> Path:
        """Plots the split-averaged ground-truth and reconstructed profile."""
        return self._overlay(z_axis, gt.mean(axis=0), pred.mean(axis=0), "Mean profile over split", figures_dir / "mean_profile.png")

    def _power_scatter(self, z_axis, gt, pred, active, n_points, seed, figures_dir) -> Path:
        """Scatters reconstructed against true integrated profile power.

        Args:
            z_axis      : Z axis of shape (H,), in metres.
            gt          : Ground-truth curves of shape (n_curves, H).
            pred        : Reconstructed curves of shape (n_curves, H).
            active      : Row indices of the active curves.
            n_points    : Maximum number of scatter points drawn.
            seed        : Seed of the subsampling generator.
            figures_dir : Root figures directory.

        Returns:
            The written figure path.
        """
        power_gt   = np.trapezoid(gt[active],   z_axis, axis=1)
        power_pred = np.trapezoid(pred[active], z_axis, axis=1)

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

    def _region_bounds(self, regions: List[CropRegion]) -> np.ndarray:
        """Returns the cumulative flat-pixel offsets delimiting each region.

        Args:
            regions : Split crop regions in the order the dataset stacked them.

        Returns:
            Offsets of shape (len(regions) + 1,) starting at zero.
        """
        sizes = np.array([region.row_size * region.col_size for region in regions], dtype=np.int64)

        return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes)])

    def _region_pixels(self, pixel_index: np.ndarray, region: CropRegion, start: int, stop: int):
        """Maps the flat pixel indices falling in one region to grid positions.

        Args:
            pixel_index : Flat pixel indices of shape (n_curves,).
            region      : The crop region being mapped.
            start       : First flat index belonging to the region.
            stop        : Exclusive last flat index belonging to the region.

        Returns:
            Tuple of the matching rows into `pixel_index` and their row and
            col positions within the region.
        """
        rows  = np.flatnonzero((pixel_index >= start) & (pixel_index < stop))
        local = pixel_index[rows] - start

        return rows, local // region.col_size, local % region.col_size

    def _latent_rgb_map(self, region: CropRegion, rgb: np.ndarray, rows: np.ndarray, row_pos: np.ndarray, col_pos: np.ndarray, components: np.ndarray, figures_dir: Path) -> Path:
        """Paints three latent components as RGB over one region's grid.

        Args:
            region      : The crop region being drawn.
            rgb         : Per-curve RGB triplets of shape (n_curves, 3) in [0, 1].
            rows        : Rows of `rgb` belonging to this region.
            row_pos     : Row positions of those rows within the region.
            col_pos     : Column positions of those rows within the region.
            components  : The three latent component indices being shown.
            figures_dir : Root figures directory.

        Returns:
            The written figure path.
        """
        canvas                  = np.zeros((region.row_size, region.col_size, 3), dtype=np.float64)
        canvas[row_pos, col_pos] = rgb[rows]
        coverage                = rows.shape[0] / float(region.row_size * region.col_size)

        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.imshow(canvas, aspect="auto", interpolation="nearest", extent=[region.col_start, region.col_end, region.row_end, region.row_start])

        ax.tick_params(which="both", color=self.RGB_FOREGROUND)
        for spine in ax.spines.values():
            spine.set_color(self.RGB_FOREGROUND)

        ax.set_xlabel("col index")
        ax.set_ylabel("row index")
        ax.set_title(f"Latent components {components[0]}, {components[1]}, {components[2]} as (R, G, B),  {coverage:.0%} of pixels sampled")

        return self._save(fig, figures_dir / "latent_rgb_maps" / f"{region.as_labeled_string()}.png")

    def _latent_rgb_maps(self, embeddings: np.ndarray, pixel_index: np.ndarray, regions: List[CropRegion], figures_dir: Path) -> List[Path]:
        """Draws one latent RGB map per split region.

        Args:
            embeddings  : Latent vectors of shape (n_curves, embedding_dim).
            pixel_index : Flat pixel index of shape (n_curves,).
            regions     : Split crop regions in dataset stacking order.
            figures_dir : Root figures directory.

        Returns:
            One figure path per region.

        Raises:
            ValueError : If the index length disagrees with the embedding count, if an index falls outside the regions, or if the regions do not account for every index.
        """
        values      = np.asarray(embeddings, dtype=np.float64)
        pixel_index = np.asarray(pixel_index, dtype=np.int64)

        if pixel_index.shape[0] != values.shape[0]:
            raise ValueError(f"the latent RGB map needs one pixel index per embedding, got {pixel_index.shape[0]} indices for {values.shape[0]} embeddings")

        components, _ = self._top_variance_components(values)
        rgb           = self._rgb_channels(values[:, components])
        bounds        = self._region_bounds(regions)

        if pixel_index.min() < 0 or pixel_index.max() >= bounds[-1]:
            raise ValueError(f"pixel indices span [{pixel_index.min()}, {pixel_index.max()}] but the split regions hold only {bounds[-1]} pixels; the index no longer matches the split geometry")

        paths   = []
        mapped  = 0

        for position, region in enumerate(regions):
            rows, row_pos, col_pos = self._region_pixels(pixel_index, region, int(bounds[position]), int(bounds[position + 1]))
            mapped               += rows.shape[0]

            paths.append(self._latent_rgb_map(region, rgb, rows, row_pos, col_pos, components, figures_dir))

        if mapped != pixel_index.shape[0]:
            raise ValueError(f"the latent RGB map placed {mapped} of {pixel_index.shape[0]} pixels; the region order no longer matches the order the dataset stacked them in")

        return paths

    def compose(self, result, z_axis: np.ndarray, mse: np.ndarray, cfg, figures_dir: Path, activity_threshold: float, pixel_index: np.ndarray, regions: List[CropRegion]) -> Dict[str, List[Path]]:
        """Renders every figure group for one inference run.

        Args:
            result             : Reconstruction result holding curves of shape (n_curves, H) and embeddings of shape (n_curves, embedding_dim).
            z_axis             : Z axis of shape (H,), in metres.
            mse                : Per-curve MSE of shape (n_curves,).
            cfg                : Inference configuration giving curve counts and seeds.
            figures_dir        : Root figures directory.
            activity_threshold : Amplitude above which a curve counts as active, as recorded by the parameter run.
            pixel_index        : Flat pixel index of shape (n_curves,).
            regions            : Split crop regions in dataset stacking order.

        Returns:
            Mapping from figure-group name to the written paths.
        """
        gt   = result.gt
        pred = result.pred

        active     = np.flatnonzero(gt.max(axis=1) > activity_threshold)
        order      = active[np.argsort(mse[active])]
        rng        = np.random.default_rng(cfg.curve_seed)

        best   = order[: cfg.n_best_curves]
        worst  = order[::-1][: cfg.n_worst_curves]
        random = rng.choice(active, size=min(cfg.n_random_curves, active.shape[0]), replace=False) if active.size else np.empty(0, dtype=np.int64)

        figures = {
            "best"               : self._reconstructions(z_axis, gt, pred, mse, best,   "best",   figures_dir),
            "worst"              : self._reconstructions(z_axis, gt, pred, mse, worst,  "worst",  figures_dir),
            "random"             : self._reconstructions(z_axis, gt, pred, mse, random, "random", figures_dir),
            "mean_profile"       : [self._mean_profile(z_axis, gt, pred, figures_dir)],
            "error_histogram"    : [self._error_histogram(mse, figures_dir)],
            "power_scatter"      : [self._power_scatter(z_axis, gt, pred, active, cfg.n_scatter_points, cfg.curve_seed, figures_dir)],
            "embedding_norm"     : [self._embedding_norm(result.embeddings, figures_dir)],
            "latent_rgb_scatter" : [self._latent_rgb_scatter(result.embeddings, cfg.n_scatter_points, cfg.curve_seed, figures_dir)],
            "latent_rgb_map"     : self._latent_rgb_maps(result.embeddings, pixel_index, regions, figures_dir),
        }

        return figures
