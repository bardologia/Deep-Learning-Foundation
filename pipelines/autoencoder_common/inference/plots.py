"""Reconstruction-error and latent-space figures shared by the autoencoders.

Provides the error histogram, embedding-norm histogram and the latent RGB
scatter in which the three highest-variance latent components colour the points.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from tools.reporting.plotting import PlotBase


class AePlotsBase(PlotBase):
    """Figure builders common to profile and image autoencoder inference.

    Attributes:
        fig_dpi  : Screen resolution used while composing figures.
        save_dpi : Resolution used when writing figures to disk.
    """

    ERROR_XLABEL    = r"$\log_{10}$ per-sample MSE"
    RGB_COMPONENTS  = 3
    RGB_PERCENTILES = (1.0, 99.0)
    RGB_SPAN_FLOOR  = 1e-12
    RGB_BRIGHTNESS  = 0.15
    RGB_BACKGROUND  = "#111111"
    RGB_FOREGROUND  = "#bbbbbb"

    def __init__(self, fig_dpi: int = 150, save_dpi: int = 300) -> None:
        """Stores the figure resolutions and applies the publication style.

        Args:
            fig_dpi  : Resolution used while composing figures.
            save_dpi : Resolution used when writing figures to disk.
        """
        self.fig_dpi  = fig_dpi
        self.save_dpi = save_dpi
        self._apply_style()

    def _bins(self, values: np.ndarray, target: int = 60) -> int:
        """Returns a histogram bin count, collapsing to one when the values barely spread."""
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size < 2:
            return 1

        lo, hi = float(finite.min()), float(finite.max())
        spread = hi - lo
        floor  = np.spacing(max(abs(lo), abs(hi), 1.0)) * target
        if spread <= floor:
            return 1

        return min(target, finite.size)

    def _error_histogram(self, mse: np.ndarray, figures_dir: Path) -> Path:
        """Plots the distribution of log10 per-sample reconstruction MSE.

        Args:
            mse         : Per-sample mean squared error of shape (n_samples,).
            figures_dir : Directory the figure is written into.

        Returns:
            Path of the saved figure.
        """
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        values = np.log10(np.maximum(mse, 1e-12))
        ax.hist(values, bins=self._bins(values), color="#34495e", edgecolor="white", linewidth=0.3)

        ax.set_xlabel(self.ERROR_XLABEL)
        ax.set_ylabel("count")
        ax.set_title("Reconstruction error distribution")

        return self._save(fig, figures_dir / "error_histogram.png")

    def _embedding_norm(self, embeddings: np.ndarray, figures_dir: Path) -> Path:
        """Plots the distribution of latent L2 norms.

        Args:
            embeddings  : Latent matrix of shape (n_samples, embedding_dim).
            figures_dir : Directory the figure is written into.

        Returns:
            Path of the saved figure.
        """
        norms = np.linalg.norm(embeddings, axis=1)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(norms, bins=self._bins(norms), color="#117a65", edgecolor="white", linewidth=0.3)

        ax.set_xlabel("embedding L2 norm")
        ax.set_ylabel("count")
        ax.set_title("Latent embedding norm distribution")

        return self._save(fig, figures_dir / "embedding_norm.png")

    def _top_variance_components(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the three highest-variance latent components and their variances.

        Args:
            embeddings : Latent matrix of shape (n_samples, embedding_dim).

        Returns:
            Tuple of the component indices, ordered by descending variance, and
            their variances, each of shape (3,).

        Raises:
            ValueError : If the embeddings are not a two-dimensional matrix or carry fewer than three components.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"the latent RGB scatter expects an (N, d) embedding matrix, got shape {embeddings.shape}")

        if embeddings.shape[1] < self.RGB_COMPONENTS:
            raise ValueError(f"the latent RGB scatter needs at least {self.RGB_COMPONENTS} embedding components, the run provides {embeddings.shape[1]}")

        variances = embeddings.var(axis=0)
        ranked    = np.argsort(variances)[::-1][: self.RGB_COMPONENTS]

        return ranked, variances[ranked]

    def _rgb_channels(self, columns: np.ndarray) -> np.ndarray:
        """Maps latent columns of shape (n_samples, 3) to percentile-stretched RGB values in [brightness, 1]."""
        low, high = np.percentile(columns, self.RGB_PERCENTILES, axis=0)
        span      = np.maximum(high - low, self.RGB_SPAN_FLOOR)
        unit      = np.clip((columns - low) / span, 0.0, 1.0)

        return self.RGB_BRIGHTNESS + (1.0 - self.RGB_BRIGHTNESS) * unit

    def _latent_rgb_scatter(self, embeddings: np.ndarray, n_points: int, seed: int, figures_dir: Path) -> Path:
        """Scatters the two leading latent components, coloured by the three leading ones.

        Args:
            embeddings  : Latent matrix of shape (n_samples, embedding_dim).
            n_points    : Maximum number of points drawn after subsampling.
            seed        : Seed of the subsampling draw.
            figures_dir : Directory the figure is written into.

        Returns:
            Path of the saved figure.
        """
        values            = np.asarray(embeddings, dtype=np.float64)
        ranked, variances = self._top_variance_components(values)
        rgb               = self._rgb_channels(values[:, ranked])

        columns                = [values[:, ranked[0]], values[:, ranked[1]], rgb[:, 0], rgb[:, 1], rgb[:, 2]]
        x, y, red, green, blue = self._paired_subsample(columns, n_points, seed=seed)

        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.set_facecolor(self.RGB_BACKGROUND)
        ax.scatter(x, y, s=5, alpha=0.7, c=np.stack([red, green, blue], axis=1), edgecolors="none")

        ax.tick_params(which="both", color=self.RGB_FOREGROUND)
        for spine in ax.spines.values():
            spine.set_color(self.RGB_FOREGROUND)

        ax.set_xlabel(f"latent component {ranked[0]}  (variance {variances[0]:.3g})")
        ax.set_ylabel(f"latent component {ranked[1]}  (variance {variances[1]:.3g})")
        ax.set_title(f"Latent components {ranked[0]}, {ranked[1]}, {ranked[2]} as (R, G, B)")

        return self._save(fig, figures_dir / "latent_rgb_scatter.png")
