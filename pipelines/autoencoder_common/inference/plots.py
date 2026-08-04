from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from tools.reporting.plotting import PlotBase


class AePlotsBase(PlotBase):
    ERROR_XLABEL = r"$\log_{10}$ per-sample MSE"

    def __init__(self, fig_dpi: int = 150, save_dpi: int = 300) -> None:
        self.fig_dpi  = fig_dpi
        self.save_dpi = save_dpi
        self._apply_style()

    def _bins(self, values: np.ndarray, target: int = 60) -> int:
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
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        values = np.log10(np.maximum(mse, 1e-12))
        ax.hist(values, bins=self._bins(values), color="#34495e", edgecolor="white", linewidth=0.3)

        ax.set_xlabel(self.ERROR_XLABEL)
        ax.set_ylabel("count")
        ax.set_title("Reconstruction error distribution")

        return self._save(fig, figures_dir / "error_histogram.png")

    def _embedding_norm(self, embeddings: np.ndarray, figures_dir: Path) -> Path:
        norms = np.linalg.norm(embeddings, axis=1)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(norms, bins=self._bins(norms), color="#117a65", edgecolor="white", linewidth=0.3)

        ax.set_xlabel("embedding L2 norm")
        ax.set_ylabel("count")
        ax.set_title("Latent embedding norm distribution")

        return self._save(fig, figures_dir / "embedding_norm.png")
