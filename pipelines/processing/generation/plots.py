from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from tools.data.io            import FileIO
from tools.reporting.plotting import PlotBase
from tools.monitoring.logger  import Logger
from tools.sar.coherence      import CoherenceEstimator


class StackPlotter(PlotBase):
    def __init__(self, run_directory: Path, max_amplitude_clip: float, logger: Logger, fig_dpi: int = 150, save_dpi: int = 300, coherence_window: tuple = (7, 7)) -> None:
        self.max_amplitude_clip  = max_amplitude_clip
        self.logger              = logger
        self.fig_dpi             = fig_dpi
        self.save_dpi            = save_dpi
        self.images_directory    = Path(run_directory) / "images"
        self.coherence_estimator = CoherenceEstimator(coherence_window)

    def _setup_output_dirs(self) -> Dict[str, Path]:
        dirs = {
            "slc"            : self.images_directory / "slc",
            "interferograms" : self.images_directory / "interferograms",
            "coherence"      : self.images_directory / "coherence",
            "dem"            : self.images_directory / "dem",
        }
        FileIO.ensure_dirs(*dirs.values())
        return dirs

    @staticmethod
    def _amplitude_db(data: np.ndarray) -> np.ndarray:
        amplitude = np.abs(data).astype(np.float32)
        return 20.0 * np.log10(np.maximum(amplitude, 1e-12))

    def _plot_amplitude(self, amplitude_db: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R      = amplitude_db.shape
        vmin, vmax = self._shared_clim(amplitude_db)

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(amplitude_db, cmap="gray", vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("amplitude [dB]")
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_linear_amplitude(self, amplitude: np.ndarray, title: str, cbar_label: str, out_path: Path) -> Path:
        Az, R      = amplitude.shape
        vmin, vmax = self._amplitude_clim(amplitude)

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(amplitude, cmap="gray", vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(cbar_label)
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_phase(self, phase: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R = phase.shape

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(phase, cmap=self._phase_cmap(), vmin=-np.pi, vmax=np.pi, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=self.PHASE_TICKS)
        cb.set_label("interferometric phase [rad]")
        cb.ax.set_yticklabels(self.PHASE_LABELS)
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_interferogram(self, interferogram: np.ndarray, title: str, out_dir: Path, stem: str) -> Dict[str, Path]:
        clip      = float(self.max_amplitude_clip)
        amplitude = np.abs(interferogram).astype(np.float32)
        phase     = np.angle(interferogram).astype(np.float32)

        return {
            "amplitude" : self._plot_linear_amplitude(amplitude, f"{title} — secondary SLC amplitude (clipped at {clip:g})", f"secondary SLC amplitude (clipped at {clip:g})", out_dir / f"{stem}_amplitude.png"),
            "phase"     : self._plot_phase(phase,                f"{title} — flattened phase",                              out_dir / f"{stem}_phase.png"),
        }

    def _plot_coherence(self, magnitude: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R = magnitude.shape

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(magnitude, cmap="gray", vmin=0.0, vmax=1.0, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("coherence")
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_coherence_pair(self, primary_amplitude: np.ndarray, interferogram: np.ndarray, title: str, out_dir: Path, stem: str) -> Dict[str, Path]:
        magnitude, phase = self.coherence_estimator.estimate_flattened(primary_amplitude, interferogram)

        return {
            "magnitude" : self._plot_coherence(magnitude, f"Coherence — {title}",       out_dir / f"{stem}_magnitude.png"),
            "phase"     : self._plot_phase(phase,         f"Coherence phase — {title}", out_dir / f"{stem}_phase.png"),
        }

    def _plot_dem(self, dem: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R      = dem.shape
        vmin, vmax = self._shared_clim(dem)
        cmap_obj   = self._cmap_with_bad("terrain")

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(dem, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("height [m]")
        fig.tight_layout()

        return self._save(fig, out_path)

    def run(
        self,
        primary_path        : Path,
        secondaries_path    : Path,
        interferograms_path : Path,
        dem_path            : Path,
        pass_labels         : Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        self.logger.section("[Stack Overview Plots]")
        self._apply_style()

        dirs  = self._setup_output_dirs()
        saved : Dict[str, Path] = {}

        primary       = np.load(str(primary_path), mmap_mode="r")
        primary_label = str(pass_labels[0]) if pass_labels else "primary"

        self.logger.subsection(f"Plotting primary SLC {tuple(primary.shape)} — {primary_label}")
        primary_amplitude = np.abs(np.asarray(primary)).astype(np.float32)
        saved["primary"]  = self._plot_amplitude(self._amplitude_db(primary_amplitude), f"Primary SLC amplitude — {primary_label}", dirs["slc"] / "primary.png")

        del primary
        gc.collect()

        secondaries   = np.load(str(secondaries_path), mmap_mode="r")
        n_secondaries = secondaries.shape[0]

        for index in range(n_secondaries):
            label = str(pass_labels[index + 1]) if pass_labels else f"pass_{index + 1:02d}"

            self.logger.subsection(f"Plotting secondary SLC {index + 1}/{n_secondaries} — {label}")
            saved[f"secondary_{index:02d}"] = self._plot_amplitude(self._amplitude_db(np.asarray(secondaries[index])), f"Secondary SLC amplitude — {label}", dirs["slc"] / f"secondary_{index + 1:02d}_{label}.png")

            gc.collect()

        del secondaries
        gc.collect()

        interferograms   = np.load(str(interferograms_path), mmap_mode="r")
        n_interferograms = interferograms.shape[0]

        for index in range(n_interferograms):
            label         = str(pass_labels[index + 1]) if pass_labels else f"pass_{index + 1:02d}"
            interferogram = np.asarray(interferograms[index])

            self.logger.subsection(f"Plotting interferogram {index + 1}/{n_interferograms} — {label}")

            outputs = self._plot_interferogram(interferogram, f"Interferogram — {primary_label} / {label}", dirs["interferograms"], f"interferogram_{index + 1:02d}_{label}")

            for kind, path in outputs.items():
                saved[f"interferogram_{index:02d}_{kind}"] = path

            self.logger.subsection(f"Plotting coherence {index + 1}/{n_interferograms} — {label}")

            coherence_outputs = self._plot_coherence_pair(primary_amplitude, interferogram, f"{primary_label} / {label}", dirs["coherence"], f"coherence_{index + 1:02d}_{label}")

            for kind, path in coherence_outputs.items():
                saved[f"coherence_{index:02d}_{kind}"] = path

            del interferogram
            gc.collect()

        del interferograms, primary_amplitude
        gc.collect()

        dem = np.asarray(np.load(str(dem_path), mmap_mode="r"), dtype=np.float32)

        self.logger.subsection(f"Plotting full DEM {tuple(dem.shape)}")
        saved["dem_full"] = self._plot_dem(dem, "DEM full", dirs["dem"] / "dem_full.png")

        del dem
        gc.collect()

        self.logger.subsection(f"Saved {len(saved)} figures → {self.images_directory}")
        return saved
