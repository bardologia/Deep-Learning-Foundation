from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from tools.data.io            import FileIO
from tools.reporting.plotting import PlotBase
from tools.monitoring.logger  import Logger


class ValueDistribution:
    PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

    def __init__(self, values: np.ndarray, scale: str = "linear", n_bins: int = 128, value_range: Optional[Tuple[float, float]] = None) -> None:
        self.values      = np.asarray(values).reshape(-1)
        self.scale       = scale
        self.n_bins      = int(n_bins)
        self.value_range = value_range

    def _finite(self) -> np.ndarray:
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            raise ValueError("ValueDistribution received no finite values; cannot summarise an empty or all-NaN field")
        return finite

    def _statistics(self, finite: np.ndarray) -> Dict[str, object]:
        quantiles = np.percentile(finite, self.PERCENTILES)

        return {
            "count"       : int(finite.size),
            "n_nonfinite" : int(self.values.size - finite.size),
            "min"         : float(finite.min()),
            "max"         : float(finite.max()),
            "mean"        : float(finite.mean()),
            "std"         : float(finite.std()),
            "median"      : float(np.median(finite)),
            "percentiles" : {f"p{p}": float(v) for p, v in zip(self.PERCENTILES, quantiles)},
        }

    def _edges(self, finite: np.ndarray) -> np.ndarray:
        if self.value_range is not None:
            lo, hi = float(self.value_range[0]), float(self.value_range[1])
        elif self.scale == "log":
            positive = finite[finite > 0.0]
            if positive.size == 0:
                raise ValueError("ValueDistribution log-scale histogram requires strictly positive values")
            lo, hi = float(positive.min()), float(positive.max())
        else:
            lo, hi = float(finite.min()), float(finite.max())

        if not hi > lo:
            hi = lo + 1e-6

        if self.scale == "log":
            return np.geomspace(max(lo, 1e-12), hi, self.n_bins + 1)

        return np.linspace(lo, hi, self.n_bins + 1)

    def _histogram(self, finite: np.ndarray, edges: np.ndarray) -> Dict[str, object]:
        counts, _ = np.histogram(finite, bins=edges)

        return {
            "scale"     : self.scale,
            "bin_edges" : [float(e) for e in edges],
            "counts"    : [int(c) for c in counts],
        }

    def compute(self) -> Dict[str, object]:
        finite = self._finite()
        edges  = self._edges(finite)

        return {
            "statistics" : self._statistics(finite),
            "histogram"  : self._histogram(finite, edges),
        }


class HistogramPlotter(PlotBase):
    def __init__(self, logger: Logger, fig_dpi: int = 150, save_dpi: int = 300) -> None:
        self.logger   = logger
        self.fig_dpi  = fig_dpi
        self.save_dpi = save_dpi

    def plot(self, histogram: Dict[str, object], statistics: Dict[str, object], title: str, x_label: str, out_path: Path, phase_ticks: bool = False) -> Path:
        self._apply_style()

        edges  = np.asarray(histogram["bin_edges"], dtype=np.float64)
        counts = np.asarray(histogram["counts"],    dtype=np.float64)
        widths = np.diff(edges)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.bar(edges[:-1], counts, width=widths, align="edge", color="#3b6ea5", edgecolor="none", zorder=2)

        ax.axvline(float(statistics["median"]), color="#b5482a", lw=1.3, ls="--", zorder=3, label=f"median = {float(statistics['median']):.3g}")

        if histogram["scale"] == "log":
            ax.set_xscale("log")

        if phase_ticks:
            ax.set_xticks(self.PHASE_TICKS)
            ax.set_xticklabels(self.PHASE_LABELS)
            ax.set_xlim(-np.pi, np.pi)

        ax.set_xlabel(x_label)
        ax.set_ylabel("pixel count")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")
        ax.margins(x=0.0)
        fig.tight_layout()

        return self._save(fig, out_path)


class StackDistributionAnalyzer:
    JSON_FILENAME = "value_distributions.json"

    def __init__(self, run_directory: Path, max_amplitude_clip: float, logger: Logger, n_bins: int = 128, fig_dpi: int = 150, save_dpi: int = 300) -> None:
        self.max_amplitude_clip = float(max_amplitude_clip)
        self.n_bins             = int(n_bins)
        self.logger             = logger
        self.distributions_dir  = Path(run_directory) / "images" / "distributions"
        self.plotter            = HistogramPlotter(logger, fig_dpi=fig_dpi, save_dpi=save_dpi)

    def _setup_output_dirs(self) -> Dict[str, Path]:
        dirs = {
            "slc"            : self.distributions_dir / "slc",
            "interferograms" : self.distributions_dir / "interferograms",
            "dem"            : self.distributions_dir / "dem",
        }
        FileIO.ensure_dirs(*dirs.values())
        return dirs

    def _pass_distributions(self, primary_path: Path, secondaries_path: Path, pass_labels: Optional[List[str]], out_dir: Path) -> Tuple[Dict[str, object], str]:
        entries: Dict[str, object] = {}

        primary       = np.load(str(primary_path), mmap_mode="r")
        primary_label = str(pass_labels[0]) if pass_labels else "primary"

        self.logger.subsection(f"Distribution — primary SLC — {primary_label}")
        entries[primary_label] = {"role": "primary", **self._complex_distributions(np.asarray(primary), primary_label, "Primary SLC", f"pass_00_{primary_label}", out_dir)}

        del primary
        gc.collect()

        secondaries   = np.load(str(secondaries_path), mmap_mode="r")
        n_secondaries = secondaries.shape[0]

        for index in range(n_secondaries):
            label = str(pass_labels[index + 1]) if pass_labels else f"pass_{index + 1:02d}"

            self.logger.subsection(f"Distribution — secondary SLC {index + 1}/{n_secondaries} — {label}")
            entries[label] = {"role": "secondary", **self._complex_distributions(np.asarray(secondaries[index]), label, "Secondary SLC", f"pass_{index + 1:02d}_{label}", out_dir)}

            gc.collect()

        del secondaries
        gc.collect()

        return entries, primary_label

    def _interferogram_distributions(self, interferograms_path: Path, pass_labels: Optional[List[str]], primary_label: str, out_dir: Path) -> Dict[str, object]:
        entries: Dict[str, object] = {}

        interferograms   = np.load(str(interferograms_path), mmap_mode="r")
        n_interferograms = interferograms.shape[0]

        for index in range(n_interferograms):
            label = str(pass_labels[index + 1]) if pass_labels else f"pass_{index + 1:02d}"

            self.logger.subsection(f"Distribution — interferogram {index + 1}/{n_interferograms} — {label}")
            entries[label] = {"reference": primary_label, **self._complex_distributions(np.asarray(interferograms[index]), f"{primary_label} / {label}", "Interferogram", f"interferogram_{index + 1:02d}_{label}", out_dir)}

            gc.collect()

        del interferograms
        gc.collect()

        return entries

    def _dem_distribution(self, dem_path: Path, out_dir: Path) -> Dict[str, object]:
        dem = np.asarray(np.load(str(dem_path), mmap_mode="r"), dtype=np.float32)

        self.logger.subsection("Distribution — DEM height")
        height = self._field_distribution(dem, "linear", "DEM height", "height [m]", out_dir / "dem_full.png")

        del dem
        gc.collect()

        return {"height": height}

    def _complex_distributions(self, field: np.ndarray, label: str, kind: str, stem: str, out_dir: Path) -> Dict[str, object]:
        amplitude = np.abs(field).astype(np.float32)
        phase     = np.angle(field).astype(np.float32)

        amplitude_result = self._field_distribution(amplitude, "log",    f"{kind} amplitude — {label}", "amplitude [linear]", out_dir / f"{stem}_amplitude.png")
        phase_result     = self._field_distribution(phase,     "linear", f"{kind} phase — {label}",     "phase [rad]",        out_dir / f"{stem}_phase.png", value_range=(-np.pi, np.pi), phase_ticks=True)

        return {"amplitude": amplitude_result, "phase": phase_result}

    def _field_distribution(self, values: np.ndarray, scale: str, title: str, x_label: str, out_path: Path, value_range: Optional[Tuple[float, float]] = None, phase_ticks: bool = False) -> Dict[str, object]:
        result = ValueDistribution(values, scale=scale, n_bins=self.n_bins, value_range=value_range).compute()

        self.plotter.plot(result["histogram"], result["statistics"], title, x_label, out_path, phase_ticks=phase_ticks)

        return result

    def _write_json(self, payload: Dict[str, object]) -> Path:
        path = self.distributions_dir / self.JSON_FILENAME
        return FileIO.save_json(payload, path)

    def run(self, primary_path: Path, secondaries_path: Path, interferograms_path: Path, dem_path: Path, pass_labels: Optional[List[str]] = None) -> Dict[str, Path]:
        self.logger.section("[Stack Value Distributions]")
        self.plotter._apply_style()

        dirs = self._setup_output_dirs()

        passes, primary_label = self._pass_distributions(primary_path, secondaries_path, pass_labels, dirs["slc"])
        interferograms        = self._interferogram_distributions(interferograms_path, pass_labels, primary_label, dirs["interferograms"])
        dem                   = self._dem_distribution(dem_path, dirs["dem"])

        payload = {
            "reference_pass"     : primary_label,
            "max_amplitude_clip" : self.max_amplitude_clip,
            "n_bins"             : self.n_bins,
            "passes"             : passes,
            "interferograms"     : interferograms,
            "dem"                : dem,
        }

        json_path = self._write_json(payload)
        self.logger.subsection(f"Wrote value distributions → {json_path}")

        return {
            "value_distributions_json" : json_path,
            "distributions_dir"        : self.distributions_dir,
        }
