from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import matplotlib.pyplot as plt
import numpy             as np

from pipelines.comparison.metric_table             import MetricTableRenderer
from pipelines.comparison.spatial_stats            import SpatialDispersion
from pipelines.processing.param_extraction.io      import ParameterRunMeta
from pipelines.shared.comparison.comparison_report import ComparisonReportBase
from tools.data.io             import FileIO
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter
from tools.reporting.reporting import ReportAssets
from tools.reporting.plotting  import PlotBase
from tools.monitoring.logger   import Logger


@dataclass
class ParamTrial:
    name               : str
    run_dir            : Path
    k_max              : int
    lambda_k           : float
    sigma_init_divisor : float
    fit_sigma          : bool
    fit_amplitude      : bool
    fit_mean           : bool
    dataset            : str = ""
    metrics            : dict = field(default_factory=dict)

    @property
    def free_per_gaussian(self) -> int:
        return int(self.fit_sigma) + int(self.fit_amplitude) + int(self.fit_mean)

    @property
    def label(self) -> str:
        base = Path(self.name).name.removeprefix("params_")
        return f"{self.dataset} · {base}" if self.dataset else base


class ParamTrialCollector:
    MARKER = ParameterRunMeta.FILENAME

    def __init__(self, params_dir: Path, run_tags: list[str], logger: Logger) -> None:
        self.params_dir = params_dir
        self.run_tags   = run_tags
        self.logger     = logger

    def _is_trial(self, run_dir: Path) -> bool:
        return (run_dir / self.MARKER).exists() and (run_dir / "parameters.npy").exists() and not ParameterRunMeta.is_external(run_dir)

    def _discover_tags(self) -> list[str]:
        if self.run_tags:
            return list(self.run_tags)

        return [
            str(marker.parent.relative_to(self.params_dir))
            for marker in sorted(self.params_dir.rglob(self.MARKER))
            if (marker.parent / "parameters.npy").exists() and not ParameterRunMeta.is_external(marker.parent)
        ]

    def _dataset_of(self, tag: str) -> str:
        parts = Path(tag).parts
        return parts[0] if len(parts) > 1 else ""

    def _build_trial(self, tag: str) -> ParamTrial:
        run_dir = self.params_dir / tag
        meta    = FileIO.load_json(run_dir / self.MARKER)

        return ParamTrial(
            name               = tag,
            run_dir            = run_dir,
            k_max              = int(meta["k_max"]),
            lambda_k           = float(meta["lambda_k"]),
            sigma_init_divisor = float(meta["sigma_init_divisor"]),
            fit_sigma          = bool(meta["fit_sigma"]),
            fit_amplitude      = bool(meta["fit_amplitude"]),
            fit_mean           = bool(meta["fit_mean"]),
            dataset            = self._dataset_of(tag),
        )

    def collect(self) -> list[ParamTrial]:
        self.logger.section("Collecting parameter-extraction trials")

        trials = []
        for tag in self._discover_tags():
            ParameterRunMeta.reject_external(self.params_dir / tag, "enter a Gaussian-fit comparison")

            if not self._is_trial(self.params_dir / tag):
                self.logger.error(f"Not a parameter-extraction trial: {self.params_dir / tag}")
                continue

            trial = self._build_trial(tag)
            self.logger.info(f"{trial.name:<48} K={trial.k_max} free/gauss={trial.free_per_gaussian}")
            trials.append(trial)

        if not trials:
            self.logger.error(f"No parameter-extraction trials found under {self.params_dir}")

        return sorted(trials, key=lambda item: (item.k_max, item.dataset, item.name))


class ParamMetrics:

    EPSILON       = 1e-12
    COLLAPSE_RATE = 0.01

    def __init__(self, pixel_sample: int, block_size: int, logger: Logger) -> None:
        self.pixel_sample  = pixel_sample
        self.block_size    = block_size
        self.logger        = logger
        self._height_cache = {}

    def _height_bins(self, trial: ParamTrial) -> int:
        tomogram_path = trial.run_dir.parent.parent / "data" / "tomogram_full.npy"
        if tomogram_path not in self._height_cache:
            tomogram = np.load(tomogram_path, mmap_mode="r")
            self._height_cache[tomogram_path] = int(tomogram.shape[0])

        return self._height_cache[tomogram_path]

    def _load(self, trial: ParamTrial) -> tuple:
        meta        = FileIO.load_json(trial.run_dir / "param_extraction_meta.json")
        summary     = FileIO.load_json(trial.run_dir / "fit_metrics_summary.json")
        diagnostics = np.load(trial.run_dir / "fit_diagnostics.npz")
        parameters  = np.load(trial.run_dir / "parameters.npy", mmap_mode="r")

        return meta, summary, diagnostics, parameters

    def _bic_median(self, trial: ParamTrial, diagnostics, best_k: np.ndarray, n_data: int) -> float:
        mse_per_k = diagnostics["mse_per_k"]
        best_idx  = np.clip(best_k.astype(np.int64) - 1, 0, mse_per_k.shape[0] - 1)

        mse_best  = np.take_along_axis(mse_per_k, best_idx[None, :, :], axis=0)[0]
        n_params  = best_k.astype(np.float64) * trial.free_per_gaussian

        valid = (best_k > 0) & (mse_best > self.EPSILON)
        bic   = n_data * np.log(np.where(valid, mse_best, 1.0)) + n_params * np.log(n_data)

        return float(np.median(bic[valid]))

    def _separation_median(self, mus: np.ndarray, active: np.ndarray, fitted: np.ndarray) -> float:
        slots   = mus.shape[0]
        mus_f   = mus.reshape(slots, -1).T
        act_f   = active.reshape(slots, -1).T
        fit_idx = np.where(fitted.reshape(-1))[0]

        if fit_idx.size > self.pixel_sample:
            picks   = np.linspace(0, fit_idx.size - 1, self.pixel_sample).astype(np.int64)
            fit_idx = fit_idx[picks]

        sample = np.where(act_f[fit_idx], mus_f[fit_idx], np.nan)
        sample = np.sort(sample, axis=1)

        gaps    = np.diff(sample, axis=1)
        has_gap = np.isfinite(gaps).any(axis=1)

        nearest = np.nanmin(np.where(np.isfinite(gaps), gaps, np.nan)[has_gap], axis=1)

        return float(np.median(nearest)) if nearest.size > 0 else float("nan")

    def _coherence(self, mus: np.ndarray, amplitudes: np.ndarray, fitted: np.ndarray) -> float:
        dominant = amplitudes.argmax(axis=0)
        mu_dom   = np.take_along_axis(mus, dominant[None, :, :], axis=0)[0]

        mu_dom = np.where(fitted, mu_dom, np.nan)

        return SpatialDispersion.block_std(mu_dom, self.block_size)

    def _summarise(self, trial: ParamTrial, meta, summary, diagnostics, parameters, n_data: int) -> dict:
        global_summary = summary["global_summary"]
        per_k_summary  = summary["per_k_summary"]

        threshold  = float(meta["activity_threshold"])
        amplitudes = np.asarray(parameters[0::3], dtype=np.float32)
        mus        = np.asarray(parameters[1::3], dtype=np.float32)

        best_k = diagnostics["best_k_map"]
        active = amplitudes > threshold
        fitted = best_k > 0

        slot_active_rate = active.reshape(active.shape[0], -1).mean(axis=1)

        return {
            "r2_median"                   : float(global_summary["r2_median"]),
            "bic_median"                  : self._bic_median(trial, diagnostics, best_k, n_data),
            "fit_coverage"                : float(global_summary["n_fitted"]) / float(global_summary["n_pixels"]),
            "active_slot_mean"            : float(active.sum(axis=0)[fitted].mean()),
            "collapsed_slot_fraction"     : float((slot_active_rate < self.COLLAPSE_RATE).mean()),
            "mu_separation_median"        : self._separation_median(mus, active, fitted),
            "mu_dominant_stability_error" : self._coherence(mus, amplitudes, fitted),
            "k_ambiguous_fraction"        : float(per_k_summary["k_ambiguous_fraction"]),
        }

    def compute(self, trial: ParamTrial) -> dict:
        self.logger.subsection(f"Measuring {trial.name}")

        n_data = self._height_bins(trial)
        meta, summary, diagnostics, parameters = self._load(trial)

        return self._summarise(trial, meta, summary, diagnostics, parameters, n_data)


class ParamComparisonPlots(PlotBase):

    PANELS = [
        ("bic_median",                  "BIC (lower better)",          "Complexity-penalised fit"),
        ("r2_median",                   "R² median",                   "Variance explained"),
        ("collapsed_slot_fraction",     "Collapsed slot fraction",     "Slot collapse"),
        ("mu_dominant_stability_error", "Dominant-height block std (m)", "Spatial coherence"),
        ("active_slot_mean",            "Mean active slots",           "Slot usage"),
    ]

    PALETTE = ["#3b6ea5", "#a5453b", "#3ba56e", "#a59a3b", "#6e3ba5"]

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def _colour(self, trials: list[ParamTrial]) -> list:
        orders = sorted({trial.k_max for trial in trials})
        index  = {value: position for position, value in enumerate(orders)}

        return [self.PALETTE[index[trial.k_max] % len(self.PALETTE)] for trial in trials]

    def _bar(self, trials: list[ParamTrial], key: str, y_label: str, title: str, path: Path):
        self._apply_style()

        labels = [trial.label for trial in trials]
        values = [trial.metrics[key] for trial in trials]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(range(len(labels)), values, color=self._colour(trials), width=0.65)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(title)

        fig.tight_layout()
        return self._save(fig, path)

    def render(self, trials: list[ParamTrial]) -> list[Path]:
        written = []

        for key, y_label, title in self.PANELS:
            path = self.out_dir / "images" / f"{key}.png"
            written.append(self._bar(trials, key, y_label, title, path))

        return written


class ParamComparisonReport(ComparisonReportBase):

    CONFIG_COLUMNS = [
        ("label",              "Trial"),
        ("k_max",              "K"),
        ("lambda_k",           "lambda"),
        ("sigma_init_divisor", "sigma init"),
        ("fit_sigma",          "fit sigma"),
        ("fit_amplitude",      "fit amp"),
        ("fit_mean",           "fit mean"),
    ]

    METRIC_COLUMNS = [
        ("r2_median",                   "R²"),
        ("bic_median",                  "BIC"),
        ("fit_coverage",                "Coverage"),
        ("active_slot_mean",            "Active slots"),
        ("collapsed_slot_fraction",     "Collapsed"),
        ("mu_separation_median",        "Separation"),
        ("mu_dominant_stability_error", "Coherence"),
        ("k_ambiguous_fraction",        "Ambiguous"),
    ]

    ORIENTATION = {
        "r2_median"                   : "higher",
        "bic_median"                  : "lower",
        "fit_coverage"                : "higher",
        "active_slot_mean"            : None,
        "collapsed_slot_fraction"     : "lower",
        "mu_separation_median"        : None,
        "mu_dominant_stability_error" : "lower",
        "k_ambiguous_fraction"        : "lower",
    }

    RANK_METRICS = [
        ("bic_median",                  "BIC"),
        ("r2_median",                   "R²"),
        ("mu_dominant_stability_error", "Coherence"),
        ("k_ambiguous_fraction",        "Decisiveness"),
    ]

    def __init__(self, out_dir: Path, logger: Logger) -> None:
        self.out_dir = Path(out_dir)
        self.logger  = logger
        self.assets  = ReportAssets(base=self.out_dir)

    def _config_table(self, trials: list[ParamTrial]) -> list[str]:
        table = MarkdownTable([label for _, label in self.CONFIG_COLUMNS])

        for trial in trials:
            cells = [ScalarFormatter.format_scalar(getattr(trial, key), precision=4) for key, _ in self.CONFIG_COLUMNS]
            table.add_row(*cells)

        return ["## Trial configurations\n"] + table.render() + [""]

    def _within_k_sections(self, trials: list[ParamTrial]) -> list[str]:
        orders = sorted({trial.k_max for trial in trials})
        lines  = []

        for order in orders:
            group = [trial for trial in trials if trial.k_max == order]
            if len(group) < 2:
                continue

            intro = f"Ranking the K={order} variants on complexity-penalised fit (BIC), variance explained, spatial coherence of the dominant height, and selection decisiveness. R² alone is not used for selection because it grows with free parameters."
            lines += self._rank_section("Trial", f"K={order} leaderboard", intro, self.RANK_METRICS, group)

        return lines

    def _notes(self) -> list[str]:
        return [
            "## Notes\n",
            "BIC penalises the extra free parameters a larger K spends, so it is comparable within a fixed K family but is reported, not used, across K. Collapsed slot fraction and slot separation expose Gaussian-slot collapse: a high collapsed fraction with small separation means the extra slots are fitting noise rather than distinct scatterers. Spatial coherence (block standard deviation of the dominant scatterer height) is low when fitted parameters vary smoothly across neighbouring pixels and high when the fit is locking onto speckle.\n",
        ]

    def _write_overview(self, trials: list[ParamTrial]) -> Path:
        lines  = self.assets.header("Parameter-Extraction Comparison Overview")
        lines += [f"Compared {len(trials)} Gaussian-fit trials across {len(set(t.k_max for t in trials))} K families.\n"]
        lines += self._config_table(trials)
        lines += self._within_k_sections(trials)
        lines += self._notes()

        out = self.out_dir / "overview.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_metrics(self, trials: list[ParamTrial], plots: list[Path]) -> Path:
        table = MetricTableRenderer.render(
            rows           = trials,
            leading        = [("Trial", lambda trial: f"`{trial.label}`"), ("K", lambda trial: str(trial.k_max))],
            metric_columns = self.METRIC_COLUMNS,
            orientation    = self.ORIENTATION,
            group_of       = lambda trial: trial.k_max,
        )

        lines  = self.assets.header("Parameter-Extraction Metrics")
        lines += ["> Best value per metric in **bold** within each K family (↑ higher is better, ↓ lower is better). The K families are separate deliverables and are never ranked against each other.\n"]
        lines += ["## Per-trial metrics\n", *table, ""]

        if plots:
            lines += ["## Comparison plots\n"]
            for path in plots:
                lines += self.assets.image(path.stem, path)

        out = self.out_dir / "metrics.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def write_all(self, trials: list[ParamTrial], plots: list[Path]) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.section("Writing comparison reports")

        written = [self._write_overview(trials), self._write_metrics(trials, plots)]
        for path in written:
            self.logger.info(f"Report written to: {path}")

        return written


class ParamExtractionComparisonPipeline:
    def __init__(self, config, out_dir: Path, logger: Logger) -> None:
        self.config  = config
        self.out_dir = out_dir
        self.logger  = logger

    def _collect(self) -> list[ParamTrial]:
        collector = ParamTrialCollector(Path(self.config.params_dir), list(self.config.run_tags), self.logger)
        return collector.collect()

    def _measure(self, trials: list[ParamTrial]) -> None:
        metrics = ParamMetrics(self.config.pixel_sample, self.config.block_size, self.logger)

        for trial in trials:
            trial.metrics = metrics.compute(trial)

    def _plot(self, trials: list[ParamTrial]) -> list[Path]:
        if not self.config.make_plots:
            return []
        return ParamComparisonPlots(self.out_dir).render(trials)

    def _report(self, trials: list[ParamTrial], plots: list[Path]) -> list[Path]:
        return ParamComparisonReport(self.out_dir, self.logger).write_all(trials, plots)

    def run(self) -> list[Path]:
        trials = self._collect()
        if not trials:
            raise RuntimeError("No parameter-extraction trials to compare")

        self._measure(trials)
        plots = self._plot(trials)

        return self._report(trials, plots)
