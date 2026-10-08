"""Export of tensorboard scalar logs to publication-quality curve figures.

Reads the event files of a run, pairs training and validation tags into shared
figures, renders one PNG per tag group, and additionally overlays the seeds of
a multi-seed trial in a single figure per tag.
"""

from __future__ import annotations

import re
import textwrap

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy             as np

from matplotlib.ticker import MaxNLocator

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from configuration.diagnostics  import TensorboardExportConfig, TensorboardExportEntryConfig
from tools.reporting.plotting   import PlotBase
from tools.runtime.run_selector import TensorboardRunSelector

ScalarSeries = Tuple[np.ndarray, np.ndarray]


@dataclass
class CurveGroup:
    """Scalar series that share one figure.

    Attributes:
        stem   : Relative output path stem of the figure, without extension.
        title  : Human-readable tag name used for the figure title.
        series : Tuples of (role label or None, step array, value array), where the role is "Training" or "Validation".
    """
    stem   : str
    title  : str
    series : List[Tuple[Optional[str], np.ndarray, np.ndarray]]


class TensorboardScalarReader:
    """Loads every scalar series from a run's tensorboard event files.

    Attributes:
        config : Export configuration pointing at the tensorboard directory.
    """

    def __init__(self, config: TensorboardExportConfig) -> None:
        """Initialises the reader.

        Args:
            config : Export configuration for one run directory.
        """
        self.config = config

    def read(self) -> Dict[str, ScalarSeries]:
        """Reads all scalar series recorded for the run.

        Returns:
            Mapping of tag to (steps, values) arrays of equal length.

        Raises:
            FileNotFoundError : If the run has no tensorboard directory.
            ValueError        : If the event files contain no scalar series.
        """
        tensorboard_dir = self.config.tensorboard_directory
        if not tensorboard_dir.is_dir():
            raise FileNotFoundError(f"No '{self.config.tensorboard_dirname}' directory at {tensorboard_dir}; expected a training run directory holding tensorboard event files")

        accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
        accumulator.Reload()

        tags = accumulator.Tags()["scalars"]
        if not tags:
            raise ValueError(f"Tensorboard log at {tensorboard_dir} contains no scalar series")

        series = {}
        for tag in tags:
            events      = accumulator.Scalars(tag)
            steps       = np.array([event.step  for event in events], dtype=np.int64)
            values      = np.array([event.value for event in events], dtype=np.float64)
            series[tag] = (steps, values)

        return series


class ScalarTagGrouper:
    """Groups scalar tags that differ only by their train/validation segment."""

    SPLIT_LABELS = {"train": "Training", "val": "Validation", "validation": "Validation"}
    SERIES_ORDER = ("Training", "Validation")

    def _split_tag(self, tag: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns the tag with its split segment removed and the split label, or (None, None)."""
        segments = tag.split("/")
        for index, segment in enumerate(segments):
            if segment in self.SPLIT_LABELS:
                reduced = "/".join(segments[:index] + segments[index + 1:])
                return reduced, self.SPLIT_LABELS[segment]
        return None, None

    def _stem(self, tag: str) -> str:
        """Returns the filesystem-safe path stem derived from a tag."""
        segments = [re.sub(r"[^A-Za-z0-9._-]+", "_", segment) for segment in tag.split("/") if segment]
        return "/".join(segments)

    def group(self, series_by_tag: Dict[str, ScalarSeries]) -> List[CurveGroup]:
        """Groups tags into figures, pairing matching training and validation curves.

        Args:
            series_by_tag : Mapping of tag to (steps, values) arrays.

        Returns:
            Curve groups sorted by output stem; a group holds two series when a
            train/validation pair was found and one otherwise.

        Raises:
            ValueError : If two tags resolve to the same split role, or if two groups would write to the same output stem.
        """
        buckets : Dict[str, list] = {}
        singles : List[str]       = []

        for tag in series_by_tag:
            reduced, label = self._split_tag(tag)
            if label is None:
                singles.append(tag)
            else:
                buckets.setdefault(reduced, []).append((label, tag))

        groups = []
        for reduced, members in buckets.items():
            if len(members) == 1:
                singles.append(members[0][1])
                continue

            labels = [label for label, _tag in members]
            if len(set(labels)) != len(labels):
                raise ValueError(f"Ambiguous train/validation pairing for '{reduced}': tags {[tag for _label, tag in members]} resolve to duplicate roles")

            ordered = sorted(members, key=lambda member: self.SERIES_ORDER.index(member[0]))
            groups.append(CurveGroup(stem=self._stem(reduced), title=reduced, series=[(label, *series_by_tag[tag]) for label, tag in ordered]))

        for tag in singles:
            groups.append(CurveGroup(stem=self._stem(tag), title=tag, series=[(None, *series_by_tag[tag])]))

        stems      = [group.stem for group in groups]
        duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
        if duplicates:
            raise ValueError(f"Tag grouping produced colliding plot paths: {duplicates}")

        return sorted(groups, key=lambda group: group.stem)


class CurveLabeler:
    """Turns snake_case tensorboard tags into readable axis labels and titles."""

    ACRONYMS = {"lr", "gpu", "cpu", "ram", "vram", "shm", "rss", "vms", "uss", "ema", "l1", "l2", "mse", "mae", "rmse", "ssim", "bic", "f1", "gt", "aux", "iqr"}

    UNIT_SUFFIXES = (("mb_s", "MB/s"), ("gb_s", "GB/s"), ("gb", "GB"), ("mb", "MB"), ("pct", "%"), ("w", "W"), ("c", "°C"))

    WRAP_WIDTH = 52

    def _token(self, token: str) -> str:
        """Returns a token upper-cased when it is a known acronym and capitalised otherwise."""
        match = re.fullmatch(r"([a-z]+)(\d*)", token)
        if match and match.group(1) in self.ACRONYMS:
            return match.group(1).upper() + match.group(2)
        return token.capitalize()

    def _split_unit(self, segment: str) -> Tuple[List[str], Optional[str]]:
        """Returns the segment's tokens with any trailing unit suffix stripped and named."""
        tokens = segment.split("_")
        for suffix, unit in self.UNIT_SUFFIXES:
            suffix_tokens = suffix.split("_")
            if len(tokens) > len(suffix_tokens) and tokens[-len(suffix_tokens):] == suffix_tokens:
                return tokens[:-len(suffix_tokens)], unit
        return tokens, None

    def _segment(self, segment: str, with_unit: bool) -> str:
        """Returns one tag segment rendered as words, optionally with its unit in parentheses."""
        tokens, unit = self._split_unit(segment) if with_unit else (segment.split("_"), None)
        text         = " ".join(self._token(token) for token in tokens if token)
        return f"{text} ({unit})" if unit else text

    def y_label(self, title: str) -> str:
        """Returns the y-axis label built from the last segment of a tag, unit included."""
        return self._segment(title.split("/")[-1], with_unit=True)

    def title(self, title: str) -> str:
        """Returns the wrapped figure title built from every segment of a tag."""
        segments = title.split("/")
        rendered = [self._segment(segment, with_unit=(index == len(segments) - 1)) for index, segment in enumerate(segments)]
        return "\n".join(textwrap.wrap(" / ".join(rendered), self.WRAP_WIDTH))


class ScalarCurvePlots(PlotBase):
    """Renders one training-curve figure per tag group.

    Attributes:
        labeler : Label formatter shared by every figure.
    """

    save_dpi = 300

    SERIES_COLORS = {"Training": "#1f77b4", "Validation": "#ff7f0e", None: "#1f77b4"}
    SPARSE_POINTS = 40
    LOG_SPAN      = 1e3

    def __init__(self) -> None:
        """Initialises the renderer with a fresh curve labeler."""
        self.labeler = CurveLabeler()

    def _log_scaled(self, value_arrays) -> bool:
        """Returns whether the pooled values are strictly positive and span at least LOG_SPAN."""
        values = np.concatenate(list(value_arrays))
        values = values[np.isfinite(values)]

        if values.size == 0 or values.min() <= 0:
            return False
        return bool(values.max() / values.min() >= self.LOG_SPAN)

    def _decorate(self, fig, ax, title: str, value_arrays, n_series: int) -> None:
        """Applies the scale, grid, labels and legend shared by all curve figures."""
        if self._log_scaled(value_arrays):
            ax.set_yscale("log")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(which="major", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)

        ax.set_title(self.labeler.title(title))
        ax.set_xlabel("Step")
        ax.set_ylabel(self.labeler.y_label(title))

        if n_series > 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        fig.tight_layout()

    def render(self, group: CurveGroup, path: Path) -> Path:
        """Renders one tag group as a step-versus-value curve figure.

        Args:
            group : Curve group holding one or two series over training steps.
            path  : Destination file for the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=(6.2, 3.9))

        for label, steps, values in group.series:
            marker = "o" if steps.size <= self.SPARSE_POINTS else None
            ax.plot(steps, values, lw=1.4, color=self.SERIES_COLORS[label], label=label, marker=marker, markersize=3.0)

        self._decorate(fig, ax, group.title, [values for _label, _steps, values in group.series], len(group.series))
        return self._save(fig, path)


@dataclass
class SeedCurveGroup:
    """Scalar series of several seeds that share one overlay figure.

    Attributes:
        stem   : Relative output path stem of the figure, without extension.
        title  : Human-readable tag name used for the figure title.
        series : Tuples of (seed name, role label or None, step array, value array).
    """
    stem   : str
    title  : str
    series : List[Tuple[str, Optional[str], np.ndarray, np.ndarray]]


class SeedScalarMerger:
    """Merges the per-seed curve groups of a trial into shared overlay groups."""

    def merge(self, groups_by_seed: Dict[str, List[CurveGroup]]) -> List[SeedCurveGroup]:
        """Collects the series of every seed under their common tag stem.

        Args:
            groups_by_seed : Curve groups keyed by seed directory name.

        Returns:
            Seed curve groups sorted by output stem.
        """
        merged: Dict[str, SeedCurveGroup] = {}

        for seed in sorted(groups_by_seed):
            for group in groups_by_seed[seed]:
                entry = merged.setdefault(group.stem, SeedCurveGroup(stem=group.stem, title=group.title, series=[]))
                for role, steps, values in group.series:
                    entry.series.append((seed, role, steps, values))

        return sorted(merged.values(), key=lambda group: group.stem)


class SeedOverlayPlots(ScalarCurvePlots):
    """Renders one figure per tag overlaying every seed of a trial."""

    PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

    ROLE_STYLES = {"Training": "-", "Validation": "--", None: "-"}

    @staticmethod
    def _series_label(seed: str, role: Optional[str]) -> str:
        """Returns the legend entry for one seed and split role."""
        if role is None:
            return seed
        return f"{seed} {'train' if role == 'Training' else 'val'}"

    def render(self, group: SeedCurveGroup, path: Path) -> Path:
        """Renders one tag with every seed overlaid, colour by seed and dash by split role.

        Args:
            group : Seed curve group holding one series per seed and role.
            path  : Destination file for the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=(6.2, 3.9))

        seeds = sorted({seed for seed, _role, _steps, _values in group.series})
        for seed, role, steps, values in group.series:
            color = self.PALETTE[seeds.index(seed) % len(self.PALETTE)]
            ax.plot(steps, values, lw=1.2, color=color, linestyle=self.ROLE_STYLES[role], label=self._series_label(seed, role))

        self._decorate(fig, ax, group.title, [values for _seed, _role, _steps, values in group.series], len(group.series))
        return self._save(fig, path)


class TensorboardExport:
    """Exports every scalar series of one run to curve figures.

    Attributes:
        config : Export configuration for the run.
        logger : Logger used for progress reporting.
    """

    def __init__(self, config: TensorboardExportConfig, logger) -> None:
        """Initialises the exporter.

        Args:
            config : Export configuration for one run directory.
            logger : Logger used for progress reporting.
        """
        self.config = config
        self.logger = logger

    def _read(self) -> Dict[str, ScalarSeries]:
        """Reads and reports the scalar series of the run."""
        self.logger.subsection(f"Reading tensorboard log: {self.config.tensorboard_directory}")
        series = TensorboardScalarReader(self.config).read()
        self.logger.ok(f"Loaded {len(series)} scalar series")
        return series

    def _group(self, series: Dict[str, ScalarSeries]) -> List[CurveGroup]:
        """Groups the scalar series into figures and reports how many are shared."""
        groups = ScalarTagGrouper().group(series)
        shared = sum(1 for group in groups if len(group.series) > 1)
        self.logger.ok(f"Grouped into {len(groups)} figures ({shared} shared train/validation)")
        return groups

    def _render(self, groups: List[CurveGroup]) -> List[Path]:
        """Renders every group into the run's plots directory under a progress bar and returns the file paths."""
        renderer = ScalarCurvePlots()
        paths    = []

        with self.logger.track() as progress:
            task = progress.add_task(f"{Path(self.config.run_directory).name} plots", total=len(groups))
            for group in groups:
                paths.append(renderer.render(group, self.config.plots_directory / f"{group.stem}.png"))
                progress.advance(task)

        return paths

    def run(self) -> dict:
        """Reads, groups and renders the run's scalar logs.

        Returns:
            Mapping with the run and plots directories, the number of scalar
            series read, the number of plots written, and their paths.
        """
        series = self._read()
        groups = self._group(series)
        paths  = self._render(groups)

        self.logger.ok(f"Wrote {len(paths)} plots to {self.config.plots_directory}")

        return {
            "run_directory"   : str(self.config.run_directory),
            "plots_directory" : str(self.config.plots_directory),
            "n_series"        : len(series),
            "n_plots"         : len(paths),
            "plot_paths"      : paths,
        }


class SeedRunGrouper:
    """Collects seed run directories under the trial directory that owns them."""

    SEED_DIR_PATTERN = re.compile(r"seed\d+")

    @classmethod
    def trials(cls, run_dirs: list[Path]) -> Dict[Path, list[Path]]:
        """Returns the trial directories holding more than one seed run.

        Args:
            run_dirs : Candidate run directories.

        Returns:
            Mapping of trial directory to its sorted seed directories,
            restricted to trials with at least two seeds.
        """
        groups: Dict[Path, list[Path]] = {}

        for run_dir in run_dirs:
            if cls.SEED_DIR_PATTERN.fullmatch(run_dir.name):
                groups.setdefault(run_dir.parent, []).append(run_dir)

        return {trial_dir: sorted(seed_dirs) for trial_dir, seed_dirs in sorted(groups.items()) if len(seed_dirs) > 1}


class SeedOverlayExport:
    """Exports seed-overlay curve figures for one multi-seed trial.

    Attributes:
        trial_dir    : Trial directory that owns the seed runs.
        seed_dirs    : Seed run directories to overlay.
        entry_config : Entry configuration used to derive per-seed configs.
        logger       : Logger used for progress reporting.
    """

    def __init__(self, trial_dir: Path, seed_dirs: list[Path], entry_config: TensorboardExportEntryConfig, logger) -> None:
        """Initialises the overlay exporter.

        Args:
            trial_dir    : Trial directory that owns the seed runs.
            seed_dirs    : Seed run directories to overlay.
            entry_config : Entry configuration used to derive per-seed configs.
            logger       : Logger used for progress reporting.
        """
        self.trial_dir    = Path(trial_dir)
        self.seed_dirs    = seed_dirs
        self.entry_config = entry_config
        self.logger       = logger

    def _read_seed(self, seed_dir: Path) -> List[CurveGroup]:
        """Returns the grouped scalar series of one seed run and logs how many series it held, so a slow event read is pinned to its seed."""
        config = self.entry_config.to_config(seed_dir)
        series = TensorboardScalarReader(config).read()

        self.logger.info(f"{seed_dir.name}: {len(series)} scalar series")
        return ScalarTagGrouper().group(series)

    def _render(self, merged: List[SeedCurveGroup], plots_directory: Path) -> List[Path]:
        """Renders one overlay figure per merged tag under a progress bar and returns the file paths."""
        renderer = SeedOverlayPlots()
        paths    = []

        with self.logger.track() as progress:
            task = progress.add_task(f"{self.trial_dir.name} overlays", total=len(merged))
            for group in merged:
                paths.append(renderer.render(group, plots_directory / f"{group.stem}.png"))
                progress.advance(task)

        return paths

    def run(self) -> dict:
        """Reads every seed and writes the overlay figures into the trial directory.

        Returns:
            Mapping with the trial and plots directories, the seed count, the
            number of plots written, and their paths.
        """
        self.logger.subsection(f"Seed overlay export: {self.trial_dir.name} ({len(self.seed_dirs)} seeds)")

        groups_by_seed = {seed_dir.name: self._read_seed(seed_dir) for seed_dir in self.seed_dirs}
        merged         = SeedScalarMerger().merge(groups_by_seed)

        plots_directory = self.trial_dir / self.entry_config.output_subdir
        paths           = self._render(merged, plots_directory)

        self.logger.ok(f"{self.trial_dir.name}: {len(paths)} seed-overlay plots -> {plots_directory}")

        return {
            "trial_directory" : str(self.trial_dir),
            "plots_directory" : str(plots_directory),
            "n_seeds"         : len(self.seed_dirs),
            "n_plots"         : len(paths),
            "plot_paths"      : paths,
        }


class TensorboardExportBatch:
    """Exports curve figures for every run matched by an entry-point filter.

    Attributes:
        entry_config : Entry configuration holding the run root and filter.
        logger       : Logger used for progress reporting.
    """

    def __init__(self, entry_config: TensorboardExportEntryConfig, logger) -> None:
        """Initialises the batch exporter.

        Args:
            entry_config : Entry configuration holding the run root and filter.
            logger       : Logger used for progress reporting.
        """
        self.entry_config = entry_config
        self.logger       = logger

    def _select_runs(self) -> list[Path]:
        """Returns the run directories matched by the configured run filter."""
        selector = TensorboardRunSelector(self.entry_config.runs_dir, self.entry_config.tensorboard_dirname, self.logger)
        return selector.resolve(self.entry_config.run_filter)

    def _run_label(self, run_dir: Path) -> str:
        """Returns the run's path below the runs root, which tells nested seed runs apart."""
        return str(Path(run_dir).relative_to(Path(self.entry_config.runs_dir)))

    def _export_run(self, run_dir: Path, position: int, total: int) -> dict:
        """Exports one run's curve figures under a numbered section and returns its export result.

        Args:
            run_dir  : Run directory holding the tensorboard event files.
            position : One-based position of the run in the batch.
            total    : Number of runs in the batch.

        Returns:
            The TensorboardExport result mapping of the run.
        """
        self.logger.section(f"Tensorboard plot export [{position}/{total}] {self._run_label(run_dir)}")

        config = self.entry_config.to_config(run_dir)
        result = TensorboardExport(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['n_plots']} plots -> {config.plots_directory}")
        return result

    def _export_seed_overlays(self, run_dirs: list[Path]) -> list[dict]:
        """Exports one overlay figure set per multi-seed trial found among the runs."""
        return [SeedOverlayExport(trial_dir, seed_dirs, self.entry_config, self.logger).run() for trial_dir, seed_dirs in SeedRunGrouper.trials(run_dirs).items()]

    def _summarize(self, run_dirs: list[Path], results: list[dict], overlays: list[dict]) -> None:
        """Closes the batch with one row per run, one per multi-seed trial and the totals, so completeness need not be counted by hand.

        Args:
            run_dirs : Run directories in batch order.
            results  : TensorboardExport result mappings in the same order.
            overlays : SeedOverlayExport result mappings, one per multi-seed trial.
        """
        rows = []
        for run_dir, result in zip(run_dirs, results):
            rows.append({
                "Run"    : self._run_label(run_dir),
                "Series" : result["n_series"],
                "Plots"  : result["n_plots"],
                "Output" : result["plots_directory"],
            })

        self.logger.metrics_table(rows, columns=["Run", "Series", "Plots", "Output"], title="Tensorboard export summary")

        if overlays:
            overlay_rows = [{"Trial": self._run_label(Path(overlay["trial_directory"])), "Seeds": overlay["n_seeds"], "Plots": overlay["n_plots"], "Output": overlay["plots_directory"]} for overlay in overlays]
            self.logger.metrics_table(overlay_rows, columns=["Trial", "Seeds", "Plots", "Output"], title="Seed overlay summary")

        n_plots         = sum(result["n_plots"] for result in results)
        n_overlay_plots = sum(overlay["n_plots"] for overlay in overlays)

        self.logger.ok(f"Exported {len(results)} run(s) ({n_plots} plots); {len(overlays)} multi-seed trial(s) ({n_overlay_plots} overlay plots)")

    def run(self) -> list[dict]:
        """Exports every selected run and the seed overlays of its multi-seed trials, then closes with the batch summary.

        Returns:
            Per-run export results followed by the per-trial overlay results.
        """
        run_dirs = self._select_runs()
        results  = [self._export_run(run_dir, position, len(run_dirs)) for position, run_dir in enumerate(run_dirs, start=1)]
        overlays = self._export_seed_overlays(run_dirs)

        self._summarize(run_dirs, results, overlays)

        return results + overlays
