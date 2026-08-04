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
    stem   : str
    title  : str
    series : List[Tuple[Optional[str], np.ndarray, np.ndarray]]


class TensorboardScalarReader:
    def __init__(self, config: TensorboardExportConfig) -> None:
        self.config = config

    def read(self) -> Dict[str, ScalarSeries]:
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
    SPLIT_LABELS = {"train": "Training", "val": "Validation", "validation": "Validation"}
    SERIES_ORDER = ("Training", "Validation")

    def _split_tag(self, tag: str) -> Tuple[Optional[str], Optional[str]]:
        segments = tag.split("/")
        for index, segment in enumerate(segments):
            if segment in self.SPLIT_LABELS:
                reduced = "/".join(segments[:index] + segments[index + 1:])
                return reduced, self.SPLIT_LABELS[segment]
        return None, None

    def _stem(self, tag: str) -> str:
        segments = [re.sub(r"[^A-Za-z0-9._-]+", "_", segment) for segment in tag.split("/") if segment]
        return "/".join(segments)

    def group(self, series_by_tag: Dict[str, ScalarSeries]) -> List[CurveGroup]:
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
    ACRONYMS = {"lr", "gpu", "cpu", "ram", "vram", "shm", "rss", "vms", "uss", "ema", "l1", "l2", "mse", "mae", "rmse", "ssim", "bic", "f1", "gt", "kz", "dem", "iqr"}

    UNIT_SUFFIXES = (("mb_s", "MB/s"), ("gb_s", "GB/s"), ("gb", "GB"), ("mb", "MB"), ("pct", "%"), ("w", "W"), ("c", "°C"))

    WRAP_WIDTH = 52

    def _token(self, token: str) -> str:
        match = re.fullmatch(r"([a-z]+)(\d*)", token)
        if match and match.group(1) in self.ACRONYMS:
            return match.group(1).upper() + match.group(2)
        return token.capitalize()

    def _split_unit(self, segment: str) -> Tuple[List[str], Optional[str]]:
        tokens = segment.split("_")
        for suffix, unit in self.UNIT_SUFFIXES:
            suffix_tokens = suffix.split("_")
            if len(tokens) > len(suffix_tokens) and tokens[-len(suffix_tokens):] == suffix_tokens:
                return tokens[:-len(suffix_tokens)], unit
        return tokens, None

    def _segment(self, segment: str, with_unit: bool) -> str:
        tokens, unit = self._split_unit(segment) if with_unit else (segment.split("_"), None)
        text         = " ".join(self._token(token) for token in tokens if token)
        return f"{text} ({unit})" if unit else text

    def y_label(self, title: str) -> str:
        return self._segment(title.split("/")[-1], with_unit=True)

    def title(self, title: str) -> str:
        segments = title.split("/")
        rendered = [self._segment(segment, with_unit=(index == len(segments) - 1)) for index, segment in enumerate(segments)]
        return "\n".join(textwrap.wrap(" / ".join(rendered), self.WRAP_WIDTH))


class ScalarCurvePlots(PlotBase):
    save_dpi = 300

    SERIES_COLORS = {"Training": "#1f77b4", "Validation": "#ff7f0e", None: "#1f77b4"}
    SPARSE_POINTS = 40
    LOG_SPAN      = 1e3

    def __init__(self) -> None:
        self.labeler = CurveLabeler()

    def _log_scaled(self, value_arrays) -> bool:
        values = np.concatenate(list(value_arrays))
        values = values[np.isfinite(values)]

        if values.size == 0 or values.min() <= 0:
            return False
        return bool(values.max() / values.min() >= self.LOG_SPAN)

    def _decorate(self, fig, ax, title: str, value_arrays, n_series: int) -> None:
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
        self._apply_style()

        fig, ax = plt.subplots(figsize=(6.2, 3.9))

        for label, steps, values in group.series:
            marker = "o" if steps.size <= self.SPARSE_POINTS else None
            ax.plot(steps, values, lw=1.4, color=self.SERIES_COLORS[label], label=label, marker=marker, markersize=3.0)

        self._decorate(fig, ax, group.title, [values for _label, _steps, values in group.series], len(group.series))
        return self._save(fig, path)


@dataclass
class SeedCurveGroup:
    stem   : str
    title  : str
    series : List[Tuple[str, Optional[str], np.ndarray, np.ndarray]]


class SeedScalarMerger:
    def merge(self, groups_by_seed: Dict[str, List[CurveGroup]]) -> List[SeedCurveGroup]:
        merged: Dict[str, SeedCurveGroup] = {}

        for seed in sorted(groups_by_seed):
            for group in groups_by_seed[seed]:
                entry = merged.setdefault(group.stem, SeedCurveGroup(stem=group.stem, title=group.title, series=[]))
                for role, steps, values in group.series:
                    entry.series.append((seed, role, steps, values))

        return sorted(merged.values(), key=lambda group: group.stem)


class SeedOverlayPlots(ScalarCurvePlots):
    PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

    ROLE_STYLES = {"Training": "-", "Validation": "--", None: "-"}

    @staticmethod
    def _series_label(seed: str, role: Optional[str]) -> str:
        if role is None:
            return seed
        return f"{seed} {'train' if role == 'Training' else 'val'}"

    def render(self, group: SeedCurveGroup, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=(6.2, 3.9))

        seeds = sorted({seed for seed, _role, _steps, _values in group.series})
        for seed, role, steps, values in group.series:
            color = self.PALETTE[seeds.index(seed) % len(self.PALETTE)]
            ax.plot(steps, values, lw=1.2, color=color, linestyle=self.ROLE_STYLES[role], label=self._series_label(seed, role))

        self._decorate(fig, ax, group.title, [values for _seed, _role, _steps, values in group.series], len(group.series))
        return self._save(fig, path)


class TensorboardExport:
    def __init__(self, config: TensorboardExportConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def _read(self) -> Dict[str, ScalarSeries]:
        self.logger.subsection(f"Reading tensorboard log: {self.config.tensorboard_directory}")
        series = TensorboardScalarReader(self.config).read()
        self.logger.ok(f"Loaded {len(series)} scalar series")
        return series

    def _group(self, series: Dict[str, ScalarSeries]) -> List[CurveGroup]:
        groups = ScalarTagGrouper().group(series)
        shared = sum(1 for group in groups if len(group.series) > 1)
        self.logger.ok(f"Grouped into {len(groups)} figures ({shared} shared train/validation)")
        return groups

    def _render(self, groups: List[CurveGroup]) -> List[Path]:
        renderer = ScalarCurvePlots()
        return [renderer.render(group, self.config.plots_directory / f"{group.stem}.png") for group in groups]

    def run(self) -> dict:
        self.logger.section(f"Tensorboard plot export: {Path(self.config.run_directory).name}")

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
    SEED_DIR_PATTERN = re.compile(r"seed\d+")

    @classmethod
    def trials(cls, run_dirs: list[Path]) -> Dict[Path, list[Path]]:
        groups: Dict[Path, list[Path]] = {}

        for run_dir in run_dirs:
            if cls.SEED_DIR_PATTERN.fullmatch(run_dir.name):
                groups.setdefault(run_dir.parent, []).append(run_dir)

        return {trial_dir: sorted(seed_dirs) for trial_dir, seed_dirs in sorted(groups.items()) if len(seed_dirs) > 1}


class SeedOverlayExport:
    def __init__(self, trial_dir: Path, seed_dirs: list[Path], entry_config: TensorboardExportEntryConfig, logger) -> None:
        self.trial_dir    = Path(trial_dir)
        self.seed_dirs    = seed_dirs
        self.entry_config = entry_config
        self.logger       = logger

    def _read_seed(self, seed_dir: Path) -> List[CurveGroup]:
        config = self.entry_config.to_config(seed_dir)
        return ScalarTagGrouper().group(TensorboardScalarReader(config).read())

    def run(self) -> dict:
        self.logger.subsection(f"Seed overlay export: {self.trial_dir.name} ({len(self.seed_dirs)} seeds)")

        groups_by_seed = {seed_dir.name: self._read_seed(seed_dir) for seed_dir in self.seed_dirs}
        merged         = SeedScalarMerger().merge(groups_by_seed)

        plots_directory = self.trial_dir / self.entry_config.output_subdir
        renderer        = SeedOverlayPlots()
        paths           = [renderer.render(group, plots_directory / f"{group.stem}.png") for group in merged]

        self.logger.ok(f"{self.trial_dir.name}: {len(paths)} seed-overlay plots -> {plots_directory}")

        return {
            "trial_directory" : str(self.trial_dir),
            "plots_directory" : str(plots_directory),
            "n_seeds"         : len(self.seed_dirs),
            "n_plots"         : len(paths),
            "plot_paths"      : paths,
        }


class TensorboardExportBatch:
    def __init__(self, entry_config: TensorboardExportEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger

    def _select_runs(self) -> list[Path]:
        selector = TensorboardRunSelector(self.entry_config.runs_dir, self.entry_config.tensorboard_dirname, self.logger)
        return selector.resolve(self.entry_config.run_filter)

    def _export_run(self, run_dir: Path) -> dict:
        config = self.entry_config.to_config(run_dir)
        result = TensorboardExport(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['n_plots']} plots -> {config.plots_directory}")
        return result

    def _export_seed_overlays(self, run_dirs: list[Path]) -> list[dict]:
        return [SeedOverlayExport(trial_dir, seed_dirs, self.entry_config, self.logger).run() for trial_dir, seed_dirs in SeedRunGrouper.trials(run_dirs).items()]

    def run(self) -> list[dict]:
        run_dirs = self._select_runs()
        results  = [self._export_run(run_dir) for run_dir in run_dirs]

        return results + self._export_seed_overlays(run_dirs)
