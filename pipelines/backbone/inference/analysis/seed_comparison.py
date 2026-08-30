"""Aggregation of seed replicas of one training into a mean +/- std report.

Each seed keeps its own finished inference; this module resolves those inference
directories, reduces their scalar metrics across seeds, and optionally attaches
per-pixel seed-disagreement maps with risk-coverage and label-suspect analysis.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipelines.backbone.inference.analysis.seed_disagreement import SeedDisagreementMaps
from pipelines.backbone.inference.report                     import Report
from pipelines.shared.inference.run_classifier               import RunDirectoryWalk
from tools.data.io                                           import FileIO
from tools.metrics.scoring                                   import SeedAggregation
from tools.monitoring.logger                                 import Logger
from tools.reporting.markdown                                import MarkdownTable, ScalarFormatter
from tools.runtime.run_tag                                   import RunTag


class SeedInferenceResolver:
    """Resolves which finished inference directory of a seed run to aggregate.

    Attributes:
        inference_subdir : Explicit inference subdirectory name, empty to take the latest.
        metrics_filename : Metrics file whose presence marks an inference as finished.
    """

    def __init__(self, inference_subdir: str, metrics_filename: str) -> None:
        """Stores the inference selection rule.

        Args:
            inference_subdir : Explicit inference subdirectory name, or empty to pick the latest finished inference per run.
            metrics_filename : Metrics filename marking a finished inference.
        """
        self.inference_subdir = inference_subdir
        self.metrics_filename = metrics_filename

    def _explicit(self, inference_root: Path) -> Path:
        """Returns the named inference subdirectory, requiring it to hold metrics.

        Args:
            inference_root : The run's inference directory.

        Returns:
            The requested inference directory.

        Raises:
            FileNotFoundError : If it holds no metrics file.
        """
        candidate = inference_root / self.inference_subdir
        if not (candidate / self.metrics_filename).is_file():
            raise FileNotFoundError(f"No {self.metrics_filename} under {candidate}; the requested inference_subdir does not hold a finished inference for this run")

        return candidate

    def _latest(self, inference_root: Path) -> Path:
        """Returns the last finished inference directory by name order.

        Args:
            inference_root : The run's inference directory.

        Returns:
            The latest inference directory holding a metrics file.

        Raises:
            FileNotFoundError : If the run has no finished inference.
        """
        candidates = sorted(d for d in inference_root.iterdir() if d.is_dir() and (d / self.metrics_filename).is_file()) if inference_root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"No inference with {self.metrics_filename} under {inference_root}; infer this run before comparing its seeds")

        return candidates[-1]

    def resolve(self, run_dir: Path) -> Path:
        """Returns the inference directory of one seed run to aggregate.

        Args:
            run_dir : Directory of the seed training run.

        Returns:
            The selected inference directory.
        """
        inference_root = Path(run_dir) / "inference"

        if self.inference_subdir:
            return self._explicit(inference_root)

        return self._latest(inference_root)


class SeedComparisonReport:
    """Reduces the metrics of one seed group to mean +/- std and writes the report.

    Attributes:
        group_dir        : Directory holding the seed runs.
        run_dirs         : Seed run directories in aggregation order.
        inference_dirs   : Inference directory chosen for each seed run.
        metrics_filename : Metrics filename read per seed and written aggregated.
        report_filename  : Filename of each seed's own report and of this report.
        logger           : Logger for the written-artifact lines.
        disagreement     : Seed-disagreement payload, or None when not computed.
        output_dir       : Group-level inference directory receiving the aggregate.
        seeds            : Seed labels, relative to the group directory.
    """

    def __init__(self, group_dir: Path, run_dirs: list[Path], inference_dirs: list[Path], output_subdir: str, metrics_filename: str, report_filename: str, logger: Logger, disagreement: dict | None = None) -> None:
        """Binds the seed group and validates that at least two seeds are present.

        Args:
            group_dir        : Directory holding the seed runs.
            run_dirs         : Seed run directories.
            inference_dirs   : Inference directory chosen for each seed run.
            output_subdir    : Name of the group-level output directory.
            metrics_filename : Metrics filename read per seed.
            report_filename  : Report filename of each seed and of this aggregate.
            logger           : Logger for the written-artifact lines.
            disagreement     : Payload from the seed-disagreement analysis, or None.

        Raises:
            ValueError : If fewer than two seed runs are given.
        """
        if len(run_dirs) < 2:
            raise ValueError(f"A seed comparison needs at least two seed runs inside {group_dir}, got {len(run_dirs)}: {[str(d) for d in run_dirs]}")

        self.group_dir        = Path(group_dir)
        self.run_dirs         = [Path(d) for d in run_dirs]
        self.inference_dirs   = [Path(d) for d in inference_dirs]
        self.metrics_filename = metrics_filename
        self.report_filename  = report_filename
        self.logger           = logger
        self.disagreement     = disagreement

        self.output_dir = self.group_dir / "inference" / output_subdir
        self.seeds      = [str(d.relative_to(self.group_dir)) for d in self.run_dirs]

    def _load_metrics(self) -> list[dict]:
        """Returns each seed's metrics dictionary, in seed order."""
        return [FileIO.load_json(inference_dir / self.metrics_filename) for inference_dir in self.inference_dirs]

    def _aggregate(self, per_seed: list[dict]) -> tuple[dict, dict]:
        """Returns the across-seed means and sample standard deviations of every metric key."""
        keys = sorted({key for metrics in per_seed for key in metrics})
        return SeedAggregation.aggregate(per_seed, keys)

    def _write_metrics(self, means: dict, stds: dict) -> Path:
        """Writes the aggregated metrics JSON for the seed group.

        Args:
            means : Across-seed mean per metric key.
            stds  : Across-seed sample standard deviation per metric key.

        Returns:
            The written JSON path.
        """
        payload = {
            "n_seeds"        : len(self.run_dirs),
            "seed_runs"      : {seed: str(run_dir) for seed, run_dir in zip(self.seeds, self.run_dirs)},
            "seed_inference" : {seed: inference_dir.name for seed, inference_dir in zip(self.seeds, self.inference_dirs)},
            "mean"           : means,
            "std"            : stds,
        }

        if self.disagreement is not None:
            payload["disagreement"] = self.disagreement["summary"]

        path = self.output_dir / self.metrics_filename
        FileIO.save_json(payload, path)

        return path

    @staticmethod
    def _fmt(value) -> str:
        """Returns a scalar rendered for the markdown tables."""
        return ScalarFormatter.format_scalar(value, precision=6, adaptive=True)

    def _mean_std_cell(self, mean: float, std: float | None) -> str:
        """Returns the "mean ± std" table cell, or the bare mean when no std exists."""
        if std is None:
            return self._fmt(mean)

        return f"{self._fmt(mean)} ± {self._fmt(std)}"

    def _build_seed_runs_section(self) -> list[str]:
        """Returns the markdown lines listing each seed run and linking its own report."""
        out = ["\n## 1. Seed runs\n"]

        table = MarkdownTable(("Seed", "Run directory", "Inference", "Report"))
        for seed, run_dir, inference_dir in zip(self.seeds, self.run_dirs, self.inference_dirs):
            report_path = inference_dir / self.report_filename
            table.add_row(f"`{seed}`", str(run_dir), f"`{inference_dir.name}`", f"[{self.report_filename}]({os.path.relpath(report_path, self.output_dir)})")

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _grouped_keys(self, means: dict) -> list[tuple[str, list[str]]]:
        """Returns the metric keys grouped into report subsections, with numbering stripped."""
        return [(title.split(" ", 1)[1], keys) for title, keys in Report.group_metric_keys(means)]

    def _build_metrics_section(self, per_seed: list[dict], means: dict, stds: dict) -> list[str]:
        """Returns the markdown lines of the grouped aggregated-metrics tables.

        Args:
            per_seed : Each seed's raw metrics dictionary, in seed order.
            means    : Across-seed mean per metric key.
            stds     : Across-seed sample standard deviation per metric key.

        Returns:
            The markdown lines of the section.
        """
        out = ["\n## 2. Aggregated metrics\n"]
        out.append(
            f"Every scalar metric aggregated over the {len(self.seeds)} seed runs: the cell is the across-seed mean, "
            "the ± term the across-seed sample standard deviation (ddof = 1). Per-seed columns show each run's raw value.\n"
        )

        for index, (title, keys) in enumerate(self._grouped_keys(means), start=1):
            out.append(f"\n### 2.{index} {title}\n")

            table = MarkdownTable(("Metric", "Mean ± seed std", *self.seeds))
            for key in keys:
                per_seed_cells = [self._fmt(metrics.get(key)) if metrics.get(key) is not None else None for metrics in per_seed]
                table.add_row(f"`{key}`", self._mean_std_cell(means[key], stds.get(key)), *per_seed_cells)

            out.append("\n".join(table.render()))
            out.append("")

        return out

    def _build_disagreement_section(self) -> list[str]:
        """Returns the markdown lines summarising the seed-disagreement maps and figures."""
        out = ["\n## 3. Seed disagreement maps\n"]
        out.append(
            "Per-pixel across-seed standard deviation of the model outputs, written as cubes into every seed's inference cube directory "
            "(they appear as metric-map layers in the cube explorer). Predicted slots are first matched to the shared GT slots per seed, "
            "so the parameter maps compare the same physical component across seeds; a slot contributes where it is matched in at "
            f"least {SeedDisagreementMaps.MIN_ACTIVE} seeds.\n"
        )

        summary = self.disagreement["summary"]
        table   = MarkdownTable(("Map", "Mean", "Median", "p95"))
        for name in SeedDisagreementMaps.MAP_TITLES:
            cells = [summary.get(f"{name}_{stat}") for stat in ("mean", "median", "p95")]
            table.add_row(f"`{name}`", *[self._fmt(cell) if cell is not None else "—" for cell in cells])

        out.append("\n".join(table.render()))
        out.append("")

        aurc = summary.get("risk_coverage_aurc")
        if aurc is not None:
            out.append(
                f"Selective prediction under seed-disagreement confidence: AURC {self._fmt(aurc)}, "
                f"risk at 50% coverage {self._fmt(summary['risk_coverage_risk_at_half_coverage'])} vs "
                f"{self._fmt(summary['risk_coverage_full_coverage_risk'])} at full coverage; "
                f"disagreement-error rank correlation {self._fmt(summary['risk_coverage_disagreement_error_spearman'])}."
            )
            out.append("")

        for name, figure_path in self.disagreement["figures"].items():
            out.append(f"![{name}]({os.path.relpath(figure_path, self.output_dir)})")
            out.append("")

        return out

    def _write_report(self, per_seed: list[dict], means: dict, stds: dict) -> Path:
        """Assembles and writes the markdown seed-comparison report.

        Args:
            per_seed : Each seed's raw metrics dictionary, in seed order.
            means    : Across-seed mean per metric key.
            stds     : Across-seed sample standard deviation per metric key.

        Returns:
            The written report path.
        """
        lines = ["# ML Template Seed-Comparison Report", ""]
        lines.append(f"_Generated on {RunTag.timestamp()}_")
        lines.append("")
        lines.append(f"Aggregate over {len(self.seeds)} seed replicas of one training, built from each run's existing inference (no inference re-run). Each seed keeps its own full report (linked below); this report reduces their metrics to mean ± std.")

        lines += self._build_seed_runs_section()
        lines += self._build_metrics_section(per_seed, means, stds)

        if self.disagreement is not None:
            lines += self._build_disagreement_section()

        path = self.output_dir / self.report_filename
        path.write_text("\n".join(lines), encoding="utf-8")

        return path

    def run(self) -> Path:
        """Aggregates the seed metrics and writes the metrics JSON and report.

        Returns:
            The written report path.
        """
        FileIO.ensure_dirs(self.output_dir)

        per_seed     = self._load_metrics()
        means, stds  = self._aggregate(per_seed)
        metrics_path = self._write_metrics(means, stds)
        report_path  = self._write_report(per_seed, means, stds)

        self.logger.info(f"Seed-comparison metrics : {metrics_path}")
        self.logger.info(f"Seed-comparison report  : {report_path}")

        return report_path


class SeedComparison:
    """Entry-level driver comparing the seed replicas of one or more run groups.

    Attributes:
        config   : Seed-comparison configuration.
        runs_dir : Root directory holding the run groups.
    """

    def __init__(self, config) -> None:
        """Stores the configuration and resolves the runs root.

        Args:
            config : Seed-comparison configuration carrying group tags, inference selection, filenames and the disagreement toggle.
        """
        self.config   = config
        self.runs_dir = Path(config.runs_dir)

    def _group_dirs(self) -> list[Path]:
        """Returns the group directories to compare.

        Returns:
            One directory per configured group tag, or the runs root itself when
            no tags are configured.

        Raises:
            FileNotFoundError : If a configured group tag has no directory.
        """
        if self.config.group_tags:
            missing = [tag for tag in self.config.group_tags if not (self.runs_dir / tag).is_dir()]
            if missing:
                raise FileNotFoundError(f"No group directory named {missing} found under {self.runs_dir}")
            return [self.runs_dir / tag for tag in self.config.group_tags]

        return [self.runs_dir]

    def _output_subdir(self) -> str:
        """Returns the group-level output directory name, defaulting to a fresh timestamp."""
        return self.config.output_subdir or self.config.inference_subdir or RunTag.now()

    def _compare_group(self, group_dir: Path, resolver: SeedInferenceResolver, output_subdir: str, logger: Logger) -> Path:
        """Aggregates one group of seed runs, optionally with disagreement maps.

        Args:
            group_dir     : Directory holding the seed runs of one training.
            resolver      : Resolver picking each seed's inference directory.
            output_subdir : Name of the group-level output directory.
            logger        : Logger for progress reporting.

        Returns:
            The written seed-comparison report path.

        Raises:
            ValueError : If the group holds fewer than two seed runs.
        """
        run_dirs = sorted(RunDirectoryWalk.walk(group_dir))
        if len(run_dirs) < 2:
            raise ValueError(f"A seed comparison needs at least two seed runs inside {group_dir}, got {len(run_dirs)}: {[str(d) for d in run_dirs]}")

        inference_dirs = [resolver.resolve(run_dir) for run_dir in run_dirs]

        disagreement = None
        if self.config.compute_disagreement:
            disagreement = SeedDisagreementMaps(
                group_dir        = group_dir,
                run_dirs         = run_dirs,
                inference_dirs   = inference_dirs,
                cubes_subdir     = self.config.cubes_subdir,
                metrics_filename = self.config.metrics_filename,
                output_dir       = group_dir / "inference" / output_subdir,
                logger           = logger,
                active_amp_thr   = self.config.active_amp_thr,
            ).run()

        return SeedComparisonReport(
            group_dir        = group_dir,
            run_dirs         = run_dirs,
            inference_dirs   = inference_dirs,
            output_subdir    = output_subdir,
            metrics_filename = self.config.metrics_filename,
            report_filename  = self.config.report_filename,
            logger           = logger,
            disagreement     = disagreement,
        ).run()

    def run(self) -> list[Path]:
        """Compares the seeds of every configured group.

        Returns:
            The written report path of each group, in group order.
        """
        with Logger(log_dir="logs", name="compare_seeds") as logger:
            logger.section("Seed comparison")

            group_dirs    = self._group_dirs()
            output_subdir = self._output_subdir()
            resolver      = SeedInferenceResolver(self.config.inference_subdir, self.config.metrics_filename)

            logger.kv_table({
                "Runs dir"     : str(self.runs_dir),
                "Groups"       : len(group_dirs),
                "Inference"    : self.config.inference_subdir or "latest per run",
                "Output"       : output_subdir,
                "Disagreement" : "maps + figures" if self.config.compute_disagreement else "disabled",
            }, title="Configuration")

            reports = []
            for group_dir in group_dirs:
                logger.subsection(f"Group: {group_dir}")
                reports.append(self._compare_group(group_dir, resolver, output_subdir, logger))

        return reports
