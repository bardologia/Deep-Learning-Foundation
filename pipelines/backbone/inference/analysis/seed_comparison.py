"""Aggregation of seed replicas of one training into a mean +/- std report.

Each seed keeps its own completed inference; this module resolves those inference
stamps for one data split, reduces their scalar metrics across seeds, and
optionally attaches per-pixel seed-disagreement maps with risk-coverage and
label-suspect analysis.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from pipelines.backbone.inference.analysis.seed_disagreement import SeedDisagreementMaps
from pipelines.backbone.inference.report                     import Report
from pipelines.shared.inference.metadata                     import CompletedInference
from pipelines.shared.inference.run_classifier               import RunDirectoryWalk
from tools.data.io                                           import FileIO
from pipelines.backbone.inference.metrics                    import Metrics
from tools.metrics.scoring                                   import SeedAggregation
from tools.monitoring.logger                                 import Logger
from tools.reporting.markdown                                import MarkdownTable, ScalarFormatter
from tools.runtime.completion                                import CompletionMarker
from tools.runtime.run_tag                                   import RunTag


class SeedInferenceResolver:
    """Resolves which completed inference stamp of a seed run to aggregate.

    Only stamps carrying a completion marker for the configured split qualify, so a
    crashed, still running or other-split inference never enters an aggregate.

    Attributes:
        inference_subdir : Explicit stamp name, empty to take the newest completed stamp of the split.
        split            : Data split the chosen stamp must have been scored on.
        metrics_filename : Metrics file every chosen stamp must hold.
        logger           : Logger receiving the warnings about unfinished stamps.
        unfinished       : Stamp names holding metrics but no completion marker, keyed by run directory.
    """

    def __init__(self, inference_subdir: str, split: str, metrics_filename: str, logger: Logger) -> None:
        """Stores the inference selection rule.

        Args:
            inference_subdir : Explicit stamp name, or empty to pick each run's newest completed stamp of the split.
            split            : Data split the stamp must record in its completion marker.
            metrics_filename : Metrics filename the chosen stamp must hold.
            logger           : Logger for the warnings about unfinished stamps.
        """
        self.inference_subdir = inference_subdir
        self.split            = split
        self.metrics_filename = metrics_filename
        self.logger           = logger

        self.unfinished: dict[Path, list[str]] = {}

    def _require_metrics(self, stamp: Path) -> None:
        """Raises FileNotFoundError when a completed stamp holds no metrics file."""
        if not (stamp / self.metrics_filename).is_file():
            raise FileNotFoundError(f"No {self.metrics_filename} under {stamp}; the completed inference wrote no metrics, re-run it before comparing seeds")

    def _explicit(self, inference_root: Path) -> Path:
        """Returns the named stamp, requiring its completion marker, split and metrics.

        Args:
            inference_root : The run's inference directory.

        Returns:
            The requested inference stamp.

        Raises:
            FileNotFoundError : If the stamp carries no completion marker or no metrics file.
            ValueError        : If the stamp was scored on another split than the configured one.
        """
        candidate = inference_root / self.inference_subdir

        if not CompletionMarker.is_complete(candidate):
            raise FileNotFoundError(f"{candidate} carries no {CompletionMarker.FILENAME}; the requested inference_subdir is unfinished, or predates completion markers and needs scripts/backfill_completion_markers.py")

        if candidate not in CompletedInference.stamps(inference_root, self.split):
            recorded = CompletionMarker.payload(candidate)["split"]
            raise ValueError(f"{candidate} was scored on split '{recorded}', not on the configured split '{self.split}'; point inference_subdir at a '{self.split}' stamp or change split")

        self._require_metrics(candidate)
        return candidate

    def _note_unfinished(self, inference_root: Path) -> None:
        """Warns about stamps that hold metrics without a completion marker and records them.

        Args:
            inference_root : The run's inference directory.
        """
        unfinished = sorted(stamp.name for stamp in inference_root.iterdir() if stamp.is_dir() and (stamp / self.metrics_filename).is_file() and not CompletionMarker.is_complete(stamp))
        if not unfinished:
            return

        self.unfinished[inference_root.parent] = unfinished
        self.logger.warning(f"{inference_root.parent.name}: {len(unfinished)} stamp(s) hold {self.metrics_filename} without a completion marker and were skipped as unfinished: {', '.join(unfinished)}")

    def _latest(self, inference_root: Path) -> Path:
        """Returns the newest completed stamp of the split, skipping unfinished stamps.

        Args:
            inference_root : The run's inference directory.

        Returns:
            The newest stamp carrying a completion marker for the split.

        Raises:
            FileNotFoundError : If the run has no completed inference on the split.
        """
        latest = CompletedInference.latest(inference_root, self.split)
        if latest is None:
            raise FileNotFoundError(f"No completed '{self.split}' inference under {inference_root}; infer this run on that split before comparing its seeds")

        self._require_metrics(latest)
        self._note_unfinished(inference_root)

        return latest

    def resolve(self, run_dir: Path) -> Path:
        """Returns the inference stamp of one seed run to aggregate.

        Args:
            run_dir : Directory of the seed training run.

        Returns:
            The selected inference stamp.
        """
        inference_root = Path(run_dir) / "inference"

        if self.inference_subdir:
            return self._explicit(inference_root)

        return self._latest(inference_root)

    @staticmethod
    def record(inference_dir: Path) -> dict:
        """Returns the stamp name, split and completion time a resolved stamp recorded.

        Args:
            inference_dir : A resolved inference stamp carrying a completion marker.

        Returns:
            Mapping with "stamp", "split" and "completed_at".
        """
        payload = CompletionMarker.payload(inference_dir)
        return {"stamp": Path(inference_dir).name, "split": payload["split"], "completed_at": payload["completed_at"]}


class SeedActivityThreshold:
    """Resolves the one activity threshold every seed inference of a group was scored with.

    Active-slot metrics and matched slots depend on that threshold, so seeds scored
    at different values, or before the value was recorded, are refused on every
    comparison path rather than averaged.

    Attributes:
        metrics_filename : Metrics file read from each seed's inference directory.
    """

    def __init__(self, metrics_filename: str) -> None:
        """Stores the metrics filename the recorded threshold is read from."""
        self.metrics_filename = metrics_filename

    def resolve(self, seeds: list[str], inference_dirs: list[Path]) -> float:
        """Returns the activity threshold shared by every seed's inference.

        Args:
            seeds          : Seed labels aligned with inference_dirs.
            inference_dirs : Resolved inference directory of each seed.

        Returns:
            The common activity threshold the seeds were scored with.

        Raises:
            KeyError   : If a seed's metrics file carries no activity_threshold, which inference written before the field existed lacks.
            ValueError : If the seeds were scored with different activity thresholds, which makes their active-slot metrics and matched slots incomparable.
        """
        thresholds = {}
        for seed, inference_dir in zip(seeds, inference_dirs):
            metrics = FileIO.load_json(Path(inference_dir) / self.metrics_filename)
            if "activity_threshold" not in metrics:
                raise KeyError(f"{Path(inference_dir) / self.metrics_filename} has no activity_threshold; re-run inference for seed {seed}")

            thresholds[seed] = float(metrics["activity_threshold"])

        if len(set(thresholds.values())) != 1:
            raise ValueError(f"The seeds were scored with different activity thresholds {thresholds}; re-run their inference against one label run before comparing them")

        return next(iter(thresholds.values()))


class SeedEnsembleEvaluation:
    """Scores the across-seed mean prediction cube against the ground-truth curves.

    Deep-ensemble evaluation on artifacts already on disk: the per-seed
    prediction cubes are averaged and scored with the same cube-wide scalar
    metrics inference reports per seed, so the ensemble gain reads directly
    against the per-seed mean without re-running any model.

    Attributes:
        PRED_CUBE : Filename of the per-seed prediction cube.
        GT_CUBE   : Filename of the ground-truth curve cube.
        cube_dirs : Cube directory of each seed, in seed order.
        logger    : Logger for the result line.
    """

    PRED_CUBE = "pred_curves.npy"
    GT_CUBE   = "gt_curves.npy"

    def __init__(self, cube_dirs: list[Path], logger: Logger) -> None:
        """Stores the seed cube directories and the logger.

        Args:
            cube_dirs : Cube directory inside each seed's inference directory.
            logger    : Logger for the result line.
        """
        self.cube_dirs = [Path(d) for d in cube_dirs]
        self.logger    = logger

    def _open(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Memory-maps every seed's prediction cube and loads the ground truth.

        Returns:
            Tuple of the per-seed prediction cubes and the float32 GT cube.

        Raises:
            FileNotFoundError : If any seed is missing its prediction or GT cube.
            ValueError        : If the cubes disagree on shape across seeds.
        """
        for cube_dir in self.cube_dirs:
            for name in (self.PRED_CUBE, self.GT_CUBE):
                if not (cube_dir / name).is_file():
                    raise FileNotFoundError(f"{cube_dir / name} is missing; re-run inference with save_cubes enabled for every seed, or disable compute_ensemble")

        preds  = [np.load(cube_dir / self.PRED_CUBE, mmap_mode="r") for cube_dir in self.cube_dirs]
        shapes = {pred.shape for pred in preds}
        if len(shapes) != 1:
            raise ValueError(f"Seed prediction cubes disagree on shape: {sorted(shapes)}; the seeds were inferred on different regions and cannot be ensembled")

        gt = np.asarray(np.load(self.cube_dirs[0] / self.GT_CUBE), dtype=np.float32)

        return preds, gt

    def run(self) -> dict:
        """Averages the seed predictions and scores the mean against the ground truth.

        Returns:
            Mapping with "n_seeds" and the ensemble's "curve_mse_gt",
            "curve_mae_gt", "curve_rmse_gt", "overall_r2_gt" and "psnr_db_gt",
            named like the per-seed scalars so the gain reads side by side.
        """
        preds, gt = self._open()

        mean = np.zeros(preds[0].shape, dtype=np.float64)
        for pred in preds:
            mean += pred
        mean = (mean / len(preds)).astype(np.float32)

        diff       = mean - gt
        mse        = float((diff * diff).mean(dtype=np.float64))
        mae        = float(np.abs(diff).mean(dtype=np.float64))
        gt_mean    = float(gt.mean(dtype=np.float64))
        overall_r2 = 1.0 - float((diff * diff).sum(dtype=np.float64)) / (float(((gt - gt_mean) ** 2).sum(dtype=np.float64)) + 1e-12)

        scalars = {
            "n_seeds"       : len(preds),
            "curve_mse_gt"  : mse,
            "curve_mae_gt"  : mae,
            "curve_rmse_gt" : float(np.sqrt(mse)),
            "overall_r2_gt" : overall_r2,
            "psnr_db_gt"    : Metrics._psnr(mean, gt),
        }

        self.logger.subsection(f"Seed ensemble : mse {mse:.4g}, overall R2 {overall_r2:.4f} over {len(preds)} seeds")

        return scalars


class SeedComparisonReport:
    """Reduces the metrics of one seed group to mean +/- std and writes the report.

    Attributes:
        group_dir          : Directory holding the seed runs.
        run_dirs           : Seed run directories in aggregation order.
        inference_dirs     : Inference stamp chosen for each seed run.
        activity_threshold : Activity threshold every seed's inference was scored with.
        metrics_filename   : Metrics filename read per seed and written aggregated.
        report_filename    : Filename of each seed's own report and of this report.
        logger             : Logger for the written-artifact lines.
        disagreement       : Seed-disagreement payload, or None when not computed.
        ensemble           : Seed-ensemble scores, or None when not computed.
        output_dir         : Group-level inference directory receiving the aggregate.
        seeds              : Seed labels, relative to the group directory.
        records            : Stamp name, split and completion time recorded per seed.
        split              : Data split every seed's stamp was scored on.
        MAX_LISTED         : Number of metric keys named in the shortfall warning before it is cut.
    """

    MAX_LISTED = 8

    def __init__(self, group_dir: Path, run_dirs: list[Path], inference_dirs: list[Path], activity_threshold: float, output_subdir: str, metrics_filename: str, report_filename: str, logger: Logger, disagreement: dict | None = None, ensemble: dict | None = None) -> None:
        """Binds the seed group and validates that at least two seeds are present.

        Args:
            group_dir          : Directory holding the seed runs.
            run_dirs           : Seed run directories.
            inference_dirs     : Inference directory chosen for each seed run.
            activity_threshold : Activity threshold every seed's inference was scored with, as resolved by SeedActivityThreshold.
            output_subdir      : Name of the group-level output directory.
            metrics_filename   : Metrics filename read per seed.
            report_filename    : Report filename of each seed and of this aggregate.
            logger             : Logger for the written-artifact lines.
            disagreement       : Payload from the seed-disagreement analysis, or None.
            ensemble           : Scores of the seed-ensemble evaluation, or None.

        Raises:
            ValueError : If fewer than two seed runs are given, or if their stamps were scored on different splits.
        """
        if len(run_dirs) < 2:
            raise ValueError(f"A seed comparison needs at least two seed runs inside {group_dir}, got {len(run_dirs)}: {[str(d) for d in run_dirs]}")

        self.group_dir          = Path(group_dir)
        self.run_dirs           = [Path(d) for d in run_dirs]
        self.inference_dirs     = [Path(d) for d in inference_dirs]
        self.activity_threshold = float(activity_threshold)
        self.metrics_filename   = metrics_filename
        self.report_filename    = report_filename
        self.logger             = logger
        self.disagreement       = disagreement
        self.ensemble           = ensemble

        self.output_dir = self.group_dir / "inference" / output_subdir
        self.seeds      = [str(d.relative_to(self.group_dir)) for d in self.run_dirs]
        self.records    = [SeedInferenceResolver.record(inference_dir) for inference_dir in self.inference_dirs]
        self.split      = self._common_split()

    def _common_split(self) -> str:
        """Returns the split every seed's stamp recorded.

        Raises:
            ValueError : If the seeds resolved to stamps of different splits.
        """
        splits = {seed: record["split"] for seed, record in zip(self.seeds, self.records)}
        if len(set(splits.values())) != 1:
            raise ValueError(f"The seeds resolved to inferences of different splits {splits}; aggregate one split at a time")

        return next(iter(splits.values()))

    def _load_metrics(self) -> list[dict]:
        """Returns each seed's metrics dictionary, in seed order."""
        return [FileIO.load_json(inference_dir / self.metrics_filename) for inference_dir in self.inference_dirs]

    def _listed(self, items: list[str]) -> str:
        """Returns the items joined for a log line, cut after MAX_LISTED with the remainder counted."""
        shown = ", ".join(items[:self.MAX_LISTED])
        if len(items) > self.MAX_LISTED:
            shown += f", and {len(items) - self.MAX_LISTED} more"

        return shown

    def _aggregate(self, per_seed: list[dict]) -> tuple[dict, dict, dict, dict]:
        """Returns the across-seed means, sample standard deviations, 95% confidence intervals and finite-sample counts of every metric key.

        A key some seeds report without a finite value is aggregated over the seeds
        that do, and one warning names those keys with the seeds that dropped out. A
        numeric key that is non-finite in every seed is dropped with its own warning.

        Args:
            per_seed : Each seed's raw metrics dictionary, in seed order.

        Returns:
            Tuple of the mean, standard-deviation, interval and count mappings; the
            count is the number of seeds whose finite value entered the mean.
        """
        keys        = sorted({key for metrics in per_seed for key in metrics})
        means, stds = SeedAggregation.aggregate(per_seed, keys)
        cis, counts = SeedAggregation.intervals(per_seed, keys)
        shortfall   = SeedAggregation.shortfall(per_seed, list(means), self.seeds)

        if shortfall:
            self.logger.warning(f"{len(shortfall)} metric(s) aggregated over fewer than {len(self.seeds)} seeds: {self._listed([f'{key} (missing {chr(44).join(missing)})' for key, missing in shortfall.items()])}")

        dropped = [key for key in keys if key not in means and any(key in metrics and isinstance(metrics[key], float) and not math.isfinite(metrics[key]) for metrics in per_seed)]
        if dropped:
            self.logger.warning(f"{len(dropped)} metric(s) dropped from the aggregate, non-finite in every seed that reports them: {self._listed(dropped)}")

        return means, stds, cis, counts

    def _write_metrics(self, means: dict, stds: dict, cis: dict, counts: dict) -> Path:
        """Writes the aggregated metrics JSON for the seed group.

        Args:
            means  : Across-seed mean per metric key.
            stds   : Across-seed sample standard deviation per metric key.
            cis    : 95% t-based confidence interval of the mean per metric key.
            counts : Number of seeds reporting a finite value per metric key.

        Returns:
            The written JSON path.
        """
        payload = {
            "n_seeds"            : len(self.run_dirs),
            "split"              : self.split,
            "activity_threshold" : self.activity_threshold,
            "seed_runs"          : {seed: str(run_dir) for seed, run_dir in zip(self.seeds, self.run_dirs)},
            "seed_inference"     : {seed: record for seed, record in zip(self.seeds, self.records)},
            "mean"               : means,
            "std"                : stds,
            "ci95"               : {key: list(interval) for key, interval in cis.items()},
            "n"                  : counts,
        }

        if self.disagreement is not None:
            payload["disagreement"] = self.disagreement["summary"]

        if self.ensemble is not None:
            payload["ensemble"] = self.ensemble

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

        table = MarkdownTable(("Seed", "Run directory", "Inference", "Split", "Completed", "Report"))
        for seed, run_dir, inference_dir, record in zip(self.seeds, self.run_dirs, self.inference_dirs, self.records):
            report_path = inference_dir / self.report_filename
            table.add_row(f"`{seed}`", str(run_dir), f"`{record['stamp']}`", record["split"], record["completed_at"], f"[{self.report_filename}]({os.path.relpath(report_path, self.output_dir)})")

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _grouped_keys(self, means: dict) -> list[tuple[str, list[str]]]:
        """Returns the metric keys grouped into report subsections, with numbering stripped."""
        return [(title.split(" ", 1)[1], keys) for title, keys in Report.group_metric_keys(means)]

    def _ci_cell(self, interval: tuple[float, float] | None) -> str:
        """Returns the "[low, high]" interval cell, or a dash when no interval exists."""
        if interval is None:
            return "—"

        return f"[{self._fmt(interval[0])}, {self._fmt(interval[1])}]"

    def _build_metrics_section(self, per_seed: list[dict], means: dict, stds: dict, cis: dict, counts: dict) -> list[str]:
        """Returns the markdown lines of the grouped aggregated-metrics tables.

        Args:
            per_seed : Each seed's raw metrics dictionary, in seed order.
            means    : Across-seed mean per metric key.
            stds     : Across-seed sample standard deviation per metric key.
            cis      : 95% confidence interval of the mean per metric key.
            counts   : Number of seeds reporting a finite value per metric key.

        Returns:
            The markdown lines of the section.
        """
        out = ["\n## 2. Aggregated metrics\n"]
        out.append(
            f"Every scalar metric aggregated over the {len(self.seeds)} seed runs: the cell is the across-seed mean, "
            "the ± term the across-seed sample standard deviation (ddof = 1), the interval a t-based 95% confidence "
            "interval of the mean, and n the number of seeds reporting a finite value. Per-seed columns show each run's raw value.\n"
        )

        for index, (title, keys) in enumerate(self._grouped_keys(means), start=1):
            out.append(f"\n### 2.{index} {title}\n")

            table = MarkdownTable(("Metric", "Mean ± seed std", "95% CI", "n", *self.seeds))
            for key in keys:
                per_seed_cells = [self._fmt(metrics.get(key)) if metrics.get(key) is not None else None for metrics in per_seed]
                table.add_row(f"`{key}`", self._mean_std_cell(means[key], stds.get(key)), self._ci_cell(cis.get(key)), str(counts.get(key, 0)), *per_seed_cells)

            out.append("\n".join(table.render()))
            out.append("")

        return out

    def _build_ensemble_section(self, means: dict, stds: dict) -> list[str]:
        """Returns the markdown lines comparing the seed-ensemble scalars with the per-seed mean.

        Args:
            means : Across-seed mean per metric key.
            stds  : Across-seed sample standard deviation per metric key.

        Returns:
            The markdown lines of the section.
        """
        out = ["\n## 3. Seed ensemble\n"]
        out.append(
            f"The {self.ensemble['n_seeds']} seed prediction cubes averaged into one deep-ensemble prediction and scored "
            "against the ground truth, next to the mean of the per-seed scores. A lower ensemble error than the per-seed "
            "mean is the usual variance-reduction gain of ensembling.\n"
        )

        table = MarkdownTable(("Metric", "Ensemble", "Per-seed mean ± std"))
        for key, value in self.ensemble.items():
            if key == "n_seeds":
                continue
            counterpart = self._mean_std_cell(means[key], stds.get(key)) if key in means else "n/a"
            table.add_row(f"`{key}`", self._fmt(value), counterpart)

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _build_disagreement_section(self) -> list[str]:
        """Returns the markdown lines summarising the seed-disagreement maps and figures."""
        out = ["\n## 4. Seed disagreement maps\n"]
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

    def _write_report(self, per_seed: list[dict], means: dict, stds: dict, cis: dict, counts: dict) -> Path:
        """Assembles and writes the markdown seed-comparison report.

        Args:
            per_seed : Each seed's raw metrics dictionary, in seed order.
            means    : Across-seed mean per metric key.
            stds     : Across-seed sample standard deviation per metric key.
            cis      : 95% confidence interval of the mean per metric key.
            counts   : Number of seeds reporting a finite value per metric key.

        Returns:
            The written report path.
        """
        lines = ["# ML Template Seed-Comparison Report", ""]
        lines.append(f"_Generated on {RunTag.timestamp()}_")
        lines.append("")
        lines.append(f"Aggregate over {len(self.seeds)} seed replicas of one training, built from each run's completed '{self.split}' inference (no inference re-run), all scored at activity threshold {self._fmt(self.activity_threshold)}. Each seed keeps its own full report (linked below); this report reduces their metrics to mean ± std with a 95% confidence interval.")

        lines += self._build_seed_runs_section()
        lines += self._build_metrics_section(per_seed, means, stds, cis, counts)

        if self.ensemble is not None:
            lines += self._build_ensemble_section(means, stds)

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

        per_seed                 = self._load_metrics()
        means, stds, cis, counts = self._aggregate(per_seed)
        metrics_path             = self._write_metrics(means, stds, cis, counts)
        report_path              = self._write_report(per_seed, means, stds, cis, counts)

        complete = sum(1 for n in counts.values() if n == len(self.seeds))
        self.logger.info(f"{len(means)} metric(s) aggregated over {len(self.seeds)} seeds, {complete} with every seed contributing")
        self.logger.info(f"Seed-comparison metrics : {metrics_path}")
        self.logger.info(f"Seed-comparison report  : {report_path}")

        return report_path


class SeedComparison:
    """Entry-level driver comparing the seed replicas of one or more run groups.

    Attributes:
        config   : Seed-comparison configuration.
        logger   : Logger shared with the caller, so a mailer or console run keeps one record.
        runs_dir : Root directory holding the run groups.
    """

    def __init__(self, config, logger: Logger) -> None:
        """Stores the configuration and the caller's logger and resolves the runs root.

        Args:
            config : Seed-comparison configuration carrying group tags, inference selection, filenames and the disagreement toggle.
            logger : Logger every message of the comparison goes through.
        """
        self.config   = config
        self.logger   = logger
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

    def _resolve_seeds(self, group_dir: Path, run_dirs: list[Path], resolver: SeedInferenceResolver) -> list[Path]:
        """Resolves each seed's inference stamp and logs the chosen stamp, split and completion time per seed.

        Args:
            group_dir : Directory holding the seed runs, the seed labels are relative to it.
            run_dirs  : Seed run directories in aggregation order.
            resolver  : Resolver picking each seed's inference stamp.

        Returns:
            The resolved inference stamps, aligned with run_dirs.
        """
        inference_dirs = [resolver.resolve(run_dir) for run_dir in run_dirs]

        rows = []
        for run_dir, inference_dir in zip(run_dirs, inference_dirs):
            record = SeedInferenceResolver.record(inference_dir)
            rows.append({"Seed": str(run_dir.relative_to(group_dir)), "Inference": record["stamp"], "Split": record["split"], "Completed": record["completed_at"]})

        self.logger.metrics_table(rows, columns=["Seed", "Inference", "Split", "Completed"], title="Seed inferences")

        return inference_dirs

    def _compare_group(self, group_dir: Path, resolver: SeedInferenceResolver, output_subdir: str) -> Path:
        """Aggregates one group of seed runs, optionally with disagreement maps.

        Args:
            group_dir     : Directory holding the seed runs of one training.
            resolver      : Resolver picking each seed's inference directory.
            output_subdir : Name of the group-level output directory.

        Returns:
            The written seed-comparison report path.

        Raises:
            ValueError : If the group holds fewer than two seed runs, or if the seeds were scored at different activity thresholds.
            KeyError   : If a seed's metrics record no activity threshold.
        """
        run_dirs = sorted(RunDirectoryWalk.walk(group_dir))
        if len(run_dirs) < 2:
            raise ValueError(f"A seed comparison needs at least two seed runs inside {group_dir}, got {len(run_dirs)}: {[str(d) for d in run_dirs]}")

        inference_dirs = self._resolve_seeds(group_dir, run_dirs, resolver)
        seeds          = [str(run_dir.relative_to(group_dir)) for run_dir in run_dirs]
        threshold      = SeedActivityThreshold(self.config.metrics_filename).resolve(seeds, inference_dirs)

        self.logger.info(f"Activity threshold {threshold:g} shared by all {len(seeds)} seeds")

        ensemble = None
        if self.config.compute_ensemble:
            ensemble = SeedEnsembleEvaluation([inference_dir / self.config.cubes_subdir for inference_dir in inference_dirs], self.logger).run()

        disagreement = None
        if self.config.compute_disagreement:
            disagreement = SeedDisagreementMaps(
                group_dir        = group_dir,
                run_dirs         = run_dirs,
                inference_dirs   = inference_dirs,
                cubes_subdir     = self.config.cubes_subdir,
                metrics_filename = self.config.metrics_filename,
                output_dir       = group_dir / "inference" / output_subdir,
                active_amp_thr   = threshold,
                logger           = self.logger,
            ).run()

        return SeedComparisonReport(
            group_dir          = group_dir,
            run_dirs           = run_dirs,
            inference_dirs     = inference_dirs,
            activity_threshold = threshold,
            output_subdir      = output_subdir,
            metrics_filename   = self.config.metrics_filename,
            report_filename    = self.config.report_filename,
            logger             = self.logger,
            disagreement       = disagreement,
            ensemble           = ensemble,
        ).run()

    def run(self) -> list[Path]:
        """Compares the seeds of every configured group.

        Returns:
            The written report path of each group, in group order.
        """
        self.logger.section("Seed comparison")

        group_dirs    = self._group_dirs()
        output_subdir = self._output_subdir()
        resolver      = SeedInferenceResolver(self.config.inference_subdir, self.config.split, self.config.metrics_filename, self.logger)

        self.logger.kv_table({
            "Runs dir"     : str(self.runs_dir),
            "Groups"       : len(group_dirs),
            "Split"        : self.config.split,
            "Inference"    : self.config.inference_subdir or f"newest completed '{self.config.split}' stamp per run",
            "Metrics file" : self.config.metrics_filename,
            "Output"       : output_subdir,
            "Disagreement" : f"maps + figures from {self.config.cubes_subdir}/" if self.config.compute_disagreement else "disabled",
        }, title="Configuration")

        reports = []
        for group_dir in group_dirs:
            self.logger.subsection(f"Group: {group_dir}")
            reports.append(self._compare_group(group_dir, resolver, output_subdir))

        if resolver.unfinished:
            skipped = sum(len(names) for names in resolver.unfinished.values())
            self.logger.warning(f"{skipped} unfinished inference stamp(s) skipped across {len(resolver.unfinished)} seed run(s)")

        self.logger.ok(f"{len(reports)} seed group(s) compared on split '{self.config.split}'")

        return reports
