"""Collection of per-trial artifacts from a benchmark or cross-validation tree.

Walks the training directory of a pipeline run, reads each trial's config,
checkpoint, completion marker and latest inference stamp into a `TrialRecord`,
and aggregates seed sweeps into a single record plus dispersion statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib     import Path

import numpy as np
import torch

from models                               import BACKBONE_HEADS
from pipelines.shared.inference.metadata  import CompletedInference
from pipelines.shared.model.model_builder import ModelBuilder
from pipelines.shared.training.seed_sweep import SeedSet
from tools.data.io                        import FileIO
from tools.metrics.scoring                import SeedAggregation
from tools.monitoring.logger              import Logger
from tools.runtime.completion             import CompletionMarker

_TOTAL_PARAMS_PATTERN = re.compile(r"\*\*Total Parameters:\*\*\s*`([\d,]+)`")
_CHECKPOINT_KEYS      = ("best_val_loss", "best_epoch", "epoch")


@dataclass
class TrialRecord:
    """Everything the comparison report knows about one trial.

    Attributes:
        name            : Trial identifier, "<trial>/<seed>" for seed-swept runs.
        run_dir         : Training directory of the trial (or of the single seed).
        parameters      : Trainable parameter count of the model, when known.
        size_match      : Width-scaling record used to equalise capacity across models.
        trainer_config  : Serialized trainer configuration from docs/.
        run_summary     : Run summary written by the trainer under meta/.
        checkpoint      : Best-checkpoint scalars plus the completed epoch count.
        training_result : Status and wall-clock duration of the training job.
        inference_dir   : Latest completed inference stamp, or None.
        metrics         : Test metrics loaded from that stamp.
        figures         : PNG figure paths of that stamp.
        animations      : GIF animation paths of that stamp.
        report_path     : Path of the per-run inference report, when present.
    """

    name            : str
    run_dir         : Path
    parameters      : int | None  = None
    size_match      : dict        = field(default_factory=dict)
    trainer_config  : dict        = field(default_factory=dict)
    run_summary     : dict        = field(default_factory=dict)
    checkpoint      : dict        = field(default_factory=dict)
    training_result : dict        = field(default_factory=dict)
    inference_dir   : Path | None = None
    metrics         : dict        = field(default_factory=dict)
    figures         : list[Path]  = field(default_factory=list)
    animations      : list[Path]  = field(default_factory=list)
    report_path     : Path | None = None

    @property
    def has_inference(self) -> bool:
        """True when a completed inference stamp was attached to this trial."""
        return self.inference_dir is not None


class SeedRunAggregator:
    """Collapses the seed runs of a trial into one record with dispersion stats.

    Attributes:
        CHECKPOINT_KEYS : Checkpoint scalars averaged across seeds.
        logger          : Logger receiving the representative-seed line of every trial and a warning when its means cover fewer seeds than it holds.
        seed_dispersion : Per-trial dispersion payloads, populated by `aggregate`.
    """

    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self, logger: Logger) -> None:
        """Starts with an empty dispersion table.

        Args:
            logger : Logger for the seed-coverage warnings and the representative-seed lines.
        """
        self.logger          = logger
        self.seed_dispersion = {}

    def _group(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        """Groups records by the seed-stripped base name, preserving discovery order."""
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(SeedSet.base(record.name), []).append(record)

        return list(groups.items())

    def _report_coverage(self, name: str, runs: list[TrialRecord], seed_names: list[str], metric_keys: list[str], contributors: dict[str, int], representative: TrialRecord) -> None:
        """Names the seed standing for the trial and warns when any metric mean covers fewer seeds than the trial holds.

        Args:
            name           : Seed-stripped trial name.
            runs           : Seed runs of the trial.
            seed_names     : Seed label of every run, aligned with runs.
            metric_keys    : Metric keys aggregated for the trial.
            contributors   : Number of seeds carrying a finite value, per metric key.
            representative : Seed run whose figures and report stand for the trial.
        """
        shortfall = SeedAggregation.shortfall([run.metrics for run in runs], metric_keys, seed_names)

        without_inference = [seed for seed, run in zip(seed_names, runs) if not run.has_inference]
        dropped_keys      = sorted(key for key, absent in shortfall.items() if set(absent) - set(without_inference))

        if without_inference or dropped_keys:
            parts = []
            if without_inference:
                parts.append(f"seeds without inference: {', '.join(without_inference)}")
            if dropped_keys:
                parts.append(f"values missing or non-finite for {', '.join(dropped_keys)}")

            used     = sorted(set(contributors.values())) or [0]
            coverage = f"{used[0]} of {len(runs)}" if len(used) == 1 else f"{used[0]} to {used[-1]} of {len(runs)}"

            self.logger.warning(f"{name}: {'; '.join(parts)}; metric means cover {coverage} seeds")

        self.logger.info(f"{name}: figures and report from {representative.name}")

    def _aggregate_group(self, name: str, runs: list[TrialRecord]) -> tuple[TrialRecord, dict]:
        """Averages the seed runs of one trial into a single record.

        Args:
            name : Seed-stripped trial name carried by the aggregate record.
            runs : Seed runs belonging to that trial.

        Returns:
            Pair of the aggregated record (metrics and checkpoint scalars are
            seed means, figures and report come from the first seed with an
            inference stamp) and its dispersion payload holding the seed count,
            the per-metric count of seeds that entered each mean, the
            best-val-loss standard deviation, per-metric standard deviations
            and the raw per-seed metric dicts.
        """
        representative = next((run for run in runs if run.has_inference), runs[0])

        seed_names   = [run.name.rsplit("/", 1)[-1] for run in runs]
        metric_dicts = [run.metrics for run in runs]
        metric_keys  = sorted({key for metrics in metric_dicts for key in metrics})

        metric_means, metric_std = SeedAggregation.aggregate(metric_dicts, metric_keys)
        ckpt_means, ckpt_std     = SeedAggregation.aggregate([run.checkpoint for run in runs], list(self.CHECKPOINT_KEYS))
        contributors             = SeedAggregation.contributors(metric_dicts, metric_keys)

        self._report_coverage(name, runs, seed_names, metric_keys, contributors, representative)

        durations = [run.training_result.get("duration_s") for run in runs]
        durations = [value for value in durations if isinstance(value, (int, float))]

        record                 = replace(representative, name=name, metrics=metric_means)
        record.checkpoint      = {**representative.checkpoint, **ckpt_means}
        record.training_result = {
            "status"     : "DONE" if all(run.training_result.get("status") == "DONE" for run in runs) else "PARTIAL",
            "duration_s" : float(np.mean(durations)) if durations else None,
        }

        dispersion = {
            "n_seeds"           : len(runs),
            "n_used"            : contributors,
            "best_val_loss_std" : ckpt_std.get("best_val_loss"),
            "metrics"           : metric_std,
            "seed_metrics"      : dict(zip(seed_names, metric_dicts)),
        }

        return record, dispersion

    def aggregate(self, records: list[TrialRecord]) -> list[TrialRecord]:
        """Replaces every multi-seed group by its aggregate record.

        Args:
            records : Per-run records as collected from disk.

        Returns:
            One record per trial; single-seed trials pass through untouched.
            `seed_dispersion` is refilled for the aggregated trials.
        """
        self.seed_dispersion = {}
        aggregated           = []

        for name, runs in self._group(records):
            if len(runs) == 1:
                aggregated.append(runs[0])
                continue

            record, dispersion = self._aggregate_group(name, runs)
            aggregated.append(record)
            self.seed_dispersion[name] = dispersion

        return aggregated


class TrialArtifactReader:
    """Reads checkpoint scalars and inference artifacts of a single trial.

    Attributes:
        inference_split : Split whose completed inference stamps are attached, or None to skip inference attachment entirely.
    """

    inference_split: str | None

    def _read_checkpoint(self, trial_dir: Path) -> dict:
        """Returns the best-checkpoint scalars and completed epoch count of a trial.

        Args:
            trial_dir : Training directory searched recursively for best_model.pt.

        Returns:
            Dict of the non-null checkpoint scalars plus "n_train_epochs", or an
            empty dict when no checkpoint exists.
        """
        checkpoint_path = next(trial_dir.rglob("best_model.pt"), None)
        if checkpoint_path is None:
            return {}

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        info       = {key: checkpoint.get(key) for key in _CHECKPOINT_KEYS}

        del checkpoint

        info["n_train_epochs"] = self._epochs_completed(checkpoint_path.parent)

        return {key: value for key, value in info.items() if value is not None}

    @staticmethod
    def _epochs_completed(run_dir: Path) -> int | None:
        """Returns the epoch count from the run's completion marker, or None when absent."""
        marker_path = CompletionMarker.path(run_dir)

        if not marker_path.is_file():
            return None

        return int(FileIO.load_json(marker_path)["epochs_completed"])

    def _attach_inference(self, record: TrialRecord) -> None:
        """Attaches the newest completed inference stamp of the configured split.

        Args:
            record : Trial record to fill in place with the inference directory, metrics, animations, figures and report path.
        """
        if self.inference_split is None:
            return

        candidates = [stamp for stamp in CompletedInference.stamps(record.run_dir / "inference", self.inference_split) if (stamp / "metrics.json").exists()]
        if not candidates:
            return

        inference_dir = candidates[-1]

        record.inference_dir = inference_dir
        record.metrics       = FileIO.load_json(inference_dir / "metrics.json")
        record.animations    = sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else []

        self._attach_figures(record, inference_dir)

        report_path = inference_dir / "report.md"
        if report_path.exists():
            record.report_path = report_path

    def _attach_figures(self, record: TrialRecord, inference_dir: Path) -> None:
        """Attaches the sorted PNG figures of an inference stamp to the record."""
        record.figures = sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else []


class TrainingResultsIndex:
    """Per-run training status and duration, from the index file or the markers.

    Attributes:
        FILENAME : Name of the index file written by the training stage.
        path     : Full path of that file inside the pipeline directory.
    """

    FILENAME = "training_results.json"

    def __init__(self, pipeline_dir: Path, logger: Logger) -> None:
        """Points the index at the pipeline directory of a benchmark or CV run."""
        self.path   = Path(pipeline_dir) / self.FILENAME
        self.logger = logger

    def _stored(self) -> dict:
        """Returns the persisted training results keyed by run name."""
        return {result["name"]: result for result in FileIO.load_json(self.path)}

    def _derived(self, runs: list[tuple[str, Path]]) -> dict:
        """Rebuilds training results from completion markers when no index exists.

        Args:
            runs : (run name, run directory) pairs to inspect.

        Returns:
            Results keyed by run name, with duration unavailable.
        """
        self.logger.warning(f"No {self.path.name} in {self.path.parent}: rebuilding training results from the completion markers of {len(runs)} run(s)")

        return {name: self._marker_result(name, run_dir) for name, run_dir in runs}

    def _marker_result(self, name: str, run_dir: Path) -> dict:
        """Returns a DONE/PARTIAL result derived from one run's completion marker."""
        return {
            "name"       : name,
            "status"     : "DONE" if CompletionMarker.is_complete(run_dir) else "PARTIAL",
            "duration_s" : None,
        }

    def build(self, runs: list[tuple[str, Path]]) -> dict:
        """Returns the training results keyed by run name, stored or marker-derived.

        Args:
            runs : (run name, run directory) pairs used for the marker fallback.
        """
        if self.path.is_file():
            return self._stored()

        return self._derived(runs)


class TrialCollector(TrialArtifactReader):
    """Builds the `TrialRecord` list of a benchmark or cross-validation run.

    Attributes:
        run_dir         : Root of the pipeline run.
        training_dir    : Directory holding one sub-directory per trial.
        pipeline_dir    : Directory holding size_match.json and training_results.json.
        inference_split : Split whose inference stamps are attached.
    """

    def __init__(self, run_dir: Path, logger: Logger, inference_split: str | None) -> None:
        """Resolves the training and pipeline sub-directories of the run.

        Args:
            run_dir         : Root of the benchmark or cross-validation run.
            logger          : Logger for per-trial progress lines.
            inference_split : Split whose completed inference stamps are attached, or None to collect training artifacts only.
        """
        self.run_dir         = run_dir
        self.training_dir    = run_dir / "training"
        self.pipeline_dir    = run_dir / "pipeline"
        self.logger          = logger
        self.inference_split = inference_split

    def _optional_json(self, path: Path) -> dict:
        """Returns the JSON payload at the path, or an empty dict when it is missing."""
        if not path.exists():
            return {}
        return FileIO.load_json(path)

    def _parse_parameters(self, trial_dir: Path, size_match: dict) -> int | None:
        """Returns the trainable parameter count of a trial.

        Args:
            trial_dir  : Training directory whose docs/model_doc.md is parsed first.
            size_match : Width-scaling record used as fallback source.

        Returns:
            Parameter count, or None when neither source records one.
        """
        summary_path = trial_dir / "docs" / "model_doc.md"

        if summary_path.exists():
            match = _TOTAL_PARAMS_PATTERN.search(summary_path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                return int(match.group(1).replace(",", ""))

        return size_match["parameters"] if "parameters" in size_match else None

    @staticmethod
    def _model_of(trial_name: str) -> str:
        """Returns the size-match key of a trial: a backbone-head model key or the bare stem."""
        base  = SeedSet.base(trial_name).split("__")[0]
        parts = base.split("-")

        if len(parts) >= 2 and parts[1] in BACKBONE_HEADS:
            return ModelBuilder.model_key(parts[0], parts[1])

        return base

    @staticmethod
    def _seed_dirs(trial_dir: Path) -> list[Path]:
        """Returns the sorted seedNN sub-directories of a trial directory."""
        return sorted(d for d in trial_dir.iterdir() if d.is_dir() and re.fullmatch(r"seed\d+", d.name))

    def _run_dirs(self) -> list[tuple[str, Path]]:
        """Returns (run name, run directory) pairs, expanding seed sweeps into one entry per seed."""
        runs = []
        for trial_dir in sorted(d for d in self.training_dir.iterdir() if d.is_dir()):
            seed_dirs = self._seed_dirs(trial_dir)

            if seed_dirs:
                runs += [(f"{trial_dir.name}/{seed_dir.name}", seed_dir) for seed_dir in seed_dirs]
            else:
                runs.append((trial_dir.name, trial_dir))

        return runs

    def _aggregate_sources(self, runs: list[tuple[str, Path]]) -> tuple[dict, dict]:
        """Returns the pipeline-level size-match record and the training results index.

        Args:
            runs : (run name, run directory) pairs used to rebuild the index when the training results file is absent.
        """
        size_match       = self._optional_json(self.pipeline_dir / "size_match.json")
        training_results = TrainingResultsIndex(self.pipeline_dir, self.logger).build(runs)

        return size_match, training_results

    def collect(self) -> list[TrialRecord]:
        """Reads every trial of the run into a record and logs its inference status.

        Returns:
            One record per run directory, or an empty list when the run has no
            training directory.
        """
        if not self.training_dir.is_dir():
            self.logger.error(f"No training directory found at: {self.training_dir}")
            return []

        runs = self._run_dirs()

        size_match, training_results = self._aggregate_sources(runs)

        records = []
        for name, trial_dir in runs:
            record = TrialRecord(name=name, run_dir=trial_dir)

            record.size_match      = size_match.get(self._model_of(name), {})
            record.trainer_config  = self._optional_json(trial_dir / "docs" / "trainer_config.json")
            record.run_summary     = self._optional_json(trial_dir / "meta" / "run_summary.json")
            record.training_result = training_results[name] if name in training_results else {}
            record.parameters      = self._parse_parameters(trial_dir, record.size_match)
            record.checkpoint      = self._read_checkpoint(trial_dir)

            self._attach_inference(record)

            status = f"inference {record.inference_dir.name}" if record.has_inference else "no inference"
            self.logger.info(f"{record.name:<22} {status}")

            records.append(record)

        return records
