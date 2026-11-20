
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib     import Path

import numpy as np
import torch

from configuration.reference_layout        import ReferenceLayout
from pipelines.shared.inference.metadata   import CompletedInference
from pipelines.shared.inference.run_layout import InferenceLayout
from pipelines.shared.training.seed_sweep  import SeedSet
from tools.data.io                         import FileIO
from tools.metrics.scoring                 import SeedAggregation
from tools.monitoring.logger               import Logger
from tools.runtime.completion              import CompletionMarker


@dataclass
class TrialRecord:

    name            : str
    run_dir         : Path
    parameters      : int | None  = None
    trainer_config  : dict        = field(default_factory=dict)
    run_summary     : dict        = field(default_factory=dict)
    checkpoint      : dict        = field(default_factory=dict)
    training_result : dict        = field(default_factory=dict)
    inference_dir   : Path | None = None
    metrics         : dict        = field(default_factory=dict)
    figures         : list[Path]  = field(default_factory=list)
    report_path     : Path | None = None

    @property
    def has_inference(self) -> bool:
        return self.inference_dir is not None


class SeedRunAggregator:

    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self, logger: Logger) -> None:
        self.logger          = logger
        self.seed_dispersion = {}

    def _group(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(SeedSet.base(record.name), []).append(record)

        return list(groups.items())

    def _report_coverage(self, name: str, runs: list[TrialRecord], seed_names: list[str], metric_keys: list[str], contributors: dict[str, int], representative: TrialRecord) -> None:
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

    CHECKPOINT_FILENAME = ReferenceLayout.CHECKPOINT

    logger          : Logger
    inference_split : str | None

    def _read_checkpoint(self, trial_dir: Path) -> dict:
        checkpoint_path = trial_dir / self.CHECKPOINT_FILENAME
        if not checkpoint_path.is_file():
            return {}

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        info       = {"best_val_loss": checkpoint["best_val_loss"], "best_epoch": checkpoint["best_epoch"] + 1, "epoch": checkpoint["epoch"] + 1}

        del checkpoint

        self.logger.debug(f"{trial_dir.name}: checkpoint scalars read from {checkpoint_path}")

        info["n_train_epochs"] = self._epochs_completed(trial_dir)

        return {key: value for key, value in info.items() if value is not None}

    @staticmethod
    def _epochs_completed(run_dir: Path) -> int | None:
        marker_path = CompletionMarker.path(run_dir)

        if not marker_path.is_file():
            return None

        return int(FileIO.load_json(marker_path)["epochs_completed"])

    def _attach_inference(self, record: TrialRecord) -> None:
        if self.inference_split is None:
            return

        candidates = [stamp for stamp in CompletedInference.stamps(record.run_dir / "inference", self.inference_split) if (stamp / "metrics.json").exists()]
        if not candidates:
            return

        inference_dir = candidates[-1]

        record.inference_dir = inference_dir
        record.metrics       = FileIO.load_json(inference_dir / "metrics.json")

        self._attach_figures(record, inference_dir)

        report_path = inference_dir / "report.md"
        if report_path.exists():
            record.report_path = report_path

    def _attach_figures(self, record: TrialRecord, inference_dir: Path) -> None:
        record.figures = sorted((inference_dir / "figures").rglob("*.png"))

    def _warn_no_inference(self, record: TrialRecord) -> None:
        if self.inference_split is None or record.has_inference:
            return

        inference_dir = record.run_dir / InferenceLayout.DIRNAME

        self.logger.warning(f"{record.name}: no completed inference on split '{self.inference_split}' under {inference_dir}; listed but not ranked")

    def _warn_unfinished(self, record: TrialRecord) -> None:
        if not (record.run_dir / self.CHECKPOINT_FILENAME).is_file():
            self.logger.warning(f"{record.name}: no {self.CHECKPOINT_FILENAME} under {record.run_dir}; training saved no best checkpoint, compared without checkpoint scalars")

        if not CompletionMarker.is_complete(record.run_dir):
            self.logger.warning(f"{record.name}: no {CompletionMarker.FILENAME} under {record.run_dir}; training did not finish (crashed or still running), compared as is")

    def _status(self, record: TrialRecord) -> str:
        inference = f"inference {record.inference_dir.name}" if record.has_inference else "no inference"
        epochs    = record.checkpoint.get("n_train_epochs", "?")
        complete  = "yes" if CompletionMarker.is_complete(record.run_dir) else "no"

        return f"{inference}, epochs {epochs}, complete {complete}"


class TrainingResultsIndex:

    FILENAME = "training_results.json"

    def __init__(self, pipeline_dir: Path, logger: Logger) -> None:
        self.path   = Path(pipeline_dir) / self.FILENAME
        self.logger = logger

    def _stored(self) -> dict:
        return {result["name"]: result for result in FileIO.load_json(self.path)}

    def _derived(self, runs: list[tuple[str, Path]]) -> dict:
        self.logger.warning(f"No {self.path.name} in {self.path.parent}: rebuilding training results from the completion markers of {len(runs)} run(s)")

        return {name: self._marker_result(name, run_dir) for name, run_dir in runs}

    def _marker_result(self, name: str, run_dir: Path) -> dict:
        return {
            "name"       : name,
            "status"     : "DONE" if CompletionMarker.is_complete(run_dir) else "PARTIAL",
            "duration_s" : None,
        }

    def build(self, runs: list[tuple[str, Path]]) -> dict:
        if self.path.is_file():
            return self._stored()

        return self._derived(runs)


class TrialCollector(TrialArtifactReader):

    def __init__(self, run_dir: Path, logger: Logger, inference_split: str | None) -> None:
        self.run_dir         = run_dir
        self.training_dir    = run_dir / "training"
        self.pipeline_dir    = run_dir / "pipeline"
        self.logger          = logger
        self.inference_split = inference_split

    def _run_json(self, record: TrialRecord, path: Path) -> dict:
        if not path.is_file():
            self.logger.warning(f"{record.name}: no {path.relative_to(record.run_dir).as_posix()} under {record.run_dir}; its report cells stay empty")
            return {}

        return FileIO.load_json(path)

    def _training_result(self, record: TrialRecord, training_results: dict) -> dict:
        if record.name in training_results:
            return training_results[record.name]

        self.logger.warning(f"{record.name}: absent from {TrainingResultsIndex.FILENAME}; training status and duration unknown")
        return {}

    @staticmethod
    def _seed_dirs(trial_dir: Path) -> list[Path]:
        return sorted(d for d in trial_dir.iterdir() if d.is_dir() and re.fullmatch(r"seed\d+", d.name))

    def _run_dirs(self) -> list[tuple[str, Path]]:
        runs = []
        for trial_dir in sorted(d for d in self.training_dir.iterdir() if d.is_dir()):
            seed_dirs = self._seed_dirs(trial_dir)

            if seed_dirs:
                runs += [(f"{trial_dir.name}/{seed_dir.name}", seed_dir) for seed_dir in seed_dirs]
            else:
                runs.append((trial_dir.name, trial_dir))

        return runs

    def _training_results_index(self, runs: list[tuple[str, Path]]) -> dict:
        return TrainingResultsIndex(self.pipeline_dir, self.logger).build(runs)

    def _summarise(self, records: list[TrialRecord]) -> None:
        n_inferred   = sum(1 for record in records if record.has_inference)
        n_unfinished = sum(1 for record in records if not CompletionMarker.is_complete(record.run_dir))
        n_no_ckpt    = sum(1 for record in records if not (record.run_dir / self.CHECKPOINT_FILENAME).is_file())
        inference    = f"{n_inferred} with inference on split '{self.inference_split}'" if self.inference_split is not None else "inference not attached"

        self.logger.ok(f"{len(records)} run(s) collected: {inference}, {n_unfinished} without {CompletionMarker.FILENAME}, {n_no_ckpt} without {self.CHECKPOINT_FILENAME}")

    def collect(self) -> list[TrialRecord]:
        runs = self._run_dirs()

        training_results = self._training_results_index(runs)

        records = []
        for name, trial_dir in runs:
            record = TrialRecord(name=name, run_dir=trial_dir)

            record.trainer_config  = self._run_json(record, trial_dir / "docs" / "trainer_config.json")
            record.run_summary     = self._run_json(record, trial_dir / "meta" / "run_summary.json")
            record.training_result = self._training_result(record, training_results)
            record.parameters      = record.run_summary["parameters"] if record.run_summary else None
            record.checkpoint      = self._read_checkpoint(trial_dir)

            self._attach_inference(record)
            self._warn_unfinished(record)
            self._warn_no_inference(record)
            self.logger.info(f"{record.name:<22} {self._status(record)}")

            records.append(record)

        self._summarise(records)

        return records
