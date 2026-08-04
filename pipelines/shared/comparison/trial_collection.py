from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field, replace
from pathlib     import Path

import numpy as np
import torch

from models                               import BACKBONE_HEADS
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
        return self.inference_dir is not None


class SeedRunAggregator:
    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self) -> None:
        self.seed_dispersion: dict = {}

    def _group(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(SeedSet.base(record.name), []).append(record)

        return list(groups.items())

    def _aggregate_group(self, name: str, runs: list[TrialRecord]) -> tuple[TrialRecord, dict]:
        representative = next((run for run in runs if run.has_inference), runs[0])

        metric_keys              = sorted({key for run in runs for key in run.metrics})
        metric_means, metric_std = SeedAggregation.aggregate([run.metrics for run in runs], metric_keys)
        ckpt_means, ckpt_std     = SeedAggregation.aggregate([run.checkpoint for run in runs], list(self.CHECKPOINT_KEYS))

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
            "best_val_loss_std" : ckpt_std.get("best_val_loss"),
            "metrics"           : metric_std,
            "seed_metrics"      : {run.name.rsplit("/", 1)[-1]: run.metrics for run in runs},
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
    def _read_checkpoint(self, trial_dir: Path) -> dict:
        checkpoint_path = next(trial_dir.rglob("best_model.pt"), None)
        if checkpoint_path is None:
            return {}

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        info       = {key: checkpoint.get(key) for key in _CHECKPOINT_KEYS}

        del checkpoint
        gc.collect()

        info["n_train_epochs"] = self._epochs_completed(checkpoint_path.parent)

        return {key: value for key, value in info.items() if value is not None}

    @staticmethod
    def _epochs_completed(run_dir: Path) -> int | None:
        marker_path = CompletionMarker.path(run_dir)

        if not marker_path.is_file():
            return None

        return int(FileIO.load_json(marker_path)["epochs_completed"])

    def _attach_inference(self, record: TrialRecord) -> None:
        inference_root = record.run_dir / "inference"
        if not inference_root.is_dir():
            return

        candidates = sorted(d for d in inference_root.iterdir() if d.is_dir() and (d / "metrics.json").exists())
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
        record.figures = sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else []


class TrialCollector(TrialArtifactReader):
    def __init__(self, run_dir: Path, logger: Logger) -> None:
        self.run_dir      = run_dir
        self.training_dir = run_dir / "training"
        self.pipeline_dir = run_dir / "pipeline"
        self.logger       = logger

    def _optional_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return FileIO.load_json(path)

    def _parse_parameters(self, trial_dir: Path, size_match: dict) -> int | None:
        summary_path = trial_dir / "docs" / "model_doc.md"

        if summary_path.exists():
            match = _TOTAL_PARAMS_PATTERN.search(summary_path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                return int(match.group(1).replace(",", ""))

        return size_match["parameters"] if "parameters" in size_match else None

    def _aggregate_sources(self) -> tuple[dict, dict]:
        size_match       = self._optional_json(self.pipeline_dir / "size_match.json")
        training_results = {r["name"]:  r for r in FileIO.load_json(self.pipeline_dir / "training_results.json")}

        return size_match, training_results

    @staticmethod
    def _model_of(trial_name: str) -> str:
        base  = SeedSet.base(trial_name).split("__")[0]
        parts = base.split("-")

        if len(parts) >= 2 and parts[1] in BACKBONE_HEADS:
            return ModelBuilder.model_key(parts[0], parts[1])

        return base

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

    def collect(self) -> list[TrialRecord]:
        if not self.training_dir.is_dir():
            self.logger.error(f"No training directory found at: {self.training_dir}")
            return []

        size_match, training_results = self._aggregate_sources()

        records = []
        for name, trial_dir in self._run_dirs():
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
