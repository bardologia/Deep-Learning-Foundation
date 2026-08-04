from __future__ import annotations

import shutil
import sys
from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.cv_report import CrossValidationReport
from pipelines.cross_validation.folds     import FoldPlanner, FoldRunNaming
from pipelines.cross_validation.workers   import FoldCollector
from pipelines.shared.training.seed_sweep import SeedSet
from tools.orchestration.gpu_queue import GpuJob
from tools.orchestration.stages    import ExperimentStage, QueuedInferenceStage, QueuedTrainingStage
from tools.monitoring.logger       import Logger
from tools.runtime.completion      import CompletionMarker
from tools.runtime.run_tag         import RunTag


class FoldTrainingStage(QueuedTrainingStage):
    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        units      = FoldRunNaming(config).units()
        self._unit = {run_name: (fold_index, seed) for fold_index, seed, run_name in units}
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=[run_name for _, _, run_name in units], logger=logger)

    def _config_kv(self) -> dict:
        return {
            "Model"      : self.config.backbone_name,
            "Folds"      : self.config.folds.n_folds,
            "Seeds"      : self.config.seeds or "—",
            "Runs"       : len(self.items),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Folds dir"  : str(self.stage_dir),
        }

    def _job(self, item: str) -> GpuJob:
        fold_index, seed = self._unit[item]

        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), "--worker", self.worker_action, "--fold", str(fold_index), *SeedSet.cli_args(seed), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / item / self.worker_logname,
        )


class FoldInferenceStage(QueuedInferenceStage):
    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        units      = FoldRunNaming(config).units()
        self._unit = {run_name: (fold_index, seed) for fold_index, seed, run_name in units}
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=[run_name for _, _, run_name in units], logger=logger)
        self.planner = planner
        self.splits  = config.inference_splits

    def _split_log(self, run_name: str, split: str) -> Path:
        return self.stage_dir / run_name / f"inference_{split}_worker.log"

    def _split_result(self, run_name: str, split: str, status: str) -> dict:
        return {
            "name"       : f"{run_name}:{split}",
            "gpu"        : None,
            "status"     : status,
            "returncode" : 0 if status == "DONE" else None,
            "duration_s" : None,
            "log_file"   : str(self._split_log(run_name, split)),
        }

    def _has_split_inference(self, run_name: str, split: str) -> bool:
        if not self.config.resume:
            return False

        return CompletionMarker.is_complete(self.stage_dir / run_name / "inference" / split)

    def _purge_unfinished_split(self, run_name: str, split: str) -> None:
        split_dir = self.stage_dir / run_name / "inference" / split
        if not self.config.resume or not split_dir.is_dir():
            return

        self.logger.warning(f"{run_name}:{split}: unfinished inference deleted before rerun")
        shutil.rmtree(split_dir)

    def _split_job(self, fold_index: int, seed: int | None, run_name: str, split: str) -> GpuJob:
        return GpuJob(
            name     = f"{run_name}:{split}",
            command  = [sys.executable, str(self.entry_script), "--worker", "infer", "--fold", str(fold_index), *SeedSet.cli_args(seed), "--split", split, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self._split_log(run_name, split),
        )

    def run(self) -> list[dict]:
        self.logger.section("Fold inference queue")
        self.logger.kv_table({
            "Folds"  : self.config.folds.n_folds,
            "Seeds"  : self.config.seeds or "—",
            "Runs"   : len(self.items),
            "Splits" : ", ".join(self.splits),
            "GPUs"   : self.config.gpus,
        }, title="Configuration")

        jobs    = []
        results = []

        for run_name in self.items:
            fold_index, seed = self._unit[run_name]
            plan             = self.planner.plan(fold_index)

            if not self._training_complete(run_name):
                self.logger.warning(f"{run_name}: training incomplete, inference skipped")
                for split in self.splits:
                    results.append(self._split_result(run_name, split, "SKIPPED"))
                continue

            for split in self.splits:
                if len(plan.split_regions.regions(split)) != 1:
                    self.logger.warning(f"{run_name}:{split}: split is disjoint, inference skipped (stitching requires one contiguous region)")
                    results.append(self._split_result(run_name, split, "SKIPPED"))
                    continue

                if self._has_split_inference(run_name, split):
                    self.logger.info(f"{run_name}:{split}: existing inference reused")
                    results.append(self._split_result(run_name, split, "DONE"))
                    continue

                self._purge_unfinished_split(run_name, split)
                jobs.append(self._split_job(fold_index, seed, run_name, split))

        if jobs:
            results.extend(self._run_queue(jobs))

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results


class CrossValidationReportStage(ExperimentStage):
    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.planner = planner

    def run(self) -> Path:
        self.logger.section("Cross-validation reports")

        splits     = self.config.inference_splits if self.config.runs_inference() else []
        model_name = self.config.backbone_name if self.config.training_type != "profile_autoencoder" else "profile_ae"

        collector = FoldCollector(run_dir=self.run_dir, splits=splits, logger=self.logger)
        base_records, records_by_split = collector.collect_by_split()

        out_dir = self.run_dir / "reports" / RunTag.now()

        report = CrossValidationReport(
            base_records     = base_records,
            records_by_split = records_by_split,
            planner          = self.planner,
            out_dir          = out_dir,
            model_name       = model_name,
            embed_images     = self.config.comparison.embed_images,
            logger           = self.logger,
            seed_dispersion  = collector.seed_dispersion,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
