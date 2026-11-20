
from __future__ import annotations

import shutil
import sys
from dataclasses import asdict
from pathlib     import Path
from typing      import Callable

from tools.data.io            import FileIO
from tools.monitoring.logger  import Logger
from tools.runtime.completion import CompletionMarker

from tools.orchestration.gpu_queue import GpuJob, GpuPoolFile, GpuQueue


class ExperimentStage:

    def __init__(self, config, run_tag: str, logger: Logger, entry_script: Path | None = None, run_dir: Path | None = None, pool_file: Path | None = None) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = run_tag
        self.logger       = logger
        self.run_dir      = Path(run_dir) if run_dir is not None else Path(config.paths.log_base_dir) / run_tag
        self._pool_file   = Path(pool_file) if pool_file is not None else None

    @property
    def pool_file(self) -> Path:
        if self._pool_file is None:
            self._pool_file = GpuPoolFile.resolve(self.config.gpus_file, self.run_dir)
        return self._pool_file

    def _run_queue(self, jobs: list[GpuJob]) -> list[dict]:
        queue   = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s, pool_file=self.pool_file)
        results = queue.run(jobs)

        self.config.gpus = list(queue.gpus)

        return [asdict(result) for result in results]

    def _order_results(self, results: list[dict], names: list[str]) -> list[dict]:
        by_name = {result["name"]: result for result in results}
        return [by_name[name] for name in names if name in by_name]

    def _write_results(self, results, path: Path) -> None:
        FileIO.save_json(results, path, indent=2)

    def _log_failures(self, failed: list[dict], name_key: str = "name") -> None:
        for result in failed:
            log_hint = f"  (see {result['log_file']})" if result["log_file"] else ""
            self.logger.error(f"FAILED  {result[name_key]}{log_hint}")


class QueuedStage(ExperimentStage):

    TRAINER_STATE = "last.pt"

    stage_subdir     : str = "training"
    worker_action    : str = ""
    worker_logname   : str = "worker.log"
    results_filename : str = ""

    def __init__(self, config, entry_script: Path, run_tag: str, items: list[str], logger: Logger, unit_dirs: dict | None = None) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.items        = items
        self.unit_dirs    = dict(unit_dirs or {})
        self.stage_dir    = self.run_dir / self.stage_subdir
        self.results_path = self.run_dir / "pipeline" / self.results_filename

    def unit_name(self, item: str) -> str:
        return self.unit_dirs.get(item, item)

    def unit_dir(self, item: str) -> Path:
        return self.stage_dir / self.unit_name(item)

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), "--worker", self.worker_action, self._worker_flag(), self._worker_value(item), "--run-name", self.unit_name(item), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.unit_dir(item) / self.worker_logname,
        )

    def _worker_flag(self) -> str:
        return "--model"

    def _worker_value(self, item: str) -> str:
        return item

    def _training_complete(self, item: str) -> bool:
        return CompletionMarker.is_complete(self.unit_dir(item))

    def _resumable(self, item: str) -> bool:
        return bool(self.config.training.resume) and (self.unit_dir(item) / self.TRAINER_STATE).is_file()

    def _scored_splits(self) -> list[str]:
        return [self.config.inference.split]

    def _dispatch(self, items: list[str]) -> list[dict]:
        if not items:
            return []

        return [{**result, "reused": False} for result in self._run_queue([self._job(item) for item in items])]

    def _reused(self, items: list[str], synthetic: Callable[[str], dict]) -> list[dict]:
        if not items:
            return []

        earlier = {result["name"]: result for result in FileIO.load_json(self.results_path)} if self.results_path.is_file() else {}
        results = []
        carried = 0

        for item in items:
            result = {**synthetic(item), "reused": True}
            if item in earlier and earlier[item]["status"] == "DONE":
                result["gpu"]        = earlier[item]["gpu"]
                result["duration_s"] = earlier[item]["duration_s"]
                result["returncode"] = earlier[item]["returncode"]
                carried             += 1
            results.append(result)

        self.logger.info(f"{carried} of {len(items)} reused unit(s) keep the GPU and duration recorded in {self.results_path}; {len(items) - carried} have none on record")

        return results

    def _inference_base(self, directory: Path) -> Path:
        return directory / "inference"

    def status_report(self) -> "UnitStatusReport":
        return UnitStatusReport(
            items          = self.items,
            unit_dir       = self.unit_dir,
            stage_dir      = self.stage_dir,
            splits         = self._scored_splits(),
            inference_base = self._inference_base,
            trainer_resume = bool(self.config.training.resume),
            resume         = bool(self.config.resume),
        )


class QueuedTrainingStage(QueuedStage):

    worker_action    : str = "train"
    summary_title    : str = "Training summary"
    worker_logname   : str = "worker.log"
    results_filename : str = "training_results.json"

    def _config_kv(self) -> dict:
        return {
            "Items"      : len(self.items),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Stage dir"  : str(self.stage_dir),
        }

    def _is_complete(self, item: str) -> bool:
        if not self.config.resume:
            return False

        return self._training_complete(item)

    def _purge_unfinished(self, item: str) -> None:
        item_dir = self.unit_dir(item)
        if not self.config.resume or not item_dir.is_dir():
            return

        if self._resumable(item):
            self.logger.info(f"{item}: unfinished run resumes from {self.TRAINER_STATE} in {item_dir}")
            return

        self.logger.warning(f"{item}: unfinished run detected, deleting {item_dir} and restarting")
        shutil.rmtree(item_dir)

    def _cached_result(self, item: str) -> dict:
        return {
            "name"       : item,
            "gpu"        : None,
            "status"     : "DONE",
            "returncode" : 0,
            "duration_s" : None,
            "log_file"   : str(self.unit_dir(item) / self.worker_logname),
        }

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] != "DONE"]

        self.logger.subsection(self.summary_title)
        self.logger.kv_table({
            "Total"  : len(results),
            "Done"   : len(done),
            "Failed" : len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)

    def run(self) -> list[dict]:
        self.logger.section("Training queue")
        self.logger.kv_table(self._config_kv(), title="Configuration")

        self.status_report().log(self.logger, "Unit status")

        cached  = [item for item in self.items if self._is_complete(item)]
        pending = [item for item in self.items if item not in cached]

        for item in cached:
            self.logger.info(f"{item}: completed run reused")

        for item in pending:
            self._purge_unfinished(item)

        reused  = self._reused(cached, self._cached_result)
        ran     = self._dispatch(pending)
        results = self._order_results(ran + reused, self.items)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results


class QueuedInferenceStage(QueuedStage):

    worker_action    : str = "infer"
    summary_title    : str = "Inference summary"
    worker_logname   : str = "inference_worker.log"
    results_filename : str = "inference_results.json"

    def _config_kv(self) -> dict:
        return {
            "Items" : len(self.items),
            "Split" : self.config.inference.split,
            "GPUs"  : self.config.gpus,
        }

    def _has_inference(self, item: str) -> bool:
        if not self.config.resume:
            return False

        inference_dir = self.unit_dir(item) / "inference"
        if not inference_dir.is_dir():
            return False

        completed = [candidate for candidate in inference_dir.iterdir() if candidate.is_dir() and CompletionMarker.is_complete(candidate)]

        return any(CompletionMarker.payload(candidate)["split"] == self.config.inference.split for candidate in completed)

    def _purge_unfinished(self, item: str) -> None:
        inference_dir = self.unit_dir(item) / "inference"
        if not self.config.resume or not inference_dir.is_dir():
            return

        for candidate in inference_dir.iterdir():
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                self.logger.warning(f"{item}: unfinished inference '{candidate.name}' deleted before rerun")
                shutil.rmtree(candidate)

    def _static_result(self, item: str, status: str) -> dict:
        return {
            "name"       : item,
            "gpu"        : None,
            "status"     : status,
            "returncode" : 0 if status == "DONE" else None,
            "duration_s" : None,
            "log_file"   : str(self.unit_dir(item) / self.worker_logname),
        }

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] == "FAILED"]

        self.logger.subsection(self.summary_title)
        self.logger.kv_table({
            "Total"   : len(results),
            "Done"    : len(done),
            "Failed"  : len(failed),
            "Skipped" : len(results) - len(done) - len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)

    def run(self) -> list[dict]:
        self.logger.section("Batch inference")
        self.logger.kv_table(self._config_kv(), title="Configuration")

        skipped = [item for item in self.items if not self._training_complete(item)]
        cached  = [item for item in self.items if item not in skipped and self._has_inference(item)]
        pending = [item for item in self.items if item not in skipped and item not in cached]

        for item in skipped:
            self.logger.warning(f"{item}: training incomplete, inference skipped")

        for item in cached:
            self.logger.info(f"{item}: existing inference reused")

        for item in pending:
            self._purge_unfinished(item)

        reused  = self._reused(cached, lambda item: self._static_result(item, "DONE"))
        ran     = self._dispatch(pending)
        results = ran + reused + [{**self._static_result(item, "SKIPPED"), "reused": False} for item in skipped]
        results = self._order_results(results, self.items)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results


class UnitStatusReport:

    REFUSED = "complete, refused (resume off)"

    def __init__(self, items: list[str], unit_dir: Callable[[str], Path], stage_dir: Path, splits: list[str], inference_base: Callable[[Path], Path], trainer_resume: bool, resume: bool, trainer_state: str = QueuedStage.TRAINER_STATE) -> None:
        self.items          = list(items)
        self.unit_dir       = unit_dir
        self.stage_dir      = Path(stage_dir)
        self.splits         = list(splits)
        self.inference_base = inference_base
        self.trainer_resume = trainer_resume
        self.resume         = resume
        self.trainer_state  = trainer_state

    def _training_status(self, directory: Path) -> str:
        if CompletionMarker.is_complete(directory):
            return "complete" if self.resume else self.REFUSED
        if not directory.is_dir():
            return "new"
        if self.trainer_resume and (directory / self.trainer_state).is_file():
            return "resumes from checkpoint"

        return "restarts"

    def _inference_status(self, directory: Path) -> str:
        if not self.splits:
            return "-"

        from pipelines.shared.inference.metadata import CompletedInference

        base  = self.inference_base(directory)
        words = [f"{split} {'done' if CompletedInference.latest(base, split) is not None else 'missing'}" for split in self.splits]

        return ", ".join(words)

    def rows(self) -> list[dict]:
        rows = []
        for item in self.items:
            directory = self.unit_dir(item)
            rows.append({
                "Unit"      : item,
                "Directory" : directory.relative_to(self.stage_dir).as_posix() if directory.is_dir() else "(new)",
                "Training"  : self._training_status(directory),
                "Inference" : self._inference_status(directory),
            })

        return rows

    def log(self, logger: Logger, title: str) -> None:
        rows      = self.rows()
        complete  = sum(1 for row in rows if row["Training"] == "complete")
        refused   = [row["Unit"] for row in rows if row["Training"] == self.REFUSED]
        resumed   = sum(1 for row in rows if row["Training"] == "resumes from checkpoint")
        fresh     = len(rows) - complete - len(refused) - resumed

        logger.subsection(title)
        logger.metrics_table(rows, columns=["Unit", "Directory", "Training", "Inference"])
        logger.info(f"{complete} unit(s) complete, {resumed} resume from their checkpoint, {fresh} train from scratch")

        if refused:
            logger.warning(f"resume is off: {len(refused)} completed unit(s) hold a finished run that is never overwritten in place, so they refuse to train and fail; set resume=True to reuse them or launch under a new run_tag: {', '.join(refused)}")
