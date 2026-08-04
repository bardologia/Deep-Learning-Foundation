from __future__ import annotations

import shutil
import sys
from dataclasses import asdict
from pathlib     import Path

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
        self.pool_file    = Path(pool_file) if pool_file is not None else GpuPoolFile.resolve(config.gpus_file, self.run_dir)

    def _run_queue(self, jobs: list[GpuJob]) -> list[dict]:
        queue = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s, pool_file=self.pool_file)
        return [asdict(result) for result in queue.run(jobs)]

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
    stage_subdir     : str = "training"
    worker_action    : str = ""
    worker_logname   : str = "worker.log"
    results_filename : str = ""

    def __init__(self, config, entry_script: Path, run_tag: str, items: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.items        = items
        self.stage_dir    = self.run_dir / self.stage_subdir
        self.results_path = self.run_dir / "pipeline" / self.results_filename

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), "--worker", self.worker_action, self._worker_flag(), self._worker_value(item), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / item / self.worker_logname,
        )

    def _worker_flag(self) -> str:
        return "--model"

    def _worker_value(self, item: str) -> str:
        return item


class QueuedTrainingStage(QueuedStage):
    worker_action    : str = "train"
    summary_title    : str = "Training summary"
    worker_logname   : str = "worker.log"
    results_filename : str = "training_results.json"

    def _config_kv(self) -> dict:
        return {
            "Items"      : len(self.items),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : f"{self.config.training.batch_size} (base; a max_batch probe stage overrides it per model, see each worker's [Loaders] table)",
            "GPUs"       : self.config.gpus,
            "Stage dir"  : str(self.stage_dir),
        }

    def _is_complete(self, item: str) -> bool:
        if not self.config.resume:
            return False

        return CompletionMarker.is_complete(self.stage_dir / item)

    def _purge_unfinished(self, item: str) -> None:
        item_dir = self.stage_dir / item
        if not self.config.resume or not item_dir.is_dir():
            return

        self.logger.warning(f"{item}: unfinished run detected, deleting and restarting")
        shutil.rmtree(item_dir)

    def _cached_result(self, item: str) -> dict:
        return {
            "name"       : item,
            "gpu"        : None,
            "status"     : "DONE",
            "returncode" : 0,
            "duration_s" : None,
            "log_file"   : str(self.stage_dir / item / self.worker_logname),
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

        cached  = [item for item in self.items if self._is_complete(item)]
        pending = [item for item in self.items if item not in cached]

        for item in cached:
            self.logger.info(f"{item}: completed run reused")

        for item in pending:
            self._purge_unfinished(item)

        ran = []
        if pending:
            ran = self._run_queue([self._job(item) for item in pending])

        results = self._order_results(ran + [self._cached_result(item) for item in cached], self.items)

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

    def _training_complete(self, item: str) -> bool:
        return CompletionMarker.is_complete(self.stage_dir / item)

    def _has_inference(self, item: str) -> bool:
        if not self.config.resume:
            return False

        inference_dir = self.stage_dir / item / "inference"
        if not inference_dir.is_dir():
            return False

        completed = [candidate for candidate in inference_dir.iterdir() if candidate.is_dir() and CompletionMarker.is_complete(candidate)]

        return any(CompletionMarker.payload(candidate)["split"] == self.config.inference.split for candidate in completed)

    def _purge_unfinished(self, item: str) -> None:
        inference_dir = self.stage_dir / item / "inference"
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
            "log_file"   : str(self.stage_dir / item / self.worker_logname),
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

        ran = []
        if pending:
            ran = self._run_queue([self._job(item) for item in pending])

        results = ran + [self._static_result(item, "DONE") for item in cached] + [self._static_result(item, "SKIPPED") for item in skipped]
        results = self._order_results(results, self.items)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results
