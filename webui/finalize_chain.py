
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from configuration.diagnostics import ReportMailEntryConfig
from web_logger                import WebLogger


class FinalizeChain:

    STEPS           = ("compare_seeds", "compare_trials", "mail_experiment_reports")
    MAIL_STEP       = "mail_experiment_reports"
    SKIP_DIRS       = ("logs", "batch_train_logs", "report")
    TAGGED_SWEEPS   = {"cross_validate": "folds", "benchmark": "training"}
    STAGING_DIRNAME = "_mailed_reports"

    @classmethod
    def flags(cls, finalize) -> dict:
        raw = dict(finalize or {})
        return {step: bool(raw.get(step)) for step in cls.STEPS}

    def steps(self, flags: dict) -> tuple:
        return tuple(step for step in self.STEPS if flags[step])

    def mail_blocked(self) -> str:
        return "" if os.environ.get(ReportMailEntryConfig.password_env, "") else ReportMailEntryConfig.password_env

    def experiment_dir(self, script: str, overrides: dict, leaves: dict, run_dir: str = "") -> str:
        if script in self.TAGGED_SWEEPS:
            run_tag = overrides.get("run_tag", "") or leaves.get("run_tag", "")
            if run_tag in ("", "None"):
                raise ValueError(f"the finalize steps on a '{script}' run need a fixed 'run_tag' to locate the {self.TAGGED_SWEEPS[script]} sweep; set one or turn the finalize switches off")

            base = overrides.get("paths.log_base_dir", "") or leaves.get("paths.log_base_dir", "") or run_dir
            return str(Path(base) / run_tag / self.TAGGED_SWEEPS[script])

        mode = overrides.get("trials_mode", "") or leaves.get("trials_mode", "")
        if not mode:
            raise ValueError(f"the finalize steps need an entry writing a sweep: a training entry exposing 'trials_mode', the cross-validation entry or the benchmark entry, which '{script}' is not; turn the finalize switches off")

        logdir = overrides.get("logdir", "") or leaves.get("logdir", "") or run_dir
        return str(Path(logdir) / mode)

    def trial_names(self, experiment_dir: str) -> list:
        root = Path(experiment_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"experiment directory not found: {root}")

        names = sorted(child.name for child in root.iterdir() if child.is_dir() and child.name not in self.SKIP_DIRS and not child.name.startswith("_"))
        if not names:
            raise FileNotFoundError(f"no trial directories under {root}")

        return names

    def step_overrides(self, step: str, experiment_dir: str, run_dir: str = "") -> dict:
        names = self.trial_names(experiment_dir)

        if step == "compare_seeds":
            return {"runs_dir": experiment_dir, "group_tags": ", ".join(names)}
        if step == "compare_trials":
            return {"runs_dir": experiment_dir, "run_tags": ", ".join(names)}

        overrides = {"experiments_dir": experiment_dir}
        if run_dir:
            overrides["staging_dir"] = str(Path(run_dir) / self.STAGING_DIRNAME)

        return overrides


class RunFinalizer:

    INTERVAL_S = 2.0
    ENDED_OK   = ("finished",)
    ENDED_BAD  = ("failed", "cancelled")

    def __init__(self, launcher, processes, resolver, notifier, logger: WebLogger) -> None:
        self.launcher  = launcher
        self.processes = processes
        self.resolver  = resolver
        self.notifier  = notifier
        self.logger    = logger
        self.chain     = FinalizeChain()
        self.watched   = {}
        self.lock      = threading.Lock()

    def plan(self, key: str, interpreter: str, overrides: dict, finalize) -> tuple:
        flags = self.chain.flags(finalize)
        steps = self.chain.steps(flags)
        if not steps:
            return None, ""

        variable = self.chain.mail_blocked() if flags[self.chain.MAIL_STEP] else ""
        if variable:
            return None, f"the report-mail finalize step needs {variable} in the server environment; export it and restart the webui, or switch the mail step off"

        resolved = self.resolver.resolve(key, interpreter)
        if not resolved.get("ok"):
            return None, f"the finalize steps need the resolved config of '{key}' to locate its sweep directory: {resolved.get('error', 'resolution failed')}"

        leaves = {leaf["path"]: str(leaf["value"]) for leaf in resolved["leaves"]}

        try:
            experiment_dir = self.chain.experiment_dir(key, overrides, leaves)
        except ValueError as error:
            return None, str(error)

        return {"script": key, "experiment_dir": experiment_dir, "steps": list(steps), "index": 0, "running": True, "step_job": None, "launching": False}, ""

    def watch(self, job_id: str, plan: dict) -> None:
        with self.lock:
            self.watched[job_id] = plan

        self.logger.muted(f"job {job_id} finalizes with {' then '.join(plan['steps'])} on {plan['experiment_dir']}")

    def pending(self) -> list:
        with self.lock:
            return [{"job_id": job_id, **chain} for job_id, chain in self.watched.items()]

    def _job_status(self, job_id: str) -> str:
        with self.processes.lock:
            record = self.processes.jobs.get(job_id)
            return record["status"] if record else ""

    def _advance(self) -> list:
        launches = []

        with self.lock:
            for job_id, chain in list(self.watched.items()):
                if chain["running"]:
                    status = self._job_status(job_id)
                    if status in self.ENDED_BAD or not status:
                        self.watched.pop(job_id)
                        self.logger.warning(f"job {job_id} ended as {status or 'gone from the job table'}, its finalize chain is dropped")
                        continue
                    if status not in self.ENDED_OK:
                        continue
                    chain["running"] = False

                if chain["step_job"]:
                    status = self._job_status(chain["step_job"])
                    if status in self.ENDED_BAD or not status:
                        step = chain["steps"][chain["index"]]
                        self.watched.pop(job_id)
                        self._report_failure(job_id, chain, step, f"job {chain['step_job']} ended as {status or 'gone from the job table'}")
                        continue
                    if status not in self.ENDED_OK:
                        continue
                    chain["step_job"] = None
                    chain["index"]   += 1

                if chain["launching"]:
                    continue

                if chain["index"] >= len(chain["steps"]):
                    self.watched.pop(job_id)
                    self.logger.ok(f"job {job_id}: finalize chain finished ({', '.join(chain['steps'])})")
                    continue

                chain["launching"] = True
                launches.append((job_id, dict(chain)))

        return launches

    def _launch_step(self, job_id: str, chain: dict) -> None:
        step = chain["steps"][chain["index"]]

        try:
            overrides = self.chain.step_overrides(step, chain["experiment_dir"])
        except FileNotFoundError as error:
            self._drop(job_id, step, str(error))
            return

        result = self.launcher.execute(step, self.launcher.preferred_interpreter(step), overrides, None, False, False)
        if not result.get("ok"):
            self._drop(job_id, step, result.get("error", "launch failed"))
            return

        with self.lock:
            live              = self.watched[job_id]
            live["step_job"]  = result["job_id"]
            live["launching"] = False

        self.logger.ok(f"job {job_id}: finalize step '{step}' running as job {result['job_id']}")

    def _drop(self, job_id: str, step: str, error: str) -> None:
        with self.lock:
            chain = self.watched.pop(job_id, None)

        if chain is not None:
            self._report_failure(job_id, chain, step, error)

    def _report_failure(self, job_id: str, chain: dict, step: str, error: str) -> None:
        self.logger.error(f"job {job_id} ({chain['script']}): finalize step '{step}' failed: {error}")
        self.notifier.system_alert(f"finalize step failed: {step}", f"{chain['script']} job {job_id}: finalize step '{step}' failed: {error}")

    def scan(self) -> None:
        for job_id, chain in self._advance():
            self._launch_step(job_id, chain)

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.scan()
            except Exception as exc:
                self.logger.error(f"finalize chain error: {exc}")
