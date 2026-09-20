"""Post-experiment chain comparing an experiment's units and mailing its reports.

``FinalizeChain`` derives what the chain does for one launch: which steps run,
which sweep directory they run on, and the overrides each step is launched
with. ``RunFinalizer`` walks that chain for a run launched on its own from the
scripts tab, so a single training, benchmark or cross-validation launch ends
with the same seed comparison, trial comparison and report mail a sequence
block ends with. The sequence runner walks its own chain block by block over
the same derivation, inside the execution workspace.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from configuration.diagnostics import ReportMailEntryConfig
from web_logger                import WebLogger


class FinalizeChain:
    """Derives the post-experiment chain comparing an experiment's units and mailing its reports.

    The chain runs on a sweep directory holding one run per compared unit: the
    ``<logdir>/<trials_mode>`` directory of a trial-sweep training entry, the
    ``<log_base_dir>/<run_tag>/folds`` directory of a cross-validation entry,
    whose fold runs are compared exactly like trials, and the
    ``<log_base_dir>/<run_tag>/training`` directory of a benchmark entry, whose
    architecture units are compared the same way.

    Attributes:
        STEPS           : Entry-point keys of the chain, launched in order on the finished run's experiment.
        MAIL_STEP       : Chain step key of the report mailer, the step whose SMTP password is preflighted before the run starts.
        SKIP_DIRS       : Directory names inside a sweep directory that are not compared units.
        TAGGED_SWEEPS   : Sweep directory name inside a run-tagged experiment, keyed by the entry point writing it.
        STAGING_DIRNAME : Directory name inside a sequence workspace receiving the mail step's staged bundles.
    """

    STEPS           = ("compare_seeds", "compare_trials", "mail_experiment_reports")
    MAIL_STEP       = "mail_experiment_reports"
    SKIP_DIRS       = ("logs", "batch_train_logs")
    TAGGED_SWEEPS   = {"cross_validate": "folds", "benchmark": "training"}
    STAGING_DIRNAME = "_mailed_reports"

    @classmethod
    def flags(cls, finalize) -> dict:
        """Returns finalize switches as one boolean per chain step, all off when absent.

        Args:
            finalize : Switches as stored or posted, a mapping of step key to flag, or None.
        """
        raw = dict(finalize or {})
        return {step: bool(raw.get(step)) for step in cls.STEPS}

    def steps(self, flags: dict) -> tuple:
        """Returns the enabled chain steps in launch order.

        Args:
            flags : Finalize switches, one boolean per step key.
        """
        return tuple(step for step in self.STEPS if flags[step])

    def mail_blocked(self) -> str:
        """Returns the SMTP password variable the server environment lacks, or an empty string.

        The report-mail step inherits the server process environment, so a
        missing password is caught before the run starts instead of failing the
        chain hours later, after training and the comparisons succeeded.
        """
        return "" if os.environ.get(ReportMailEntryConfig.password_env, "") else ReportMailEntryConfig.password_env

    def experiment_dir(self, script: str, overrides: dict, leaves: dict, run_dir: str = "") -> str:
        """Returns the sweep directory the chain runs on.

        The training entries write their sweep into ``<logdir>/<trials_mode>``,
        the cross-validation entry writes its fold runs into
        ``<log_base_dir>/<run_tag>/folds`` and the benchmark entry writes its
        units into ``<log_base_dir>/<run_tag>/training``, so the directory
        follows from the effective mode or run tag: the launch's override when
        it sets one and the entry's default otherwise. The sweep root follows
        the same effective log directory: a sequence's workspace injection
        already redirected it into the workspace, so a deliberately external
        directory is honoured here exactly as it is at launch, and an entry
        without a log-directory leaf falls back to the workspace itself.

        Args:
            script    : Entry-point key of the launch.
            overrides : Launch overrides, a sequence's workspace already injected.
            leaves    : Resolved leaf defaults of the entry, keyed by path.
            run_dir   : Isolated workspace directory of a sequence execution, or empty for a run launched on its own.

        Raises:
            ValueError : If the entry exposes no sweep the chain can run on, or a run-tagged entry leaves the run tag to the timestamp default.
        """
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
        """Returns the sorted trial directory names inside one experiment directory.

        Args:
            experiment_dir : Sweep directory of the finished run.

        Raises:
            FileNotFoundError : If the directory is missing or holds no trial directory.
        """
        root = Path(experiment_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"experiment directory not found: {root}")

        names = sorted(child.name for child in root.iterdir() if child.is_dir() and child.name not in self.SKIP_DIRS and not child.name.startswith("_"))
        if not names:
            raise FileNotFoundError(f"no trial directories under {root}")

        return names

    def step_overrides(self, step: str, experiment_dir: str, run_dir: str = "") -> dict:
        """Returns the launch overrides of one chain step.

        The comparison steps run over every unit of the experiment directory,
        and the mailing step scans that directory's parent restricted to this
        experiment. Every step is read against the unit runs first, so a run
        that wrote no sweep stops the chain instead of mailing an empty bundle.
        A sequence execution stages its bundle inside its own workspace so
        blocks of other executions never overwrite the same staging archive; a
        run launched on its own stages where the mail entry config points.

        Args:
            step           : Entry-point key of the chain step.
            experiment_dir : Sweep directory of the finished run.
            run_dir        : Isolated workspace directory of a sequence execution holding the mail step's staging tree, or empty to stage at the entry's default.

        Raises:
            FileNotFoundError : If the experiment directory holds no unit runs.
        """
        names = self.trial_names(experiment_dir)

        if step == "compare_seeds":
            return {"runs_dir": experiment_dir, "group_tags": ", ".join(names)}
        if step == "compare_trials":
            return {"runs_dir": experiment_dir, "run_tags": ", ".join(names)}

        overrides = {"experiments_dir": str(Path(experiment_dir).parent), "experiment_filter": Path(experiment_dir).name}
        if run_dir:
            overrides["staging_dir"] = str(Path(run_dir) / self.STAGING_DIRNAME)

        return overrides


class RunFinalizer:
    """Chains the compare-and-mail steps onto a run launched on its own from the scripts tab.

    A launch that switches any finalize step on registers its chain here. The
    watch loop polls the launched job through the process manager and, once it
    finished, launches the enabled steps one after another on the sweep
    directory the run wrote, each step attached so its exit code decides
    whether the next one starts. The chain is derived before the run starts, so
    a launch that names no sweep the chain can run on is refused instead of
    failing hours later. A run that failed, was cancelled or left the job table
    drops its chain, and so does a failed step: a comparison that crashed
    leaves the mail step nothing to send. The chains live in memory only, so a
    console restart forgets the ones still waiting.

    Attributes:
        launcher  : Run launcher executing each chain step.
        processes : Process manager whose job table supplies the watched job states.
        resolver  : Config resolver supplying the leaf defaults the sweep directory is derived from.
        notifier  : Job notifier chain failures are pushed through.
        logger    : Console logger.
        chain     : Derivation of the chain steps, the sweep directory and the step overrides.
        watched   : Live chain state keyed by the job id of the run it hangs on.
        lock      : Guards the watch table.
    """

    INTERVAL_S = 2.0
    ENDED_OK   = ("finished",)
    ENDED_BAD  = ("failed", "cancelled")

    def __init__(self, launcher, processes, resolver, notifier, logger: WebLogger) -> None:
        """Stores the collaborators and starts with no watched runs."""
        self.launcher  = launcher
        self.processes = processes
        self.resolver  = resolver
        self.notifier  = notifier
        self.logger    = logger
        self.chain     = FinalizeChain()
        self.watched   = {}
        self.lock      = threading.Lock()

    def plan(self, key: str, interpreter: str, overrides: dict, finalize) -> tuple:
        """Returns the chain a launch asks for, or the error that blocks the launch.

        Args:
            key         : Entry-point key being launched.
            interpreter : Interpreter of the launch, also used to resolve the entry's leaf defaults.
            overrides   : Config leaf overrides of the launch.
            finalize    : Finalize switches of the launch, or None when it asks for no chain.

        Returns:
            Pair of the chain state and an empty error; ``(None, "")`` when no
            step is switched on, and ``(None, error)`` when the mail password
            is missing, the entry's config cannot be resolved, or the entry
            writes no sweep the chain can run on.
        """
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
        """Registers a launched run's chain, walked once that run finished.

        Args:
            job_id : Console job id of the launched run.
            plan   : Chain state built by `plan`.
        """
        with self.lock:
            self.watched[job_id] = plan

        self.logger.muted(f"job {job_id} finalizes with {' then '.join(plan['steps'])} on {plan['experiment_dir']}")

    def pending(self) -> list:
        """Returns one snapshot per watched chain, for the console to report what is still queued."""
        with self.lock:
            return [{"job_id": job_id, **chain} for job_id, chain in self.watched.items()]

    def _job_status(self, job_id: str) -> str:
        """Returns the process manager's status for one job, or an empty string when it left the job table."""
        with self.processes.lock:
            record = self.processes.jobs.get(job_id)
            return record["status"] if record else ""

    def _advance(self) -> list:
        """Advances every watched chain and returns the steps that are ready to launch.

        A chain waits for its run to finish, then walks its steps one at a
        time: the next step starts only once the previous one exited cleanly,
        and the chain is dropped as soon as the run or a step ends badly.

        Returns:
            List of (job id of the watched run, chain state snapshot) pairs
            whose next step is ready; the launch itself runs with the lock
            released.
        """
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
        """Launches one chain step outside the lock and records its job id or its failure.

        The step always runs attached and on this machine, whatever the run it
        hangs on did: a detached process hides its exit code, so a crashed
        comparison would read as finished and wave the chain through to a mail
        step that has nothing to send.

        Args:
            job_id : Console job id of the run the chain hangs on.
            chain  : Chain state snapshot naming the step index and the experiment directory.
        """
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
        """Drops one chain whose step could not start and reports the failure.

        Args:
            job_id : Console job id of the run the chain hangs on.
            step   : Entry-point key of the step that failed to start.
            error  : Failure description.
        """
        with self.lock:
            chain = self.watched.pop(job_id, None)

        if chain is not None:
            self._report_failure(job_id, chain, step, error)

    def _report_failure(self, job_id: str, chain: dict, step: str, error: str) -> None:
        """Logs and notifies one broken chain step.

        Args:
            job_id : Console job id of the run the chain hangs on.
            chain  : Chain state of the dropped chain.
            step   : Entry-point key of the step that failed.
            error  : Failure description.
        """
        self.logger.error(f"job {job_id} ({chain['script']}): finalize step '{step}' failed: {error}")
        self.notifier.system_alert(f"finalize step failed: {step}", f"{chain['script']} job {job_id}: finalize step '{step}' failed: {error}")

    def scan(self) -> None:
        """Runs one advance pass over every watched chain, then launches what it marked.

        The advance pass is the only part holding the watch lock; the launches
        it collected run afterwards with the lock released, each recording its
        own outcome back into the watch table.
        """
        for job_id, chain in self._advance():
            self._launch_step(job_id, chain)

    def start(self) -> None:
        """Starts the background polling thread that walks the watched chains forever."""
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()

    def _watch(self) -> None:
        """Scans on the fixed interval forever, logging any scan failure instead of exiting."""
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.scan()
            except Exception as exc:
                self.logger.error(f"finalize chain error: {exc}")
