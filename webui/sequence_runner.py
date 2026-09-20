"""Execution engine that walks experiment sequences over the job backends.

Holds the live state of every started sequence, launches a block as soon as all
of its predecessor blocks have finished, launches independent blocks in
parallel, holds checkpoint blocks until they are explicitly released, skips the
descendants of a failed block, and injects the sequence's shared dataset into
each block's overrides at launch time. Every block runs through the process
manager.

An isolated sequence gets a fresh workspace directory per execution under the
repository's ``runs/sequences`` tree; every block's run-output root and every
runs-tree input root are redirected into it, so repeated executions never
collide and downstream blocks read exactly the runs this execution produced.
A start may instead name the workspace of an earlier execution to resume it:
the run reuses that directory, so the blocks that still launch read and write
the runs the earlier execution already produced. Every isolated execution
mirrors its live state into a JSON file inside the workspace whenever it
changes; the in-memory state dies with the server, but the file survives, and
a resume scan reads it to report which blocks already finished so the resumed
start can bypass them instead of redoing them. That bypass list lives only in
the execution's live state, never in the stored definition, so a later plain
start in a fresh workspace runs every block again.

A block whose skip toggle is set is bypassed instead of launched: it counts as
satisfied for its successors, so after a crash or stop the finished parts of a
sequence can be marked skip and a restart replays only the rest. The toggle is
read fresh from the store on every scan, so it can be flipped mid-run for any
block that has not launched yet.

A block carries one finalize switch per post-experiment step: the seed
comparison aggregating each trial's seed runs, the trial comparison writing the
experiment's comparison reports, and the report mailer sending the experiment
bundle. The enabled steps are chained in that order onto the block's successful
end. The chain runs on the block's trial-sweep directory inside the isolated
workspace and does not gate the block's successors: they launch as soon as the
block's own job finishes, while the chain runs beside them. A failed chain step
marks the block finalize_failed, which still satisfies its successors but ends
the sequence as failed.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from web_logger import WebLogger

from configuration.diagnostics      import ReportMailEntryConfig
from configuration.reference_layout import ReferenceLayout


class SequenceFinalizer:
    """Derives the post-experiment chain comparing a block's trials and mailing its reports.

    The chain runs on a sweep directory holding one run per compared unit: the
    ``<workspace>/<trials_mode>`` directory of a trial-sweep block, and the
    ``<workspace>/<run_tag>/folds`` directory of a cross-validation block,
    whose fold runs are compared exactly like trials.

    Attributes:
        STEPS           : Entry-point keys of the chain, launched in order on the finished block's experiment.
        MAIL_STEP       : Chain step key of the report mailer, the step whose SMTP password is preflighted at sequence start.
        SKIP_DIRS       : Directory names inside a sweep directory that are not compared units.
        CV_SCRIPT       : Entry-point key of the cross-validation entry.
        CV_SWEEP        : Sweep directory name inside a cross-validation run holding the fold runs.
        STAGING_DIRNAME : Directory name inside a workspace receiving the mail step's staged bundles.
    """

    STEPS           = ("compare_seeds", "compare_trials", "mail_experiment_reports")
    MAIL_STEP       = "mail_experiment_reports"
    SKIP_DIRS       = ("logs", "batch_train_logs")
    CV_SCRIPT       = "cross_validate"
    CV_SWEEP        = "folds"
    STAGING_DIRNAME = "_mailed_reports"

    def mail_preflight(self, blocks: list) -> str:
        """Returns the start-blocking error when mailing is enabled without an SMTP password.

        The report-mail chain step inherits the server process environment, so a
        missing password variable is caught at sequence start instead of failing
        the chain hours later, after training and comparisons succeeded.

        Args:
            blocks : Block definitions of the sequence being started.

        Returns:
            Error naming the mailing blocks and the variable, or an empty string.
        """
        mailers = [block["name"] or block["title"] for block in blocks if not block["skip"] and block["finalize"][self.MAIL_STEP]]

        if not mailers or os.environ.get(ReportMailEntryConfig.password_env, ""):
            return ""

        return f"block(s) {', '.join(mailers)} enable the report-mail finalize step, but the server environment has no {ReportMailEntryConfig.password_env}; export it and restart the webui, or switch the mail step off"

    def steps(self, flags: dict) -> tuple:
        """Returns the enabled chain steps in launch order.

        Args:
            flags : Finalize switches of the block, one boolean per step key.
        """
        return tuple(step for step in self.STEPS if flags[step])

    def experiment_dir(self, script: str, overrides: dict, leaves: dict, run_dir: str) -> str:
        """Returns the block's sweep directory inside the workspace.

        The training entries write their sweep into ``<logdir>/<trials_mode>``
        and the cross-validation entry writes its fold runs into
        ``<log_base_dir>/<run_tag>/folds``, so the experiment directory follows
        from the effective mode or run tag: the block's override when it sets
        one and the entry's default otherwise. The sweep root follows the same
        effective ``logdir``: the workspace injection already redirected a
        runs-tree value into the workspace, so a deliberately external
        directory is honoured here exactly as it is at launch, and an entry
        without a ``logdir`` leaf falls back to the workspace itself.

        Args:
            script    : Entry-point key of the block.
            overrides : Launch overrides of the block, workspace already injected.
            leaves    : Resolved leaf defaults of the block's entry, keyed by path.
            run_dir   : Isolated workspace directory of the execution.

        Raises:
            ValueError : If the entry exposes no sweep the chain can run on, or a cross-validation block leaves the run tag to the timestamp default.
        """
        if script == self.CV_SCRIPT:
            run_tag = overrides.get("run_tag", "") or leaves.get("run_tag", "")
            if run_tag in ("", "None"):
                raise ValueError("the finalize steps on a cross-validation block need a fixed 'run_tag' override to locate the fold sweep; set one or turn the finalize switches off")
            return str(Path(run_dir) / run_tag / self.CV_SWEEP)

        mode = overrides.get("trials_mode", "") or leaves.get("trials_mode", "")
        if not mode:
            raise ValueError(f"the finalize steps need a trial-sweep entry exposing 'trials_mode' or the cross-validation entry, which '{script}' is not; turn the finalize switches off for this block")

        logdir = overrides.get("logdir", "") or leaves.get("logdir", "") or run_dir
        return str(Path(logdir) / mode)

    def trial_names(self, experiment_dir: str) -> list:
        """Returns the sorted trial directory names inside one experiment directory.

        Args:
            experiment_dir : Trial-sweep directory of the finished block.

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

    def step_overrides(self, step: str, experiment_dir: str, run_dir: str) -> dict:
        """Returns the launch overrides of one chain step.

        The comparison steps run over every unit of the experiment directory,
        and the mailing step scans that directory's parent restricted to this
        experiment, staging its bundle inside the workspace so blocks of other
        sequence executions never overwrite the same staging archive.

        Args:
            step           : Entry-point key of the chain step.
            experiment_dir : Sweep directory of the finished block.
            run_dir        : Isolated workspace directory of the execution, holding the mail step's staging tree.

        Raises:
            FileNotFoundError : If the experiment directory holds no unit runs for a comparison step.
        """
        if step == "compare_seeds":
            return {"runs_dir": experiment_dir, "group_tags": ", ".join(self.trial_names(experiment_dir))}
        if step == "compare_trials":
            return {"runs_dir": experiment_dir, "run_tags": ", ".join(self.trial_names(experiment_dir))}

        return {"experiments_dir": str(Path(experiment_dir).parent), "experiment_filter": Path(experiment_dir).name, "staging_dir": str(Path(run_dir) / self.STAGING_DIRNAME)}


class SequenceRunner:
    """Advances started sequences by polling job states and launching ready blocks.

    Attributes:
        store            : Sequence store supplying the current definition of each sequence.
        processes        : Process manager whose job table supplies local block job states.
        launcher         : Run launcher executing local blocks.
        resolver         : Config resolver used to find which dataset fields a block's entry exposes.
        finalizer        : Builder of the compare-and-mail chain appended to finalize blocks.
        notifier         : Job notifier block failures and sequence endings are pushed through.
        logger           : Console logger.
        active           : Live sequence states keyed by sequence id.
        released         : Released checkpoint block ids per sequence id.
        persisted        : Last state-file text written per sequence id, to skip unchanged writes.
        lock             : Guards the active table, the released sets, and the state-file staleness check and write.
    """

    INTERVAL_S     = 2.0
    ENDED_OK       = ("finished",)
    ENDED_BAD      = ("failed", "cancelled")
    STATE_FILENAME = "sequence_state.json"

    def __init__(self, store, processes, launcher, resolver, notifier, logger: WebLogger) -> None:
        """Stores the collaborators and starts with no active sequences."""
        self.store     = store
        self.processes = processes
        self.launcher  = launcher
        self.resolver  = resolver
        self.finalizer = SequenceFinalizer()
        self.notifier  = notifier
        self.logger    = logger
        self.active    = {}
        self.released  = {}
        self.persisted = {}
        self.lock      = threading.Lock()

    def start_sequence(self, sequence_id: str, resume_dir: str = "", bypass: list | None = None) -> dict:
        """Starts one sequence, in a fresh workspace or resuming a previous one.

        A non-empty ``resume_dir`` names the workspace directory of an earlier
        isolated execution; the run reuses it instead of creating a fresh one,
        and the blocks named in ``bypass`` are bypassed for this execution
        only, so the remaining blocks read and write the runs that execution
        produced. The bypass list is never written into the stored definition,
        so a later plain start runs every block again.

        Args:
            sequence_id : Identifier of the sequence to start.
            resume_dir  : Workspace directory of the execution to resume, or empty to start fresh.
            bypass      : Identifiers of blocks bypassed for this execution only, typically the ones a resume scan reported finished.

        Returns:
            Result with the fresh state, or ``ok`` False when the sequence is
            unknown, empty, already running, the resume workspace is invalid,
            or a block would mail reports without the SMTP password in the
            server environment.
        """
        entry = self.store.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}
        if not entry["blocks"]:
            return {"ok": False, "error": "the sequence has no blocks"}

        error = self.finalizer.mail_preflight(entry["blocks"])
        if error:
            return {"ok": False, "error": error}

        run_dir, error = self._resume_workspace(entry, resume_dir) if resume_dir else ("", "")
        if error:
            return {"ok": False, "error": error}

        with self.lock:
            state = self.active.get(sequence_id)
            if state is not None and state["status"] in ("running", "held"):
                return {"ok": False, "error": "the sequence is already running"}

            self.active[sequence_id] = {
                "sequence_id" : sequence_id,
                "status"      : "running",
                "started_at"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run_dir"     : run_dir or (self._make_workspace(entry) if entry["isolate"] else ""),
                "bypass"      : sorted(str(block_id) for block_id in (bypass or [])),
                "blocks"      : {block["block_id"]: self._fresh_block_state() for block in entry["blocks"]},
            }
            self.released[sequence_id] = set()

        self.logger.ok(f"started sequence '{entry['name']}' ({sequence_id}) with {len(entry['blocks'])} blocks" + (f", resuming workspace {run_dir}" if run_dir else ""))
        self.scan()
        return {"ok": True, "state": self.state(sequence_id)}

    def resume_scan(self, sequence_id: str, resume_dir: str) -> dict:
        """Reads the execution state a previous run recorded into its workspace.

        The runner mirrors every isolated execution's live state into the
        workspace, so a scan of that file reports which blocks already ran to
        completion; the caller feeds those back as the resumed start's bypass
        list. A block whose finalize chain failed counts as finished here: its
        own job completed, so it is proposed for bypassing instead of being
        retrained. A block the recorded execution itself bypassed as part of
        its resume list also counts as finished, so chained resumes keep
        proposing it; a block bypassed by a manual skip toggle does not, since
        nothing shows it ever ran.

        Args:
            sequence_id : Identifier of the sequence being resumed.
            resume_dir  : Workspace directory of the execution to resume.

        Returns:
            Result with ``recorded`` False when the workspace holds no state
            file, or the recorded per-block statuses plus the ids of the blocks
            that finished and still exist in the sequence; ``ok`` False when the
            sequence is unknown or the workspace is invalid.
        """
        entry = self.store.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}

        run_dir, error = self._resume_workspace(entry, resume_dir)
        if error:
            return {"ok": False, "error": error}

        path = Path(run_dir) / self.STATE_FILENAME
        if not path.is_file():
            return {"ok": True, "recorded": False, "run_dir": run_dir, "statuses": {}, "done": []}

        recorded = json.loads(path.read_text(encoding="utf-8"))
        statuses = {block_id: block["status"] for block_id, block in recorded["blocks"].items()}
        current  = {block["block_id"] for block in entry["blocks"]}
        resumed  = set(recorded["bypass"])
        done     = sorted(block_id for block_id, status in statuses.items() if (status in ("done", "finalize_failed") or (status == "bypassed" and block_id in resumed)) and block_id in current)

        return {"ok": True, "recorded": True, "run_dir": run_dir, "statuses": statuses, "done": done}

    def release(self, sequence_id: str, block_id: str) -> dict:
        """Releases one held checkpoint block so the next scan launches it.

        Args:
            sequence_id : Identifier of the running sequence.
            block_id    : Identifier of the held block.

        Returns:
            Result with the updated state, or ``ok`` False when the sequence is
            not active or the block is not currently held.
        """
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None:
                return {"ok": False, "error": "sequence is not running"}

            block = state["blocks"].get(block_id)
            if block is None or block["status"] != "held":
                return {"ok": False, "error": "block is not held at a checkpoint"}

            self.released[sequence_id].add(block_id)

        self.logger.ok(f"released checkpoint block {block_id} of sequence {sequence_id}")
        self.scan()
        return {"ok": True, "state": self.state(sequence_id)}

    def stop_sequence(self, sequence_id: str) -> dict:
        """Stops one sequence, cancelling its running blocks and skipping the rest.

        Blocks whose launch is still in flight are cancelled too; the launching
        thread sees the status change when it records the outcome and stops the
        job it just submitted.

        Args:
            sequence_id : Identifier of the sequence to stop.

        Returns:
            Result with the stopped state, or ``ok`` False when it is not active.
        """
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None or state["status"] not in ("running", "held"):
                return {"ok": False, "error": "sequence is not running"}

            state["status"] = "stopped"
            running = []
            for block in state["blocks"].values():
                if block["status"] in ("running", "launching"):
                    if block["job_id"]:
                        running.append(block["job_id"])
                    block["status"] = "cancelled"
                elif block["status"] == "finalizing":
                    if block["finalize"]["job_id"]:
                        running.append(block["finalize"]["job_id"])
                    block["status"] = "cancelled"
                elif block["status"] in ("pending", "held"):
                    block["status"] = "skipped"

        for job_id in running:
            self._backend(job_id).stop(job_id)

        self._persist_states()
        self.logger.warning(f"stopped sequence {sequence_id}, {len(running)} running block(s) stopped")
        return {"ok": True, "state": self.state(sequence_id)}

    def clear_state(self, sequence_id: str) -> dict:
        """Discards one sequence's finished execution state so the board returns to idle.

        The workspace state file is left in place, so a later resume scan still
        sees which blocks the execution finished; only the in-memory state that
        the board and the rail render is dropped.

        Args:
            sequence_id : Identifier of the sequence whose state is discarded.

        Returns:
            Result with ``cleared`` True when a state was dropped, False when
            there was none, or ``ok`` False while the sequence is still active.
        """
        with self.lock:
            state = self.active.get(sequence_id)
            if state is not None and state["status"] in ("running", "held"):
                return {"ok": False, "error": "stop the sequence before clearing its state"}

            cleared = self.active.pop(sequence_id, None) is not None
            self.released.pop(sequence_id, None)
            self.persisted.pop(sequence_id, None)

        if cleared:
            self.logger.muted(f"cleared execution state of sequence {sequence_id}")
        return {"ok": True, "cleared": cleared}

    def state(self, sequence_id: str) -> dict | None:
        """Returns a snapshot of one sequence's live state, or None when never started."""
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None:
                return None
            return {**state, "blocks": {block_id: self._copy_block_state(block) for block_id, block in state["blocks"].items()}}

    def states(self) -> dict:
        """Returns a snapshot of every live sequence state keyed by sequence id."""
        with self.lock:
            return {sequence_id: {**state, "blocks": {block_id: self._copy_block_state(block) for block_id, block in state["blocks"].items()}} for sequence_id, state in self.active.items()}

    @staticmethod
    def _fresh_block_state() -> dict:
        """Returns the initial live state of one block."""
        return {"status": "pending", "job_id": None, "error": "", "experiment_dir": "", "finalize": None}

    @staticmethod
    def _copy_block_state(block_state: dict) -> dict:
        """Returns a snapshot copy of one block's live state, its finalize chain included."""
        finalize = block_state["finalize"]
        return {**block_state, "finalize": dict(finalize) if finalize else None}

    def _resume_workspace(self, entry: dict, resume_dir: str) -> tuple:
        """Validates the workspace of a previous execution against the sequences tree.

        Args:
            entry      : Sequence definition being started.
            resume_dir : Workspace directory fed by the caller.

        Returns:
            Pair of the normalised workspace path and an error message; exactly one of the two is non-empty.
        """
        if not entry["isolate"]:
            return "", "the sequence does not isolate its executions, so there is no workspace to resume; turn 'Isolated dir' on"

        workspace = Path(resume_dir).expanduser()
        tree      = self.store.paths.runs_dir / "sequences"
        if workspace.parent != tree:
            return "", f"the resume workspace must be a directory directly under {tree}"
        if not workspace.is_dir():
            return "", f"the resume workspace does not exist: {workspace}"

        return str(workspace), ""

    def _make_workspace(self, entry: dict) -> str:
        """Creates and returns a fresh per-execution run directory for an isolated sequence.

        The directory lives under ``<repo_root>/runs/sequences`` and is named after
        the sequence and a start stamp; a numeric suffix keeps it unique when the
        same sequence starts twice within one second.

        Args:
            entry : Sequence definition supplying the name.
        """
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", entry["name"]).strip("_").lower() or "sequence"
        base = self.store.paths.runs_dir / "sequences" / f"{slug}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        candidate = base
        counter   = 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}_{counter}")
            counter  += 1

        candidate.mkdir(parents=True)
        self.logger.ok(f"sequence workspace {candidate}")
        return str(candidate)

    def _backend(self, job_id: str):
        """Returns the process manager owning the job."""
        return self.processes

    def _job_status(self, job_id: str) -> str:
        """Returns the process manager's status for one job, or "failed" when it is gone.

        Args:
            job_id : Console job id of a launched block.
        """
        with self.processes.lock:
            record = self.processes.jobs.get(job_id)
            return record["status"] if record else "failed"

    def _launch_overrides(self, entry: dict, block: dict, interpreter: str, run_dir: str) -> tuple:
        """Returns the block's overrides with the dataset and workspace injected, plus the resolved leaf defaults.

        The dataset injection targets whichever dataset fields the block's entry
        config actually exposes: ``paths.dataset_path`` and ``paths.parameters_path``
        for single-dataset entries, and ``dataset_base_path`` plus ``dataset_filter``
        for multi-dataset entries, derived from the selected dataset directory.
        The workspace injection redirects the entry's run-output root into the
        sequence's run directory: ``paths.log_base_dir`` for the entries that
        expose it, ``logdir`` for the training entries that name it that way, and
        ``runs_dir`` for run-consuming inputs. The last two are only redirected
        when their effective value points at the project's runs tree, taking the
        block's own override when it sets one and the entry's default otherwise,
        so a deliberately external directory is left alone.

        Args:
            entry       : Sequence definition supplying the shared dataset selection.
            block       : Block definition supplying the stored overrides.
            interpreter : Interpreter used to resolve the entry's config leaves.
            run_dir     : Isolated workspace directory of this execution, or empty.

        Returns:
            Pair of the launch overrides and the leaf defaults keyed by path;
            the overrides are None when the entry could not be resolved, and
            the leaves are empty when no resolution was needed.
        """
        overrides = dict(block["overrides"])
        dataset   = entry["dataset"]["dataset_path"]
        if not dataset and not run_dir:
            return overrides, {}

        resolved = self.resolver.resolve(block["script"], interpreter)
        if not resolved.get("ok"):
            return None, {}

        leaves = {leaf["path"]: str(leaf["value"]) for leaf in resolved["leaves"]}
        params = entry["dataset"]["parameters_path"]

        if dataset:
            if "paths.dataset_path" in leaves:
                overrides["paths.dataset_path"] = dataset
            if "paths.parameters_path" in leaves and params:
                overrides["paths.parameters_path"] = params
            if "dataset_base_path" in leaves:
                overrides["dataset_base_path"] = str(Path(dataset).parent)
            if "dataset_filter" in leaves:
                overrides["dataset_filter"] = Path(dataset).name

        if run_dir:
            if "paths.log_base_dir" in leaves:
                overrides["paths.log_base_dir"] = run_dir
            for path in ("logdir", "runs_dir"):
                if path in leaves and self._runs_tree_path(overrides.get(path, leaves[path])):
                    overrides[path] = run_dir

        return overrides, leaves

    def _runs_tree_path(self, value: str) -> bool:
        """Returns whether a path value sits inside the project's runs tree.

        A relative value counts as in-tree when it names the reference layout's
        run-output root or a path below it, since the entry configs default their
        run-output roots to checkout-relative paths.

        Args:
            value : Effective value of a runs-tree input, the block's override when it sets one and the entry's resolved default otherwise.
        """
        reference = str(ReferenceLayout.RUNS_ROOT)

        if value == reference or value.startswith(f"{reference}/"):
            return True

        return any(value == root or value.startswith(f"{root}/") for root in self._runs_roots())

    def _runs_roots(self) -> tuple:
        """Returns the path roots whose subtrees count as the project's runs tree.

        The root is the checkout's own run-output directory, which the machine
        profile may have moved elsewhere; the reference layout's relative root is
        matched by ``_runs_tree_path`` itself.

        Raises:
            ValueError : If the root cannot be determined from the configuration.
        """
        roots = tuple(root for root in (str(self.store.paths.runs_dir),) if root.strip("/") not in ("", str(ReferenceLayout.RUNS_ROOT)))

        if not roots:
            raise ValueError("cannot determine the project's runs tree: the checkout run directory is not configured")
        return roots

    def _launch_block(self, entry: dict, block: dict, run_dir: str) -> None:
        """Launches one ready block outside the runner lock and records the outcome.

        Config resolution and the launch run with the lock released, so the
        sequence API stays responsive for the whole submission.

        Args:
            entry   : Sequence definition the block belongs to.
            block   : Block definition to launch, read fresh from the store.
            run_dir : Isolated workspace directory of this execution, or empty.
        """
        interpreter       = block["interpreter"] or self.launcher.preferred_interpreter(block["script"])
        overrides, leaves = self._launch_overrides(entry, block, interpreter, run_dir)

        if overrides is None:
            self._record_launch(entry, block, "", f"could not resolve the config of '{block['script']}' to inject the dataset and workspace")
            return

        experiment_dir = ""
        if any(block["finalize"].values()):
            try:
                experiment_dir = self.finalizer.experiment_dir(block["script"], overrides, leaves, run_dir)
            except ValueError as error:
                self._record_launch(entry, block, "", str(error))
                return

        result = self.launcher.execute(block["script"], interpreter, overrides, None, entry["detach"], False)
        if not result.get("ok"):
            self._record_launch(entry, block, "", result.get("error", "launch failed"))
            return
        job_id = result["job_id"]

        self._record_launch(entry, block, job_id, "", experiment_dir)

    def _record_launch(self, entry: dict, block: dict, job_id: str, error: str, experiment_dir: str = "") -> None:
        """Writes a finished launch back into the live state, tearing the job down when it is stale.

        A stop landing while the launch was in flight has already moved the
        block off "launching", so the outcome is discarded and the job that was
        just started is stopped instead of being adopted.

        Args:
            entry          : Sequence definition the block belongs to.
            block          : Block definition that was launched.
            job_id         : Console job id of the started job, or empty on failure.
            error          : Failure description, or empty when the launch succeeded.
            experiment_dir : Trial-sweep directory the block's finalize chain runs on, or empty.
        """
        with self.lock:
            state       = self.active.get(entry["sequence_id"])
            block_state = state["blocks"].get(block["block_id"]) if state else None
            stale       = block_state is None or block_state["status"] != "launching"

            if not stale and error:
                block_state["status"] = "failed"
                block_state["error"]  = error
            elif not stale:
                block_state["status"]         = "running"
                block_state["job_id"]         = job_id
                block_state["experiment_dir"] = experiment_dir

        if stale:
            if job_id:
                self._backend(job_id).stop(job_id)
            self.logger.warning(f"sequence {entry['sequence_id']}: block '{block['title']}' was stopped while launching, {'job ' + job_id + ' stopped' if job_id else 'no job started'}")
        elif error:
            self.logger.error(f"sequence {entry['sequence_id']}: block '{block['title']}' failed to launch: {error}")
            self.notifier.system_alert(f"sequence block failed: {block['title']}", f"sequence '{entry['name']}': failed to launch: {error}")
        else:
            self.logger.ok(f"sequence {entry['sequence_id']}: launched block '{block['title']}' as job {job_id}")

    def _launch_finalize_step(self, entry: dict, block: dict, block_state: dict, run_dir: str) -> None:
        """Launches one step of a block's compare-and-mail chain and records the outcome.

        The step runs locally through the process manager whatever the block's
        own target, so the comparisons and the mailing always read the local
        tree.

        Args:
            entry       : Sequence definition the block belongs to.
            block       : Block definition whose chain is advancing.
            block_state : Live state snapshot supplying the step index and experiment directory.
            run_dir     : Isolated workspace directory of the execution.
        """
        step = block_state["finalize"]["steps"][block_state["finalize"]["step"]]

        try:
            overrides = self.finalizer.step_overrides(step, block_state["experiment_dir"], run_dir)
        except FileNotFoundError as error:
            self._record_finalize_launch(entry, block, step, "", str(error))
            return

        result = self.launcher.execute(step, self.launcher.preferred_interpreter(step), overrides, None, entry["detach"], False)
        if not result.get("ok"):
            self._record_finalize_launch(entry, block, step, "", result.get("error", "launch failed"))
            return

        self._record_finalize_launch(entry, block, step, result["job_id"], "")

    def _record_finalize_launch(self, entry: dict, block: dict, step: str, job_id: str, error: str) -> None:
        """Writes a finished chain-step launch back into the live state, tearing the job down when it is stale.

        A stop landing while the step launched has already moved the block off
        "finalizing", so the outcome is discarded and the job stopped instead
        of being adopted, mirroring the block-launch path.

        Args:
            entry  : Sequence definition the block belongs to.
            block  : Block definition whose chain step was launched.
            step   : Entry-point key of the launched step.
            job_id : Console job id of the started job, or empty on failure.
            error  : Failure description, or empty when the launch succeeded.
        """
        with self.lock:
            state       = self.active.get(entry["sequence_id"])
            block_state = state["blocks"].get(block["block_id"]) if state else None
            stale       = block_state is None or block_state["status"] != "finalizing"

            if not stale and error:
                block_state["status"] = "finalize_failed"
                block_state["error"]  = f"finalize step '{step}' failed to launch: {error}"
            elif not stale:
                block_state["finalize"]["job_id"]    = job_id
                block_state["finalize"]["launching"] = False

        if stale:
            if job_id:
                self.processes.stop(job_id)
            self.logger.warning(f"sequence {entry['sequence_id']}: block '{block['title']}' left finalizing while step '{step}' launched, {'job ' + job_id + ' stopped' if job_id else 'no job started'}")
        elif error:
            self.logger.error(f"sequence {entry['sequence_id']}: block '{block['title']}' finalize step '{step}' failed to launch: {error}")
            self.notifier.system_alert(f"finalize step failed: {block['title']}", f"sequence '{entry['name']}': finalize step '{step}' failed to launch: {error}")
        else:
            self.logger.ok(f"sequence {entry['sequence_id']}: block '{block['title']}' finalize step '{step}' running as job {job_id}")

    def _advance(self, sequence_id: str, state: dict) -> list:
        """Advances one running sequence: syncs jobs, walks finalize chains, marks ready blocks.

        A block that is ready to run is marked "launching" and returned instead
        of being launched here, so the caller can perform the blocking launch
        with the runner lock released; a finalize chain step that is ready to
        run is returned the same way. A block marked skip (or bypassed for this
        execution) turns bypassed only once its own predecessors are satisfied,
        so a skip mark never unlocks the blocks behind it ahead of their turn.
        A finishing block with any finalize switch
        on moves to "finalizing" instead of "done" and its enabled steps launch
        one by one, without holding back its successors.

        Args:
            sequence_id : Identifier of the sequence being advanced.
            state       : Its live state, mutated in place under the runner lock.

        Returns:
            List of launch tuples for outside the lock: ("block", entry, block, run_dir)
            for ready blocks and ("finalize", entry, block, block_state, run_dir)
            for ready chain steps.
        """
        entry = self.store.get(sequence_id)
        if entry is None:
            state["status"] = "stopped"
            return []

        blocks = {block["block_id"]: block for block in entry["blocks"]}
        for block_id in [block_id for block_id in state["blocks"] if block_id not in blocks]:
            state["blocks"].pop(block_id)
        for block_id in blocks:
            state["blocks"].setdefault(block_id, self._fresh_block_state())

        for block_id, block_state in state["blocks"].items():
            if block_state["status"] != "running":
                continue
            status = self._job_status(block_state["job_id"])
            if status in self.ENDED_OK and any(blocks[block_id]["finalize"].values()) and state["run_dir"]:
                block_state["status"]   = "finalizing"
                block_state["finalize"] = {"steps": list(self.finalizer.steps(blocks[block_id]["finalize"])), "step": 0, "job_id": None, "launching": False}
            elif status in self.ENDED_OK:
                block_state["status"] = "done"
            elif status in self.ENDED_BAD:
                block_state["status"] = "failed"
                block_state["error"]  = f"job {block_state['job_id']} ended as {status}"
                self.notifier.system_alert(f"sequence block failed: {blocks[block_id]['title']}", f"sequence '{entry['name']}': {block_state['error']}")

        launches = []

        for block_id, block_state in state["blocks"].items():
            if block_state["status"] != "finalizing":
                continue

            chain = block_state["finalize"]
            if chain["job_id"]:
                status = self._job_status(chain["job_id"])
                if status in self.ENDED_OK:
                    chain["job_id"] = None
                    chain["step"] += 1
                elif status in self.ENDED_BAD:
                    block_state["status"] = "finalize_failed"
                    block_state["error"]  = f"finalize step '{chain['steps'][chain['step']]}' job {chain['job_id']} ended as {status}"
                    self.notifier.system_alert(f"finalize step failed: {blocks[block_id]['title']}", f"sequence '{entry['name']}': {block_state['error']}")
                    continue

            if chain["job_id"] is None and not chain["launching"]:
                if chain["step"] >= len(chain["steps"]):
                    block_state["status"] = "done"
                else:
                    chain["launching"] = True
                    launches.append(("finalize", entry, blocks[block_id], self._copy_block_state(block_state), state["run_dir"]))
        changed  = True
        while changed:
            changed = False
            for block_id, block in blocks.items():
                block_state = state["blocks"][block_id]
                if block_state["status"] not in ("pending", "held"):
                    continue

                pred_states = [state["blocks"][pred]["status"] for pred in block["predecessors"]]
                if any(pred in ("failed", "skipped") for pred in pred_states):
                    block_state["status"] = "skipped"
                    changed = True
                    continue

                if any(pred not in ("done", "bypassed", "finalizing", "finalize_failed") for pred in pred_states):
                    continue

                if block["skip"] or block_id in state["bypass"]:
                    block_state["status"] = "bypassed"
                    changed = True
                    continue

                if block["checkpoint"] and block_id not in self.released[sequence_id]:
                    block_state["status"] = "held"
                    continue

                block_state["status"] = "launching"
                launches.append(("block", entry, block, state["run_dir"]))
                changed = True

        statuses = [block_state["status"] for block_state in state["blocks"].values()]
        if any(status in ("running", "launching", "finalizing") for status in statuses):
            state["status"] = "running"
        elif any(status == "held" for status in statuses):
            state["status"] = "held"
        elif any(status == "pending" for status in statuses):
            state["status"] = "running"
        elif any(status in ("failed", "skipped", "cancelled", "finalize_failed") for status in statuses):
            state["status"] = "failed"
        else:
            state["status"] = "done"

        self._notify_ended(entry, state, blocks)

        return launches

    def _notify_ended(self, entry: dict, state: dict, blocks: dict) -> None:
        """Pushes one notification when an advance pass moved a sequence into a terminal state.

        A failed sequence reports every broken block with its error at high
        priority; a finished one reports the block count at default priority.
        The advance pass only runs on running or held sequences, so the
        transition into "failed" or "done" happens exactly once and the
        notification cannot repeat.

        Args:
            entry  : Sequence definition supplying the name.
            state  : Live state whose status was just recomputed.
            blocks : Current block definitions keyed by block id.
        """
        if state["status"] == "failed":
            broken = [f"{blocks[block_id]['title']}: {block_state['error'] or block_state['status']}" for block_id, block_state in state["blocks"].items() if block_state["status"] in ("failed", "finalize_failed", "cancelled")]
            self.notifier.system_alert(f"sequence failed: {entry['name']}", "; ".join(broken) or "a block was skipped without a recorded error")
        elif state["status"] == "done":
            self.notifier.system_alert(f"sequence finished: {entry['name']}", f"all {len(state['blocks'])} block(s) completed", "default")

    def _persist_states(self) -> None:
        """Mirrors every isolated sequence's live state into its workspace when it changed.

        The state file is what a later resume scan reads to know which blocks
        already finished, so it survives the server restart the in-memory
        state dies with. The staleness check and the write both run under the
        runner lock, and the write lands through a staging file replaced over
        the target, so the stop path and the watch scan can never interleave
        their rewrites and a reader never sees a half-written file.
        """
        with self.lock:
            for sequence_id, state in self.active.items():
                if not state["run_dir"]:
                    continue

                snapshot = {**state, "blocks": {block_id: self._copy_block_state(block) for block_id, block in state["blocks"].items()}}
                text     = json.dumps(snapshot, indent=2) + "\n"
                if self.persisted.get(sequence_id) == text:
                    continue

                path    = Path(state["run_dir"]) / self.STATE_FILENAME
                staging = path.with_suffix(".tmp")
                staging.write_text(text, encoding="utf-8")
                staging.replace(path)
                self.persisted[sequence_id] = text

    def scan(self) -> None:
        """Runs one advance pass over every running or held sequence, then launches what it marked.

        The advance pass is the only part holding the runner lock; the launches
        it collected run afterwards with the lock released, and each records its
        own outcome back into the live state, after which the workspace state
        files are refreshed.
        """
        with self.lock:
            launches = []
            for sequence_id, state in self.active.items():
                if state["status"] in ("running", "held"):
                    launches += self._advance(sequence_id, state)

        for launch in launches:
            if launch[0] == "block":
                self._launch_block(*launch[1:])
            else:
                self._launch_finalize_step(*launch[1:])

        self._persist_states()

    def start(self) -> None:
        """Starts the background polling thread that advances sequences forever."""
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()

    def _watch(self) -> None:
        """Scans on the fixed interval forever, logging any scan failure instead of exiting."""
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.scan()
            except Exception as exc:
                self.logger.error(f"sequence runner error: {exc}")
