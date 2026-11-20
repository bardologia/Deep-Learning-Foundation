
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from web_logger import WebLogger

from configuration.reference_layout import ReferenceLayout
from finalize_chain                 import FinalizeChain
from sequence_decisions             import TuningVerdict


class SequenceRunner:

    INTERVAL_S     = 2.0
    ENDED_OK       = ("finished",)
    ENDED_BAD      = ("failed", "cancelled")
    STATE_FILENAME = "sequence_state.json"

    def __init__(self, store, processes, launcher, resolver, notifier, logger: WebLogger) -> None:
        self.store     = store
        self.processes = processes
        self.launcher  = launcher
        self.resolver  = resolver
        self.chain     = FinalizeChain()
        self.verdict   = TuningVerdict()
        self.notifier  = notifier
        self.logger    = logger
        self.active    = {}
        self.released  = {}
        self.persisted = {}
        self.lock      = threading.Lock()

    def start_sequence(self, sequence_id: str, resume_dir: str = "", bypass: list | None = None) -> dict:
        entry = self.store.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}
        if not entry["blocks"]:
            return {"ok": False, "error": "the sequence has no blocks"}

        error = self._mail_preflight(entry["blocks"])
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

    def _mail_preflight(self, blocks: list) -> str:
        mailers  = [block["name"] or block["title"] for block in blocks if not block["skip"] and block["finalize"][self.chain.MAIL_STEP]]
        variable = self.chain.mail_blocked() if mailers else ""

        if not variable:
            return ""

        return f"block(s) {', '.join(mailers)} enable the report-mail finalize step, but the server environment has no {variable}; export it and restart the webui, or switch the mail step off"

    def resume_scan(self, sequence_id: str, resume_dir: str) -> dict:
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
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None:
                return None
            return {**state, "blocks": {block_id: self._copy_block_state(block) for block_id, block in state["blocks"].items()}}

    def states(self) -> dict:
        with self.lock:
            return {sequence_id: {**state, "blocks": {block_id: self._copy_block_state(block) for block_id, block in state["blocks"].items()}} for sequence_id, state in self.active.items()}

    @staticmethod
    def _fresh_block_state() -> dict:
        return {"status": "pending", "job_id": None, "error": "", "experiment_dir": "", "finalize": None, "verdict": None}

    @staticmethod
    def _copy_block_state(block_state: dict) -> dict:
        finalize = block_state["finalize"]
        verdict  = block_state["verdict"]
        return {**block_state, "finalize": dict(finalize) if finalize else None, "verdict": dict(verdict) if verdict else None}

    def _resume_workspace(self, entry: dict, resume_dir: str) -> tuple:
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
        return self.processes

    def _job_status(self, job_id: str) -> str:
        with self.processes.lock:
            record = self.processes.jobs.get(job_id)
            return record["status"] if record else "failed"

    def _launch_overrides(self, entry: dict, block: dict, interpreter: str, run_dir: str) -> tuple:
        overrides = dict(block["overrides"])
        dataset   = entry["dataset"]["dataset_path"]
        if not dataset and not run_dir and not any(block["finalize"].values()):
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
        reference = str(ReferenceLayout.RUNS_ROOT)

        if value == reference or value.startswith(f"{reference}/"):
            return True

        return any(value == root or value.startswith(f"{root}/") for root in self._runs_roots())

    def _runs_roots(self) -> tuple:
        roots = tuple(root for root in (str(self.store.paths.runs_dir),) if root.strip("/") not in ("", str(ReferenceLayout.RUNS_ROOT)))

        return roots

    def _launch_block(self, entry: dict, block: dict, run_dir: str) -> None:
        interpreter       = block["interpreter"] or self.launcher.preferred_interpreter(block["script"])
        overrides, leaves = self._launch_overrides(entry, block, interpreter, run_dir)

        if overrides is None:
            self._record_launch(entry, block, "", f"could not resolve the config of '{block['script']}' to inject the dataset and workspace")
            return

        experiment_dir = ""
        if any(block["finalize"].values()):
            try:
                experiment_dir = self.chain.experiment_dir(block["script"], overrides, leaves, run_dir)
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
        step = block_state["finalize"]["steps"][block_state["finalize"]["step"]]

        try:
            overrides = self.chain.step_overrides(step, block_state["experiment_dir"], run_dir)
        except FileNotFoundError as error:
            self._record_finalize_launch(entry, block, step, "", str(error))
            return

        result = self.launcher.execute(step, self.launcher.preferred_interpreter(step), overrides, None, False, False)
        if not result.get("ok"):
            self._record_finalize_launch(entry, block, step, "", result.get("error", "launch failed"))
            return

        self._record_finalize_launch(entry, block, step, result["job_id"], "")

    def _record_finalize_launch(self, entry: dict, block: dict, step: str, job_id: str, error: str) -> None:
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

    def _finish(self, block: dict, block_state: dict) -> None:
        if self.verdict.records(block):
            block_state["status"]  = "deciding"
            block_state["verdict"] = {"launching": False, "values": {}}
        else:
            block_state["status"] = "done"

    def _advance(self, sequence_id: str, state: dict) -> list:
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
            if status in self.ENDED_OK and any(blocks[block_id]["finalize"].values()):
                block_state["status"]   = "finalizing"
                block_state["finalize"] = {"steps": list(self.chain.steps(blocks[block_id]["finalize"])), "step": 0, "job_id": None, "launching": False}
            elif status in self.ENDED_OK:
                self._finish(blocks[block_id], block_state)
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
                    self._finish(blocks[block_id], block_state)
                else:
                    chain["launching"] = True
                    launches.append(("finalize", entry, blocks[block_id], self._copy_block_state(block_state), state["run_dir"]))

        for block_id, block_state in state["blocks"].items():
            if block_state["status"] == "deciding" and not block_state["verdict"]["launching"]:
                block_state["verdict"]["launching"] = True
                launches.append(("verdict", entry, blocks[block_id], state["run_dir"]))

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
                    if block_state["status"] != "held":
                        block_state["status"] = "held"
                        self.notifier.system_alert(f"sequence block held: {block['title']}", f"sequence '{entry['name']}': '{block['name'] or block['title']}' waits at its checkpoint; release it in the Sequences tab to continue", "default")
                    continue

                block_state["status"] = "launching"
                launches.append(("block", entry, block, state["run_dir"]))
                changed = True

        statuses = [block_state["status"] for block_state in state["blocks"].values()]
        if any(status in ("running", "launching", "finalizing", "deciding") for status in statuses):
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
        if state["status"] == "failed":
            broken = [f"{blocks[block_id]['title']}: {block_state['error'] or block_state['status']}" for block_id, block_state in state["blocks"].items() if block_state["status"] in ("failed", "finalize_failed", "cancelled")]
            self.notifier.system_alert(f"sequence failed: {entry['name']}", "; ".join(broken) or "a block was skipped without a recorded error")
        elif state["status"] == "done":
            self.notifier.system_alert(f"sequence finished: {entry['name']}", f"all {len(state['blocks'])} block(s) completed", "default")

    def _persist_states(self) -> None:
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

    def study_progress(self, sequence_id: str, block_id: str) -> dict:
        entry = self.store.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}

        block = next((item for item in entry["blocks"] if item["block_id"] == block_id), None)
        if block is None:
            return {"ok": False, "error": "block not found"}
        if block["script"] != self.verdict.SCRIPT:
            return {"ok": False, "error": f"block '{block['title']}' runs no tuning study"}

        run_tag = block["overrides"].get("run_tag", "")
        if run_tag in ("", "None"):
            return {"ok": False, "error": "the block needs a fixed 'run_tag' to locate its study"}

        with self.lock:
            state   = self.active.get(sequence_id)
            run_dir = state["run_dir"] if state is not None else ""

        if not run_dir:
            return {"ok": False, "error": "the sequence has not run in an isolated workspace, so there is no study to read"}

        files = {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(run_dir).glob(f"{run_tag}/*/{self.verdict.PROGRESS}"))}
        return {"ok": True, "studies": files}

    def _leaves(self, script: str, interpreter: str) -> dict:
        resolved = self.resolver.resolve(script, interpreter)
        if not resolved.get("ok"):
            raise ValueError(f"could not resolve the config of '{script}': {resolved.get('error', 'unknown error')}")

        return {leaf["path"]: str(leaf["value"]) for leaf in resolved["leaves"]}

    def _record_verdict(self, entry: dict, block: dict, run_dir: str) -> None:
        interpreter = block["interpreter"] or self.launcher.preferred_interpreter(block["script"])
        values      = {}
        error       = ""

        try:
            axes              = self.verdict.axes(block)
            overrides, leaves = self._launch_overrides(entry, block, interpreter, run_dir)
            if overrides is None:
                raise ValueError(f"could not resolve the config of '{block['script']}' to locate its best configuration")
            if not leaves:
                leaves = self._leaves(block["script"], interpreter)

            values = self.verdict.values(self.verdict.read(self.verdict.run_dir(overrides, leaves, run_dir)), axes)
            for axis, value in values.items():
                result = self.store.decide(entry["sequence_id"], block["block_id"], axis, value)
                if not result["ok"]:
                    raise ValueError(result["error"])
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            error = str(exc)

        with self.lock:
            state       = self.active.get(entry["sequence_id"])
            block_state = state["blocks"].get(block["block_id"]) if state else None
            stale       = block_state is None or block_state["status"] != "deciding"

            if not stale and error:
                block_state["status"] = "failed"
                block_state["error"]  = f"verdict not recorded: {error}"
            elif not stale:
                block_state["status"]            = "done"
                block_state["verdict"]["values"] = values

        if stale:
            self.logger.warning(f"sequence {entry['sequence_id']}: block '{block['title']}' left deciding while its verdict was read; nothing recorded")
        elif error:
            self.logger.error(f"sequence {entry['sequence_id']}: block '{block['title']}' failed to record its verdict: {error}")
            self.notifier.system_alert(f"sequence block failed: {block['title']}", f"sequence '{entry['name']}': verdict not recorded: {error}")
        else:
            recorded = ", ".join(f"{axis} = {value}" for axis, value in values.items())
            self.logger.ok(f"sequence {entry['sequence_id']}: block '{block['title']}' recorded {recorded}")
            self.notifier.system_alert(f"sequence block decided: {block['title']}", f"sequence '{entry['name']}': '{block['name'] or block['title']}' recorded {recorded}", "default")

    def scan(self) -> None:
        with self.lock:
            launches = []
            for sequence_id, state in self.active.items():
                if state["status"] in ("running", "held"):
                    launches += self._advance(sequence_id, state)

        for launch in launches:
            if launch[0] == "block":
                self._launch_block(*launch[1:])
            elif launch[0] == "verdict":
                self._record_verdict(*launch[1:])
            else:
                self._launch_finalize_step(*launch[1:])

        self._persist_states()

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.scan()
            except Exception as exc:
                self.logger.error(f"sequence runner error: {exc}")
