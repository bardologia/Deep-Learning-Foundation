
from __future__ import annotations

import json
import re
from pathlib import Path

from web_logger import WebLogger


class DatasetBrowser:

    SEED_DIR        = re.compile(r"seed\d+")
    LAYOUT_FILE     = "dataset.json"
    CHECKPOINT_NAME = "best_model.pt"
    INFERENCE_DIR   = "inference"
    RUN_MARKER      = "meta"
    RUN_MAX_DEPTH   = 6

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger

    def datasets(self, raw_base: str) -> dict:
        base = self._directory(raw_base)
        if base is None:
            return {"ok": False, "error": f"not a directory: {raw_base}"}

        entries = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            layout = entry / self.LAYOUT_FILE
            item   = {"name": entry.name, "path": str(entry), "is_dataset": layout.is_file()}

            if item["is_dataset"]:
                payload         = json.loads(layout.read_text(encoding="utf-8"))
                item["inputs"]  = len(payload["input_channels"])
                item["targets"] = len(payload["target_channels"])

            entries.append(item)

        self.logger.info(f"datasets: listed {len(entries)} under {base}")
        return {"ok": True, "base": str(base), "datasets": entries}

    def runs(self, raw_bases: list[str], seed_units: bool = False, split: str = "") -> dict:
        roots = [resolved for resolved in (self._directory(raw) for raw in raw_bases) if resolved is not None]
        if not roots:
            return {"ok": False, "error": f"no run roots: {raw_bases}"}

        entries = []
        for base in roots:
            base_entries = []
            for run_dir in self._run_dirs(base):
                base_entries.append({
                    "name"           : self._run_name(base, run_dir),
                    "path"           : str(run_dir),
                    "has_checkpoint" : (run_dir / self.CHECKPOINT_NAME).is_file(),
                    "has_inference"  : self._has_inference(run_dir, split),
                })

            entries += self._with_seed_units(base, base_entries) if seed_units else base_entries

        base_label = ", ".join(str(root) for root in roots)
        self.logger.info(f"runs: listed {len(entries)} under {base_label}")
        return {"ok": True, "base": base_label, "runs": entries}

    def _with_seed_units(self, base: Path, run_entries: list[dict]) -> list[dict]:
        units = {}
        for entry in run_entries:
            run_dir = Path(entry["path"])
            if self.SEED_DIR.fullmatch(run_dir.name) is None or run_dir.parent == base:
                continue

            parent = str(run_dir.parent)
            if parent not in units:
                units[parent] = {
                    "name"           : str(run_dir.parent.relative_to(base)),
                    "path"           : parent,
                    "has_checkpoint" : True,
                    "has_inference"  : True,
                    "own_inference"  : self._has_inference(run_dir.parent),
                    "n_seeds"        : 0,
                }

            unit = units[parent]
            unit["has_checkpoint"] = unit["has_checkpoint"] and entry["has_checkpoint"]
            unit["has_inference"]  = unit["has_inference"] and entry["has_inference"]
            unit["n_seeds"]       += 1

        merged  = []
        emitted = set()
        for entry in run_entries:
            parent = str(Path(entry["path"]).parent)
            if parent in units and parent not in emitted:
                merged.append(units[parent])
                emitted.add(parent)
            merged.append(entry)

        return merged

    def run_groups(self, raw_bases: list[str]) -> dict:
        roots = [resolved for resolved in (self._directory(raw) for raw in raw_bases) if resolved is not None]
        if not roots:
            return {"ok": False, "error": f"no run roots: {raw_bases}"}

        entries = []
        for base in roots:
            groups: dict[Path, int] = {}
            for run_dir in self._run_dirs(base):
                if run_dir.parent != base:
                    groups[run_dir.parent] = groups.get(run_dir.parent, 0) + 1

            for group_dir, n_runs in sorted(groups.items()):
                entries.append({
                    "name"   : str(group_dir.relative_to(base)),
                    "path"   : str(group_dir),
                    "n_runs" : n_runs,
                })

        base_label = ", ".join(str(root) for root in roots)
        self.logger.info(f"run_groups: listed {len(entries)} under {base_label}")
        return {"ok": True, "base": base_label, "groups": entries}

    def experiments(self, raw_base: str, marker: str) -> dict:
        from tools.reporting.report_mailer import ExperimentScanner
        base = self._directory(raw_base)
        if base is None:
            return {"ok": False, "error": f"not a directory: {raw_base}"}
        if not marker.strip():
            return {"ok": False, "error": "no comparison dirname given"}

        scanner = ExperimentScanner(base)

        entries = []
        for experiment in scanner.scan():
            comparisons = experiment.comparison_root(marker)
            entries.append({
                "name"          : scanner.name(experiment),
                "path"          : str(experiment.root),
                "kind"          : experiment.KIND,
                "n_trials"      : len(experiment.trials()),
                "n_comparisons" : sum(1 for child in comparisons.iterdir() if child.is_dir()) if comparisons.is_dir() else 0,
            })

        self.logger.info(f"experiments: listed {len(entries)} at or below {base}")
        return {"ok": True, "base": str(base), "experiments": entries}

    def _run_dirs(self, base: Path):
        if (base / self.RUN_MARKER).is_dir():
            yield base
            return

        yield from self._walk_runs(base, 0)

    def _run_name(self, base: Path, run_dir: Path) -> str:
        return run_dir.name if run_dir == base else str(run_dir.relative_to(base))

    def _walk_runs(self, directory: Path, depth: int):
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            if (entry / self.RUN_MARKER).is_dir():
                yield entry
            elif depth < self.RUN_MAX_DEPTH:
                yield from self._walk_runs(entry, depth + 1)

    def _has_inference(self, run_dir: Path, split: str = "") -> bool:
        from pipelines.shared.inference.metadata import CompletedInference

        inference_dir = run_dir / self.INFERENCE_DIR
        if not inference_dir.is_dir():
            return False

        if not split:
            return any(entry.is_dir() for entry in inference_dir.iterdir())

        return CompletedInference.latest(inference_dir, split) is not None

    def _directory(self, raw: str) -> Path | None:
        if not raw or not raw.strip():
            return None

        path = Path(raw).expanduser()
        if not path.is_absolute():
            return None

        path = path.resolve()
        return path if path.is_dir() else None
