
from __future__ import annotations

from pathlib import Path

from pipelines.shared.inference.metadata import CompletedInference
from tools.data.io                       import FileIO
from tools.runtime.completion            import CompletionMarker
from tools.training.checkpoint           import TrainerState


class SweepResumeRoot:

    @staticmethod
    def head_matches(dir_name: str, head: str) -> bool:
        return dir_name == head or dir_name.endswith(f"_{head}")

    @classmethod
    def matches(cls, root: Path, name: str) -> list[Path]:
        parts = Path(name).parts
        hits  = [child for child in sorted(root.iterdir()) if child.is_dir() and cls.head_matches(child.name, parts[0])]

        for part in parts[1:]:
            hits = [hit / part for hit in hits if (hit / part).is_dir()]

        return hits

    @classmethod
    def resolve(cls, resume_from: str, unit_names: list[str]) -> Path:
        target = Path(resume_from).expanduser()

        if not target.is_dir():
            raise FileNotFoundError(f"resume_from points at '{target}', which does not exist; paste the sweep folder holding the runs, or one of its run directories")

        for name in unit_names:
            parts = Path(name).parts

            if len(target.parts) <= len(parts):
                continue

            tail = target.parts[len(target.parts) - len(parts):]

            if tail[1:] == tuple(parts[1:]) and cls.head_matches(tail[0], parts[0]):
                return Path(*target.parts[:len(target.parts) - len(parts)])

        if any(cls.matches(target, name) for name in unit_names):
            return target

        raise SystemExit(f"resume_from '{target}' holds no run directory matching any of the {len(unit_names)} planned units (e.g. '{unit_names[0]}', matched with or without the composed label prefix); paste the sweep folder or one of its runs, with the entry settings matching the original launch")


class SweepResumePlan:

    EXCLUDED_DIRS = ("batch_train_logs",)

    def __init__(self, root: Path, unit_names: list[str], split: str | None = None) -> None:
        self.root       = Path(root)
        self.unit_names = list(unit_names)
        self.split      = split
        self.completed  = []
        self.resumable  = []
        self.restarting = []
        self.missing    = []
        self.inferred   = []
        self.unplanned  = []

    @property
    def unfinished(self) -> list[str]:
        pending = set(self.resumable) | set(self.restarting)

        return [name for name in self.unit_names if name in pending]

    def _has_inference(self, hits: list[Path]) -> bool:
        return any(CompletedInference.stamps(hit / "inference", self.split) for hit in hits)

    def _classify_planned(self) -> None:
        for name in self.unit_names:
            hits = SweepResumeRoot.matches(self.root, name)

            if any(CompletionMarker.is_complete(hit) for hit in hits):
                self.completed.append(name)
                if self.split is not None and self._has_inference(hits):
                    self.inferred.append(name)
            elif any(TrainerState.path(hit).is_file() for hit in hits):
                self.resumable.append(name)
            elif hits:
                self.restarting.append(name)
            else:
                self.missing.append(name)

    def _collect_unplanned(self) -> None:
        heads = {Path(name).parts[0] for name in self.unit_names}

        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir() or candidate.name in self.EXCLUDED_DIRS:
                continue

            if not any(SweepResumeRoot.head_matches(candidate.name, head) for head in heads):
                self.unplanned.append(candidate.name)

    def summary(self) -> dict:
        complete = f"{len(self.completed)} (reused)"
        if self.split is not None:
            complete += f", {len(self.completed) - len(self.inferred)} of them without '{self.split}' inference"

        return {
            "Sweep root" : str(self.root),
            "Complete"   : complete,
            "Unfinished" : f"{len(self.resumable)} with a trainer state (resume from {TrainerState.FILENAME}), {len(self.restarting)} without (restart from scratch)",
            "Missing"    : f"{len(self.missing)} (train from scratch)",
            "Unplanned"  : f"{len(self.unplanned)} (left untouched)",
        }

    def columns(self) -> list[str]:
        return ["Run", "Status", "Inference"] if self.split is not None else ["Run", "Status"]

    def rows(self) -> list[dict]:
        status = {}
        status.update({name: "complete (reused)" for name in self.completed})
        status.update({name: f"unfinished, trainer state saved (resumes from {TrainerState.FILENAME})" for name in self.resumable})
        status.update({name: "unfinished, no trainer state (restarts from scratch)" for name in self.restarting})
        status.update({name: "missing (trains from scratch)" for name in self.missing})

        rows = []
        for name in self.unit_names:
            row = {"Run": name, "Status": status[name]}

            if self.split is not None:
                row["Inference"] = f"{self.split} {'done' if name in self.inferred else 'missing'}" if name in self.completed else "-"

            rows.append(row)

        return rows

    def _log_units(self, logger) -> None:
        logger.metrics_table(self.rows(), columns=self.columns())

        for name in self.unplanned:
            logger.warning(f"{name}: present under the sweep root but not in the current plan; left untouched")

    def log(self, logger) -> None:
        logger.subsection("Sweep resume")
        logger.kv_table(self.summary(), title="Reconciliation")
        self._log_units(logger)

    def warn_existing_root(self, logger, unit_resume: bool, trainer_resume: bool) -> None:
        found = len(self.completed) + len(self.unfinished)
        if found == 0:
            return

        completed  = f"{len(self.completed)} completed unit(s) are reused without training" if unit_resume else f"{len(self.completed)} completed unit(s) refuse to train over their finished runs and fail (resume=False)"
        unfinished = f"{len(self.unfinished)} unfinished unit(s) continue from {TrainerState.FILENAME} where it exists (training.resume=True)" if trainer_resume else f"{len(self.unfinished)} unfinished unit(s) are deleted and restart from scratch (training.resume=False)"

        logger.warning(f"{self.root} already holds runs of {found} planned unit(s) from an earlier launch although resume_from was not given: {completed}; {unfinished}. Pass resume_from={self.root} to resume the sweep explicitly, or another logdir or run_name for a fresh one")
        self._log_units(logger)

    def build(self) -> "SweepResumePlan":
        self._classify_planned()
        self._collect_unplanned()
        return self


class SweepResults:

    @staticmethod
    def merge(results: list[dict], path: Path, logger) -> list[dict]:
        path = Path(path)
        if not path.is_file():
            return list(results)

        earlier  = FileIO.load_json(path)
        by_name  = {record["name"]: record for record in results}
        replaced = [record["name"] for record in earlier if record["name"] in by_name]

        merged = []
        for record in earlier:
            merged.append(by_name.pop(record["name"]) if record["name"] in by_name else record)
        merged.extend(by_name.values())

        logger.warning(f"{path.name} already holds {len(earlier)} entry(ies) from an earlier launch: {len(earlier) - len(replaced)} kept, {len(replaced)} replaced by this launch's results, {len(merged) - len(earlier)} added")

        return merged
