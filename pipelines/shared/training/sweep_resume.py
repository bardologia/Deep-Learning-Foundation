"""Reconciliation of a resumed sweep against the run directories on disk.

Resolves the pasted resume path to the sweep root and classifies every planned
unit as complete, unfinished or missing, so the launcher can reuse finished
runs, resume unfinished ones and train the rest from scratch.
"""

from __future__ import annotations

from pathlib import Path

from tools.runtime.completion import CompletionMarker


class SweepResumeRoot:
    """Resolves a pasted resume path to the root directory of a sweep."""

    @staticmethod
    def head_matches(dir_name: str, head: str) -> bool:
        """Returns whether a directory name equals the unit head or ends in it.

        Args:
            dir_name : Name of a directory under the sweep root.
            head     : First path component of a planned unit name.

        Returns:
            True when the directory is the head itself or the head carrying a
            composed label prefix.
        """
        return dir_name == head or dir_name.endswith(f"_{head}")

    @classmethod
    def matches(cls, root: Path, name: str) -> list[Path]:
        """Returns the directories under the root matching a planned unit name.

        Args:
            root : Sweep root directory.
            name : Planned unit name, possibly with nested path components.

        Returns:
            Existing directories whose first component matches the unit head,
            with or without a label prefix, and whose remaining components
            match exactly.
        """
        parts = Path(name).parts
        hits  = [child for child in sorted(root.iterdir()) if child.is_dir() and cls.head_matches(child.name, parts[0])]

        for part in parts[1:]:
            hits = [hit / part for hit in hits if (hit / part).is_dir()]

        return hits

    @classmethod
    def resolve(cls, resume_from: str, unit_names: list[str]) -> Path:
        """Resolves the resume path to the sweep root holding the planned units.

        Accepts either the sweep root itself or one of the run directories
        inside it, in which case the unit's path components are stripped off.

        Args:
            resume_from : Pasted path of the sweep folder or one of its runs.
            unit_names  : Names of the units planned by the current launch.

        Returns:
            The sweep root directory.

        Raises:
            FileNotFoundError : If the path does not exist.
            SystemExit        : If no planned unit matches anything under the path.
        """
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
    """Classifies the planned units of a resumed sweep against what is on disk.

    Attributes:
        EXCLUDED_DIRS : Directory names under the root that are never runs.
        root          : Sweep root directory.
        unit_names    : Names of the units planned by the current launch.
        completed     : Units with a completion marker, reused as they are.
        unfinished    : Units with a run directory but no completion marker.
        missing       : Units with no run directory at all.
        unplanned     : Directories under the root that no planned unit claims.
    """

    EXCLUDED_DIRS = ("batch_train_logs",)

    def __init__(self, root: Path, unit_names: list[str]) -> None:
        """Stores the sweep root and planned units with empty classifications."""
        self.root       = Path(root)
        self.unit_names = list(unit_names)
        self.completed  = []
        self.unfinished = []
        self.missing    = []
        self.unplanned  = []

    def _classify_planned(self) -> None:
        """Sorts each planned unit into completed, unfinished or missing."""
        for name in self.unit_names:
            hits = SweepResumeRoot.matches(self.root, name)

            if any(CompletionMarker.is_complete(hit) for hit in hits):
                self.completed.append(name)
            elif hits:
                self.unfinished.append(name)
            else:
                self.missing.append(name)

    def _collect_unplanned(self) -> None:
        """Records run directories under the root that no planned unit claims."""
        heads = {Path(name).parts[0] for name in self.unit_names}

        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir() or candidate.name in self.EXCLUDED_DIRS:
                continue

            if not any(SweepResumeRoot.head_matches(candidate.name, head) for head in heads):
                self.unplanned.append(candidate.name)

    def summary(self) -> dict:
        """Returns the reconciliation counts as a table-ready mapping."""
        return {
            "Sweep root" : str(self.root),
            "Complete"   : f"{len(self.completed)} (reused)",
            "Unfinished" : f"{len(self.unfinished)} (resume from checkpoint)",
            "Missing"    : f"{len(self.missing)} (train from scratch)",
            "Unplanned"  : f"{len(self.unplanned)} (left untouched)",
        }

    def rows(self) -> list[dict]:
        """Returns one row per planned unit with its resume status, in plan order."""
        status = {}
        status.update({name: "complete (reused)" for name in self.completed})
        status.update({name: "unfinished (resumes from checkpoint)" for name in self.unfinished})
        status.update({name: "missing (trains from scratch)" for name in self.missing})

        return [{"Run": name, "Status": status[name]} for name in self.unit_names]

    def log(self, logger) -> None:
        """Logs the reconciliation summary, the per-unit table and any unplanned runs."""
        logger.subsection("Sweep resume")
        logger.kv_table(self.summary(), title="Reconciliation")
        logger.metrics_table(self.rows(), columns=["Run", "Status"])

        for name in self.unplanned:
            logger.warning(f"{name}: present under the sweep root but not in the current plan; left untouched")

    def build(self) -> "SweepResumePlan":
        """Classifies the planned and unplanned runs and returns the plan itself."""
        self._classify_planned()
        self._collect_unplanned()
        return self
