
from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.inference.run_classifier import InferenceLayout


class RunSelector:

    def __init__(self, runs_dir: Path, marker: str, logger, action: str = "process") -> None:
        self.runs_dir = Path(runs_dir)
        self.marker   = marker
        self.logger   = logger
        self.action   = action

    def _discover(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Runs directory does not exist: {self.runs_dir}")

        checkpoints = self.runs_dir.rglob(self.marker)
        run_dirs    = sorted({path.parent for path in checkpoints if path.is_file()})

        if not run_dirs:
            raise FileNotFoundError(f"No '{self.marker}' found in any directory under {self.runs_dir}")

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        rows = []
        for index, run_dir in enumerate(run_dirs, start=1):
            checkpoint = run_dir / self.marker
            size_mb    = checkpoint.stat().st_size / (1024 * 1024)
            rows.append({
                "#"          : index,
                "Run"        : str(run_dir.relative_to(self.runs_dir)),
                "Checkpoint" : f"{size_mb:,.1f} MB",
            })

        self.logger.metrics_table(rows, columns=["#", "Run", "Checkpoint"], title=f"Runs under {self.runs_dir}")

    def _prompt(self, run_dirs: list[Path]) -> list[Path]:
        raw       = input(f"Select run(s) to {self.action} [1-{len(run_dirs)}, comma-separated, or 'all']: ").strip()
        selection = self._parse(raw, run_dirs)

        self.logger.ok(f"Selected {len(selection)} run(s): {', '.join(run_dir.name for run_dir in selection)}")
        return selection

    def _parse(self, raw: str, run_dirs: list[Path]) -> list[Path]:
        if raw == "" or raw.lower() in ("all", "*"):
            return run_dirs

        indices = []
        for token in raw.replace(",", " ").split():
            if not token.isdigit():
                raise ValueError(f"Invalid selection token '{token}'; expected run numbers 1-{len(run_dirs)} or 'all'")

            index = int(token)
            if index < 1 or index > len(run_dirs):
                raise ValueError(f"Selection {index} is out of range 1-{len(run_dirs)}")

            indices.append(index)

        ordered = sorted(dict.fromkeys(indices))
        return [run_dirs[index - 1] for index in ordered]

    def _nested(self, run_dirs: list[Path], name: str) -> list[Path]:
        nested = [run_dir for run_dir in run_dirs if run_dir.parent.name == name or str(run_dir.relative_to(self.runs_dir)).startswith(f"{name}/")]

        if nested:
            self.logger.ok(f"'{name}' expanded to {len(nested)} nested run(s)")
        return nested

    def _lookup(self, run_dirs: list[Path], names: list[str]) -> list[Path]:
        by_key    = {}
        ambiguous = {}
        for run_dir in run_dirs:
            for key in (run_dir.name, str(run_dir.relative_to(self.runs_dir))):
                if key in by_key and by_key[key] != run_dir:
                    ambiguous.setdefault(key, [by_key[key]]).append(run_dir)
                by_key[key] = run_dir

        for key in ambiguous:
            del by_key[key]

        selection = []
        for name in names:
            if name in ambiguous:
                candidates = ", ".join(str(run_dir.relative_to(self.runs_dir)) for run_dir in ambiguous[name])
                raise ValueError(f"run name '{name}' is ambiguous under {self.runs_dir}: {candidates}; pass the path relative to {self.runs_dir}")

            run_dir = by_key.get(name)
            if run_dir is not None:
                selection.append(run_dir)
                continue

            nested = self._nested(run_dirs, name)
            if not nested:
                raise FileNotFoundError(f"No run '{name}' with '{self.marker}' under {self.runs_dir}, and no nested runs beneath a directory of that name")
            selection.extend(nested)

        ordered = list(dict.fromkeys(selection))
        self.logger.ok(f"Selected {len(ordered)} run(s): {', '.join(str(run_dir.relative_to(self.runs_dir)) for run_dir in ordered)}")
        return ordered

    def select(self) -> list[Path]:
        run_dirs = self._discover()
        self._present(run_dirs)
        return self._prompt(run_dirs)

    def filter(self, names: list[str]) -> list[Path]:
        run_dirs = self._discover()
        self._present(run_dirs)
        return self._lookup(run_dirs, names)

    def all(self) -> list[Path]:
        run_dirs = self._discover()
        self._present(run_dirs)
        self.logger.ok(f"Selected all {len(run_dirs)} run(s)")
        return run_dirs

    def resolve(self, run_filter: list[str] | None) -> list[Path]:
        if run_filter:
            return self.filter(run_filter)
        if sys.stdin.isatty():
            return self.select()

        return self.all()


class ReportRunSelector(RunSelector):

    def __init__(self, runs_dir: Path, inference_dirname: str, report_filename: str, logger) -> None:
        super().__init__(runs_dir, inference_dirname, logger, action="collect")
        self.report_filename = report_filename

    def _reports_of(self, inference_dir: Path) -> list[Path]:
        reports = list(inference_dir.glob(f"*/{self.report_filename}"))

        for family in InferenceLayout.SUBDIRS.values():
            reports += list(inference_dir.glob(f"{family}/*/{self.report_filename}"))

        return sorted((path for path in reports if path.is_file()), key=lambda path: path.parent.name)

    def _discover(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Runs directory does not exist: {self.runs_dir}")

        inference_dirs = {path for path in self.runs_dir.rglob(self.marker) if path.is_dir()}
        run_dirs       = sorted(inference_dir.parent for inference_dir in inference_dirs if self._reports_of(inference_dir))

        if not run_dirs:
            raise FileNotFoundError(f"No '{self.report_filename}' found in any '{self.marker}' output under {self.runs_dir}")

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        rows = []
        for index, run_dir in enumerate(run_dirs, start=1):
            reports = self._reports_of(run_dir / self.marker)
            rows.append({
                "#"       : index,
                "Run"     : str(run_dir.relative_to(self.runs_dir)),
                "Reports" : f"{len(reports)} report(s), latest {reports[-1].parent.name}",
            })

        self.logger.metrics_table(rows, columns=["#", "Run", "Reports"], title=f"Runs under {self.runs_dir}")


class TensorboardRunSelector(RunSelector):

    EVENT_PATTERN = "events.out.tfevents.*"

    def __init__(self, runs_dir: Path, tensorboard_dirname: str, logger) -> None:
        super().__init__(runs_dir, tensorboard_dirname, logger, action="export")

    def _discover(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Runs directory does not exist: {self.runs_dir}")

        events   = self.runs_dir.rglob(f"{self.marker}/{self.EVENT_PATTERN}")
        run_dirs = sorted({path.parent.parent for path in events if path.is_file()})

        if not run_dirs:
            raise FileNotFoundError(f"No '{self.marker}' directory with event files found under {self.runs_dir}")

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        rows = []
        for index, run_dir in enumerate(run_dirs, start=1):
            events  = [path for path in (run_dir / self.marker).glob(self.EVENT_PATTERN) if path.is_file()]
            size_mb = sum(path.stat().st_size for path in events) / (1024 * 1024)
            rows.append({
                "#"      : index,
                "Run"    : str(run_dir.relative_to(self.runs_dir)),
                "Events" : f"{len(events)} file(s), {size_mb:,.1f} MB",
            })

        self.logger.metrics_table(rows, columns=["#", "Run", "Events"], title=f"Runs under {self.runs_dir}")
