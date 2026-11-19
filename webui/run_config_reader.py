
from __future__ import annotations

import json

from pathlib import Path

from config_value_render import ConfigValueRenderer
from web_logger          import WebLogger


class RunConfigReader:

    CANDIDATES      = ("docs/resolved_entry_config.json", "pipeline/resolved_config.json", "resolved_config.json")
    PRUNED          = ("docs", "pipeline")
    IDENTITY_LEAVES = ("run_name", "resume", "resume_from")
    MAX_DEPTH       = 6

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger

    def _config_file(self, run_dir: Path) -> Path | None:
        for relative in self.CANDIDATES:
            candidate = run_dir / relative
            if candidate.is_file():
                return candidate
        return None

    def _walk(self, directory: Path, depth: int):
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in self.PRUNED:
                continue

            found = self._config_file(entry)
            if found is not None:
                yield entry, found

            if depth < self.MAX_DEPTH:
                yield from self._walk(entry, depth + 1)

    def list_runs(self, raw_base: str) -> dict:
        base = Path(raw_base).expanduser() if raw_base else None
        if base is None or not base.is_dir():
            return {"ok": False, "error": f"not a directory: {raw_base}"}

        entries = []
        for run_dir, found in self._walk(base, 0):
            entries.append({
                "name"   : str(run_dir.relative_to(base)),
                "path"   : str(run_dir),
                "source" : str(found.relative_to(run_dir)),
            })

        self.logger.info(f"run configs: listed {len(entries)} under {base}")
        return {"ok": True, "base": str(base), "runs": entries}

    def _is_identity(self, path: str) -> bool:
        return path.rsplit(".", 1)[-1] in self.IDENTITY_LEAVES

    def read(self, raw_path: str) -> dict:
        run_dir = Path(raw_path).expanduser() if raw_path else None
        if run_dir is None or not run_dir.is_dir():
            return {"ok": False, "error": f"not a run directory: {raw_path}"}

        found = self._config_file(run_dir)
        if found is None:
            return {"ok": False, "error": f"{run_dir.name} holds no resolved config (looked for {', '.join(self.CANDIDATES)}); only runs launched after configs were persisted can be copied"}

        try:
            mapping = json.loads(found.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"could not read {found}: {exc}"}

        if not isinstance(mapping, dict):
            return {"ok": False, "error": f"{found} does not hold a config mapping"}

        rendered = {str(path): ConfigValueRenderer.render(value) for path, value in mapping.items() if not self._is_identity(str(path))}
        identity = sorted(str(path) for path in mapping if self._is_identity(str(path)))

        self.logger.info(f"run configs: read {len(rendered)} fields from {found}, {len(identity)} identity fields withheld")
        return {"ok": True, "run": run_dir.name, "source": str(found.relative_to(run_dir)), "config": rendered, "identity": identity}


class ResumeRecording:

    MAX_CLIMB = 8

    @staticmethod
    def locate(raw_path: str) -> Path | None:
        from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
        from pipelines.shared.orchestration.resumed_run    import StagedRunLocator, TuningRunLocator
        from pipelines.shared.training.resume_source       import SweepRecording

        target = Path(raw_path).expanduser() if raw_path else None
        if target is None or not target.is_dir():
            return None

        own = ResolvedEntryConfigWriter.path_for(target)
        if own.is_file():
            return own

        for candidate in (target, *list(target.parents)[: ResumeRecording.MAX_CLIMB]):
            for probe in (StagedRunLocator.config_path(candidate), SweepRecording.path(candidate), TuningRunLocator.config_path(candidate)):
                if probe.is_file():
                    return probe

        return None

    @staticmethod
    def values(raw_path: str) -> dict:
        found = ResumeRecording.locate(raw_path)
        if found is None:
            return {}

        try:
            mapping = json.loads(found.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

        if not isinstance(mapping, dict):
            return {}

        return {str(path): ConfigValueRenderer.render(value) for path, value in mapping.items()}
