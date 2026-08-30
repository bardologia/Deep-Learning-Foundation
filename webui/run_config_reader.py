"""Recovery of a finished run's resolved entry configuration for the launch pages.

Backs the "copy configs from a previous run" control: lists the run directories
under a root that carry a resolved config file, and reads one such file back as
dotted config paths with values rendered by the shared ``ConfigValueRenderer``,
exactly as the config resolver renders defaults, so the launch form can diff them
against its leaves. The identity and lifecycle fields that name the source run
are withheld, so copying a configuration never points the new launch at the run
it was copied from.
"""

from __future__ import annotations

import json

from pathlib import Path

from config_value_render import ConfigValueRenderer
from web_logger          import WebLogger


class RunConfigReader:
    """Finds and reads the resolved config files written into run directories.

    Attributes:
        CANDIDATES      : Config file locations searched inside a run directory, in priority order.
        PRUNED          : Directory names that hold a run's own artifacts and can never be a nested run root, so the walk does not descend into them.
        IDENTITY_LEAVES : Leaf names naming the source run's own directory or resume state; withheld from every read.
        MAX_DEPTH       : Deepest directory level below the root the walk reaches.
        logger          : Web logger used to record what each listing and read found.
    """

    CANDIDATES      = ("docs/resolved_entry_config.json", "pipeline/resolved_config.json", "resolved_config.json")
    PRUNED          = ("docs", "pipeline")
    IDENTITY_LEAVES = ("run_name", "resume", "resume_from")
    MAX_DEPTH       = 6

    def __init__(self, logger: WebLogger) -> None:
        """Stores the logger used to report listing and read results."""
        self.logger = logger

    def _config_file(self, run_dir: Path) -> Path | None:
        """Returns the first resolved config file present in a run directory, or None."""
        for relative in self.CANDIDATES:
            candidate = run_dir / relative
            if candidate.is_file():
                return candidate
        return None

    def _walk(self, directory: Path, depth: int):
        """Yields (run directory, config file) pairs below a root, recursing to the depth cap.

        A directory holding a config is both yielded and descended into: parameter
        runs live under ``<dataset>/params/<name>`` of an already preprocessed
        dataset, and benchmark units under a staged root that carries its own
        config, so stopping at the first hit would hide them.

        Args:
            directory : Directory whose children are inspected.
            depth     : Recursion level below the listing root.

        Yields:
            Tuples of (run directory, resolved config file inside it).
        """
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in self.PRUNED:
                continue

            found = self._config_file(entry)
            if found is not None:
                yield entry, found

            if depth < self.MAX_DEPTH:
                yield from self._walk(entry, depth + 1)

    def list_runs(self, raw_base: str) -> dict:
        """Lists every run below a root that holds a resolved config file.

        Args:
            raw_base : Absolute path of the run root to search.

        Returns:
            Dict with ``ok``, the resolved ``base`` and a ``runs`` list of entries
            carrying the run name relative to the base, its absolute path and the
            config file's path relative to the run directory; ``ok`` False with an
            error when the base is not a directory.
        """
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
        """Returns whether a dotted config path names the source run rather than its settings."""
        return path.rsplit(".", 1)[-1] in self.IDENTITY_LEAVES

    def read(self, raw_path: str) -> dict:
        """Reads the resolved config of one run as dotted paths with rendered values.

        The run's identity and lifecycle fields are split out into ``identity``
        instead of being served as copyable settings: applying them would point
        the new launch at the source run's own directory, where the resume gate
        either skips the launch or deletes the run that was copied from.

        Args:
            raw_path : Absolute path of the run directory.

        Returns:
            Dict with ``ok``, the ``run`` name, the ``source`` file relative to the
            run directory, the ``config`` mapping of copyable dotted config paths
            to rendered string values and the ``identity`` list of the withheld
            paths; ``ok`` False with an error when the directory holds no resolved
            config file or the file is unreadable.
        """
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
