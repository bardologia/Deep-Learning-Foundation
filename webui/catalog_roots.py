from __future__ import annotations

import threading
from pathlib import Path


class CatalogRoots:

    NOT_SET = "set the runs directory in the Results tab first"

    def __init__(self) -> None:
        self.lock  = threading.Lock()
        self.roots = set()

    @staticmethod
    def resolve(raw: str, empty_error: str = NOT_SET) -> tuple[Path | None, str]:
        raw = (raw or "").strip()
        if not raw:
            return None, empty_error

        root = Path(raw).expanduser()
        if not root.is_absolute():
            return None, "an absolute path is required"

        root = root.resolve()
        if not root.is_dir():
            return None, f"not a directory: {root}"

        return root, ""

    def add(self, root: Path) -> None:
        with self.lock:
            self.roots.add(str(root))

    def known(self, raw: str) -> bool:
        return raw in self.snapshot()

    def contains(self, target: Path) -> bool:
        return any(target.is_relative_to(root) for root in self.snapshot())

    def enclosing(self, target: Path) -> Path | None:
        matches = [root for root in self.snapshot() if target.is_relative_to(root)]
        if not matches:
            return None
        return Path(max(matches, key=len))

    def snapshot(self) -> tuple:
        with self.lock:
            return tuple(self.roots)

    def open(self, raw: str, empty_error: str = NOT_SET) -> tuple[Path | None, str]:
        root, error = self.resolve(raw, empty_error)
        if error:
            return None, error

        self.add(root)
        return root, ""


class RunScanner:

    STAMP_MARKER = "inference/*/cubes/pred_curves.npy"

    def __init__(self, roots: CatalogRoots) -> None:
        self.roots = roots

    @staticmethod
    def _entry(root: Path, run_dir: Path, stamp: str, target: Path) -> dict:
        return {
            "id"    : str(target),
            "run"   : run_dir.name,
            "group" : str(run_dir.relative_to(root).parent),
            "stamp" : stamp,
        }

    def stamps(self, base: str, required: tuple[str, ...] = ()) -> dict:
        root, error = self.roots.open(base)
        if error:
            return {"ok": False, "error": error, "entries": []}

        entries = []
        for marker in sorted(root.rglob(self.STAMP_MARKER)):
            stamp_dir = marker.parent.parent
            if any(not (stamp_dir / rel).is_file() for rel in required):
                continue

            run_dir = stamp_dir.parent.parent
            entries.append(self._entry(root, run_dir, stamp_dir.name, stamp_dir))

        entries.sort(key=lambda entry: entry["id"], reverse=True)
        return {"ok": True, "root": str(root), "entries": entries}

    def checkpoint_runs(self, base: str, checkpoint_name: str, config_name: str) -> dict:
        root, error = self.roots.open(base)
        if error:
            return {"ok": False, "error": error, "entries": []}

        entries = []
        for marker in sorted(root.rglob(checkpoint_name)):
            run_dir = marker.parent
            if not (run_dir / "meta" / config_name).is_file():
                continue

            entries.append(self._entry(root, run_dir, "", run_dir))

        entries.sort(key=lambda entry: entry["id"], reverse=True)
        return {"ok": True, "root": str(root), "entries": entries}
