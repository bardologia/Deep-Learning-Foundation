from __future__ import annotations

import re
from pathlib      import Path
from urllib.parse import quote

from catalog_roots import CatalogRoots
from web_logger    import WebLogger


class ResultsBrowser:

    IMAGE_SUFFIXES     = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
    ANIMATION_SUFFIXES = {".gif"}
    CONFIG_SUFFIXES    = {".json", ".yaml", ".yml", ".toml", ".ini"}
    MARKDOWN_SUFFIXES  = {".md"}
    LOG_SUFFIXES       = {".log", ".txt", ".out"}

    ANSI_ESCAPES = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")

    SKIPPED_DIRS   = {"__pycache__", ".git", ".ipynb_checkpoints"}
    MAX_DEPTH      = 10
    MAX_TEXT_BYTES = 262144

    STAGE_MARKERS = (
        ("benchmark",        ("pipeline/resolved_config.json", "results/benchmark_overview.md", "benchmark_overview.md")),
        ("preprocess",       ("images/slc", "images/interferograms", "images/dem")),
        ("param extraction", ("images/colormaps", "images/example_fits", "fit_report.md")),
        ("training",         ("docs/trainer_config.json", "last.pt", "tensorboard")),
        ("inference",        ("inference", "figures", "animations")),
    )

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.roots  = CatalogRoots()

    def tree(self, raw_path: str) -> dict:
        root, error = self.roots.open(raw_path, "an absolute path is required")
        if error:
            return {"ok": False, "error": error}

        self.logger.info(f"results: opened {root}")

        return {
            "ok"    : True,
            "root"  : str(root),
            "name"  : root.name,
            "stage" : self._stage(root),
            "tree"  : self._node(root, root, 0),
        }

    def folder(self, raw_root: str, rel: str) -> dict:
        if not self.roots.known(raw_root):
            return {"ok": False, "error": "path not opened"}

        folder = (Path(raw_root) / rel).resolve() if rel else Path(raw_root)
        if not folder.is_relative_to(raw_root) or not folder.is_dir():
            return {"ok": False, "error": "unknown folder"}

        markdown, images, animations, configs, logs, other = [], [], [], [], [], []

        for entry in sorted(folder.iterdir()):
            if not entry.is_file():
                continue

            suffix = entry.suffix.lower()

            if suffix in self.MARKDOWN_SUFFIXES:
                markdown.append({"name": entry.name, "text": self._read_text(entry)})
            elif suffix in self.IMAGE_SUFFIXES:
                images.append({"name": entry.stem, "url": self._url(entry)})
            elif suffix in self.ANIMATION_SUFFIXES:
                animations.append({"name": entry.stem, "url": self._url(entry)})
            elif suffix in self.CONFIG_SUFFIXES:
                configs.append({"name": entry.name, "kind": suffix.lstrip("."), "text": self._read_text(entry)})
            elif suffix in self.LOG_SUFFIXES:
                logs.append({"name": entry.name, "size": entry.stat().st_size, "text": self._read_log(entry)})
            else:
                other.append({"name": entry.name, "size": entry.stat().st_size})

        return {
            "ok"         : True,
            "root"       : raw_root,
            "rel"        : rel,
            "abs"        : str(folder),
            "markdown"   : markdown,
            "images"     : images,
            "animations" : animations,
            "configs"    : configs,
            "logs"       : logs,
            "other"      : other,
        }

    def catalog(self, datasets_raw: str, logs_raw: str) -> dict:
        return {
            "ok"       : True,
            "datasets" : self._catalog_datasets(datasets_raw),
            "runs"     : self._catalog_runs(logs_raw),
        }

    def gallery(self, raw_root: str) -> dict:
        if not self.roots.known(raw_root):
            return {"ok": False, "error": "path not opened"}

        root   = Path(raw_root)
        groups = []
        self._collect_gallery(root, root, 0, groups)

        total = sum(len(group["images"]) for group in groups)
        return {"ok": True, "root": raw_root, "total": total, "groups": groups}

    def file_path(self, raw_path: str) -> Path | None:
        target = Path(raw_path).resolve()
        if not self.roots.contains(target):
            return None
        if not target.is_file():
            return None
        return target

    def _catalog_datasets(self, raw: str) -> dict:
        root, error = self.roots.resolve(raw, "not set")
        if error:
            return {"error": error, "items": []}

        items = []
        for entry in self._subdirs(root):
            params     = []
            params_dir = entry / "params"
            if params_dir.is_dir():
                params = [{"name": child.name, "path": str(child)} for child in self._subdirs(params_dir)]
            items.append({"name": entry.name, "path": str(entry), "params": params})

        return {"error": "", "items": items}

    def _catalog_runs(self, raw: str) -> dict:
        root, error = self.roots.resolve(raw, "not set")
        if error:
            return {"error": error, "items": []}

        dirs = self._subdirs(root)
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        return {"error": "", "items": [{"name": d.name, "path": str(d), "stage": self._stage(d)} for d in dirs]}

    def _subdirs(self, directory: Path) -> list:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            entries = []
        return [entry for entry in entries if entry.is_dir() and entry.name not in self.SKIPPED_DIRS and not entry.name.startswith(".")]

    def _collect_gallery(self, directory: Path, root: Path, depth: int, groups: list) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            entries = []

        images = []
        for entry in entries:
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix in self.IMAGE_SUFFIXES:
                images.append({"name": entry.stem, "url": self._url(entry), "kind": "img"})
            elif suffix in self.ANIMATION_SUFFIXES:
                images.append({"name": entry.stem, "url": self._url(entry), "kind": "gif"})

        if images:
            rel = "" if directory == root else str(directory.relative_to(root))
            groups.append({"rel": rel, "images": images})

        for entry in entries:
            if entry.is_dir() and entry.name not in self.SKIPPED_DIRS and not entry.name.startswith(".") and depth < self.MAX_DEPTH:
                self._collect_gallery(entry, root, depth + 1, groups)

    def _node(self, directory: Path, root: Path, depth: int) -> dict:
        counts   = {"markdown": 0, "images": 0, "animations": 0, "configs": 0, "logs": 0, "other": 0}
        children = []

        try:
            entries = sorted(directory.iterdir())
        except OSError:
            entries = []

        for entry in entries:
            if entry.is_dir():
                if entry.name in self.SKIPPED_DIRS or entry.name.startswith("."):
                    continue
                if depth < self.MAX_DEPTH:
                    children.append(self._node(entry, root, depth + 1))
                continue

            suffix = entry.suffix.lower()

            if suffix in self.MARKDOWN_SUFFIXES:
                counts["markdown"] += 1
            elif suffix in self.IMAGE_SUFFIXES:
                counts["images"] += 1
            elif suffix in self.ANIMATION_SUFFIXES:
                counts["animations"] += 1
            elif suffix in self.CONFIG_SUFFIXES:
                counts["configs"] += 1
            elif suffix in self.LOG_SUFFIXES:
                counts["logs"] += 1
            else:
                counts["other"] += 1

        rel = "" if directory == root else str(directory.relative_to(root))

        return {
            "name"     : directory.name,
            "rel"      : rel,
            "counts"   : counts,
            "children" : children,
        }

    def _stage(self, root: Path) -> str:
        for stage, markers in self.STAGE_MARKERS:
            if any((root / marker).exists() for marker in markers):
                return stage
        return "results"

    def _read_text(self, target: Path) -> str:
        try:
            raw = target.read_bytes()
        except OSError:
            return ""

        text = raw[: self.MAX_TEXT_BYTES].decode("utf-8", errors="replace")
        if len(raw) > self.MAX_TEXT_BYTES:
            text += "\n\n[truncated]"
        return text

    def _read_log(self, target: Path) -> str:
        try:
            raw = target.read_bytes()
        except OSError:
            return ""

        if len(raw) <= self.MAX_TEXT_BYTES:
            return self._clean_log(raw.decode("utf-8", errors="replace"))

        head = self._clean_log(raw[: self.MAX_TEXT_BYTES].decode("utf-8", errors="replace"))
        return head + f"\n\n[truncated: showing the first {self.MAX_TEXT_BYTES // 1024} KB of {len(raw) // 1024} KB]"

    def _clean_log(self, text: str) -> str:
        text = self.ANSI_ESCAPES.sub("", text.replace("\r\n", "\n"))

        lines = [line.rsplit("\r", 1)[-1] if "\r" in line else line for line in text.split("\n")]
        text  = "\n".join(lines)

        return "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)

    def _url(self, target: Path) -> str:
        return "/resultsmedia?path=" + quote(str(target))
