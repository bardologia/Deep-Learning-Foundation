
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing  import List, Tuple

from configuration.diagnostics                 import ArtifactMailConfig
from pipelines.shared.dataset.dataset_files    import DatasetFiles
from pipelines.shared.inference.run_classifier import RunDirectoryWalk
from tools.data.archives                       import ArchivePacker
from tools.runtime.run_selector                import RunSelector


class DatasetSelector(RunSelector):

    def __init__(self, datasets_dir: Path, logger) -> None:
        super().__init__(datasets_dir, DatasetFiles.LAYOUT, logger, action="mail")

    def _discover(self) -> list[Path]:
        return sorted(child for child in self.runs_dir.iterdir() if child.is_dir() and (child / self.marker).is_file())

    def _present(self, run_dirs: list[Path]) -> None:
        rows = []
        for index, dataset_dir in enumerate(run_dirs, start=1):
            files = DatasetFiles.read(dataset_dir)
            rows.append({
                "#"       : index,
                "Dataset" : dataset_dir.name,
                "Grid"    : f"{files.rows} x {files.cols}",
                "Inputs"  : str(len(files.input_channels)),
                "Targets" : str(len(files.target_channels)),
            })

        self.logger.metrics_table(rows, columns=["#", "Dataset", "Grid", "Inputs", "Targets"], title=f"Datasets under {self.runs_dir}")


class RunContentSelector(RunSelector):

    SEARCH_ENTRY = "train_backbone"

    def __init__(self, runs_dir: Path, logger) -> None:
        super().__init__(runs_dir, RunDirectoryWalk.MARKER, logger, action="mail")

    def _discover(self) -> list[Path]:
        if (self.runs_dir / self.marker).is_dir():
            return [self.runs_dir]

        run_dirs = sorted(RunDirectoryWalk.walk(self.runs_dir))

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        self.logger.info(f"Scanning {len(run_dirs)} run(s) under {self.runs_dir} for their file counts and sizes")

        rows = []
        with self.logger.track(transient=True) as progress:
            task = progress.add_task("Scanning runs", total=len(run_dirs))

            for index, run_dir in enumerate(run_dirs, start=1):
                files = [path for path in run_dir.rglob("*") if path.is_file()]
                size  = sum(path.stat().st_size for path in files)
                rows.append({
                    "#"       : index,
                    "Run"     : self._name(run_dir),
                    "Files"   : str(len(files)),
                    "On disk" : f"{size / (1024 * 1024):,.1f} MB",
                })
                progress.advance(task)

        self.logger.metrics_table(rows, columns=["#", "Run", "Files", "On disk"], title=f"Runs at or below {self.runs_dir}")

    def _name(self, run_dir: Path) -> str:
        return run_dir.name if run_dir == self.runs_dir else str(run_dir.relative_to(self.runs_dir))


class ExcludedContent:

    NAMED_LIMIT = 20

    def __init__(self, source_dir: Path, patterns: List[str], suffixes: List[str], max_bytes: int, logger) -> None:
        self.source_dir = Path(source_dir)
        self.patterns   = list(patterns)
        self.suffixes   = [suffix.lower() for suffix in suffixes]
        self.max_bytes  = int(max_bytes)
        self.logger     = logger

        self.dropped = []

    def _path_rule(self, relative: Path) -> str:
        candidates = [relative] + [parent for parent in relative.parents if parent != Path(".")]

        for candidate in candidates:
            for pattern in self.patterns:
                if fnmatch(candidate.as_posix(), pattern) or fnmatch(candidate.name, pattern):
                    return f"path {pattern}"

        return ""

    def _suffix_rule(self, relative: Path) -> str:
        suffix = relative.suffix.lower()

        return f"suffix {suffix}" if suffix in self.suffixes else ""

    def _size_rule(self, path: Path) -> str:
        if not self.max_bytes:
            return ""

        return self._size_label() if path.stat().st_size > self.max_bytes else ""

    def _rule(self, path: Path, relative: Path) -> str:
        pattern = self._path_rule(relative)
        if pattern:
            return pattern

        suffix = self._suffix_rule(relative)
        if suffix:
            return suffix

        return self._size_rule(path)

    def _report(self) -> None:
        if not self.dropped:
            return

        groups = {}
        for _relative, size, rule in self.dropped:
            count, total = groups.get(rule, (0, 0))
            groups[rule] = (count + 1, total + size)

        rows = [{"Rule": rule, "Files": str(count), "Size": f"{total / (1024 * 1024):,.1f} MB"} for rule, (count, total) in sorted(groups.items())]
        self.logger.metrics_table(rows, columns=["Rule", "Files", "Size"], title=f"Left out of {self.source_dir}")

    def _name_oversized(self) -> None:
        oversized = sorted(((relative, size) for relative, size, rule in self.dropped if rule == self._size_label()), key=lambda item: item[1], reverse=True)
        if not oversized:
            return

        names = [f"{relative} ({size / (1024 * 1024):,.1f} MB)" for relative, size in oversized[:self.NAMED_LIMIT]]
        more  = f" and {len(oversized) - self.NAMED_LIMIT} more" if len(oversized) > self.NAMED_LIMIT else ""

        self.logger.warning(f"{len(oversized)} file(s) of {self.source_dir} left out for being {self._size_label()}, largest first: {', '.join(names)}{more}")

    def _size_label(self) -> str:
        return f"over {self.max_bytes / (1024 * 1024):,.1f} MB"

    def filter(self, paths: List[Path]) -> List[Path]:
        kept = []
        for path in paths:
            relative = path.relative_to(self.source_dir)
            rule     = self._rule(path, relative)

            if rule:
                self.dropped.append((relative, path.stat().st_size, rule))
            else:
                kept.append(path)

        self._report()
        self._name_oversized()
        return kept


class ArtifactBundle:

    def __init__(self, config: ArtifactMailConfig, logger) -> None:
        self.config = config
        self.logger = logger

        self.source_dir = Path(config.source_dir)
        self.excluded   = ExcludedContent(self.source_dir, config.excluded_paths, config.excluded_suffixes, config.max_file_bytes, logger)

    def _required_roots(self) -> List[Path]:
        roots = []
        for relative in self.config.required_paths:
            path = self.source_dir / relative

            roots.append(path)

        return roots

    def _optional_roots(self) -> List[Path]:
        roots = []
        for relative in self.config.optional_paths:
            path = self.source_dir / relative

            if path.exists():
                roots.append(path)
            else:
                self.logger.warning(f"No '{relative}' under {self.source_dir}; it is not attached")

        return roots

    def _entries(self, roots: List[Path]) -> List[Tuple[Path, str]]:
        files = []
        for root in roots:
            files += [root] if root.is_file() else [path for path in sorted(root.rglob("*")) if path.is_file()]

        ordered = self.excluded.filter(list(dict.fromkeys(files)))
        return [(path, f"{self.config.label}/{path.relative_to(self.source_dir).as_posix()}") for path in ordered]

    def _report(self, entries: List[Tuple[Path, str]]) -> None:
        groups = {}
        for path, _arcname in entries:
            key          = path.relative_to(self.source_dir).parts[0]
            count, total = groups.get(key, (0, 0))
            groups[key]  = (count + 1, total + path.stat().st_size)

        rows = [{"Path": key, "Files": str(count), "Size": f"{total / (1024 * 1024):,.1f} MB"} for key, (count, total) in sorted(groups.items())]
        self.logger.metrics_table(rows, columns=["Path", "Files", "Size"], title=f"Gathered from {self.source_dir}")

    def stage(self) -> List[Path]:
        self.logger.subsection(f"Gathering artifacts: {self.config.label}")

        roots   = self._required_roots() + self._optional_roots()
        entries = self._entries(roots)

        self._report(entries)

        return ArchivePacker(entries, self.config.staging_dir, self.config.archive_stem, self.config.max_attachment_bytes, self.logger).pack()


class RunContentBundle(ArtifactBundle):

    def _required_roots(self) -> List[Path]:
        return [self.source_dir]
