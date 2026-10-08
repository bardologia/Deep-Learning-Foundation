"""Discovery and archiving of the artifacts a dataset, parameter or training run holds.

Three selectors find the sources: generated dataset folders under a dataset
root, parameter-extraction runs anywhere beneath it, and the training runs at
or below a runs path. ArtifactBundle turns one such source into the zip
volumes that carry its artifacts, keeping the directory tree of the source so
unpacking every volume into one folder reconstructs it, and ExcludedContent
drops from them whatever is too heavy to travel by mail. RunContentBundle
gathers a training run that way: the whole tree, minus what those rules leave
out.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing  import List, Tuple

from configuration.diagnostics                 import ArtifactMailConfig
from pipelines.shared.inference.run_classifier import RunDirectoryWalk
from tools.data.archives                       import ArchivePacker
from tools.runtime.run_selector                import RunSelector


class DatasetSelector(RunSelector):
    """Selects generated dataset directories by the dataset layout they carry.

    Attributes:
        LAYOUT_RELPATH : Layout file the dataset generation writes into every dataset.
        PARAMS_DIRNAME : Directory holding the dataset's parameter-extraction runs.
    """

    LAYOUT_RELPATH = "data/dataset.json"
    PARAMS_DIRNAME = "params"

    def __init__(self, datasets_dir: Path, logger) -> None:
        """Initializes the selector for dataset mailing.

        Args:
            datasets_dir : Root directory holding one folder per generated dataset.
            logger       : Logger for the dataset table and selection messages.
        """
        super().__init__(datasets_dir, self.LAYOUT_RELPATH, logger, action="mail")

    def _discover(self) -> list[Path]:
        """Returns the sorted direct children carrying a dataset layout.

        Raises:
            FileNotFoundError : If the dataset root is missing or no child carries a layout.
        """
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Datasets directory does not exist: {self.runs_dir}")

        datasets = sorted(child for child in self.runs_dir.iterdir() if child.is_dir() and (child / self.marker).is_file())

        if not datasets:
            raise FileNotFoundError(f"No dataset with a '{self.marker}' layout found under {self.runs_dir}; run generate_dataset first")

        return datasets

    def _present(self, run_dirs: list[Path]) -> None:
        """Logs a numbered table of the datasets with their parameter-run counts."""
        rows = []
        for index, dataset_dir in enumerate(run_dirs, start=1):
            params = dataset_dir / self.PARAMS_DIRNAME
            rows.append({
                "#"       : index,
                "Dataset" : dataset_dir.name,
                "Params"  : str(sum(1 for child in params.iterdir() if child.is_dir()) if params.is_dir() else 0),
            })

        self.logger.metrics_table(rows, columns=["#", "Dataset", "Params"], title=f"Datasets under {self.runs_dir}")


class ParamRunSelector(RunSelector):
    """Selects parameter-extraction runs by the extraction metadata they carry.

    Attributes:
        META_FILENAME       : Metadata file the extraction pipeline writes into every run.
        PARAMETERS_FILENAME : Fitted parameter stack a completed run holds.
        SUMMARY_FILENAME    : Metrics summary the parameter analysis writes.
    """

    META_FILENAME       = "param_extraction_meta.json"
    PARAMETERS_FILENAME = "parameters.npy"
    SUMMARY_FILENAME    = "fit_metrics_summary.json"

    def __init__(self, params_dir: Path, logger) -> None:
        """Initializes the selector for parameter-run mailing.

        Args:
            params_dir : Root directory searched for parameter-extraction runs, the dataset root.
            logger     : Logger for the run table and selection messages.
        """
        super().__init__(params_dir, self.META_FILENAME, logger, action="mail")

    def _discover(self) -> list[Path]:
        """Returns the sorted runs carrying both the extraction metadata and the parameter stack.

        Raises:
            FileNotFoundError : If the root is missing or holds no completed parameter run.
        """
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Parameter runs directory does not exist: {self.runs_dir}")

        markers  = self.runs_dir.rglob(self.marker)
        run_dirs = sorted({path.parent for path in markers if path.is_file() and (path.parent / self.PARAMETERS_FILENAME).is_file()})

        if not run_dirs:
            raise FileNotFoundError(f"No parameter run with a '{self.marker}' and a '{self.PARAMETERS_FILENAME}' found under {self.runs_dir}; run extract_params first")

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        """Logs a numbered table of the parameter runs with their dataset and analysis state."""
        rows = []
        for index, run_dir in enumerate(run_dirs, start=1):
            rows.append({
                "#"        : index,
                "Run"      : str(run_dir.relative_to(self.runs_dir)),
                "Dataset"  : run_dir.parent.parent.name,
                "Analysed" : "yes" if (run_dir / self.SUMMARY_FILENAME).is_file() else "no",
            })

        self.logger.metrics_table(rows, columns=["#", "Run", "Dataset", "Analysed"], title=f"Parameter runs under {self.runs_dir}")


class RunContentSelector(RunSelector):
    """Selects training runs by the meta directory every run writes.

    The root may be a whole runs tree, one family directory, a sweep, or a
    single run: a run is recognised by what it holds, never by its name, and a
    run is never descended into, so the seed runs of a trial are listed as the
    separate runs they are.

    Attributes:
        SEARCH_ENTRY : Entry named in the error raised when the root holds no run.
    """

    SEARCH_ENTRY = "train_backbone"

    def __init__(self, runs_dir: Path, logger) -> None:
        """Initializes the selector for run-content mailing.

        Args:
            runs_dir : Path searched for runs at or below it, or one run directory itself.
            logger   : Logger for the run table and selection messages.
        """
        super().__init__(runs_dir, RunDirectoryWalk.MARKER, logger, action="mail")

    def _discover(self) -> list[Path]:
        """Returns the runs at or below the root, sorted, or the root itself when it is a run.

        Raises:
            FileNotFoundError : If the root is missing or holds no run.
        """
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Runs directory does not exist: {self.runs_dir}")

        if (self.runs_dir / self.marker).is_dir():
            return [self.runs_dir]

        run_dirs = sorted(RunDirectoryWalk.walk(self.runs_dir))

        if not run_dirs:
            raise FileNotFoundError(f"No run with a '{self.marker}' directory found at or below {self.runs_dir}; run {self.SEARCH_ENTRY} first")

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        """Logs a numbered table of the runs with the files they hold and their size on disk.

        Every file of every run is listed and stat'ed first, which on network
        storage can take minutes, so the scan is announced and tracked by a
        progress bar advancing once per run.
        """
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
        """Returns the listed name of one run, its path below the root or its own name when the root is the run."""
        return run_dir.name if run_dir == self.runs_dir else str(run_dir.relative_to(self.runs_dir))


class ExcludedContent:
    """Drops the gathered files too heavy to mail and reports what was left out.

    Three rules decide, in this order. A path glob is matched against a file's
    path relative to the source directory, against every one of its parent
    directories and against the bare name of each of them, so ``cubes`` drops a
    directory of that name wherever it sits while ``inference/*/cubes`` drops
    only the ones inside an inference stamp. A suffix names a file extension and
    is matched case-insensitively. The size cap drops a single file above it,
    which is what keeps an unforeseen heavy artifact out without naming it.

    Attributes:
        source_dir  : Directory the gathered files are read relative to.
        patterns    : Path globs whose matches are left out.
        suffixes    : Lowercased file extensions left out.
        max_bytes   : Upper bound of one kept file in bytes; zero turns the size rule off.
        logger      : Logger for the table of what was left out and the files the size cap dropped.
        dropped     : One (relative path, size in bytes, rule) triple per file left out, filled by the filtering.
        NAMED_LIMIT : Number of files the size-cap warning names before it only counts the rest.
    """

    NAMED_LIMIT = 20

    def __init__(self, source_dir: Path, patterns: List[str], suffixes: List[str], max_bytes: int, logger) -> None:
        """Initializes the filter with the directory paths are read against, its three rules and the logger."""
        self.source_dir = Path(source_dir)
        self.patterns   = list(patterns)
        self.suffixes   = [suffix.lower() for suffix in suffixes]
        self.max_bytes  = int(max_bytes)
        self.logger     = logger

        self.dropped = []

    def _path_rule(self, relative: Path) -> str:
        """Returns the path glob matching the file or one of its parents, empty when none does.

        Args:
            relative : Path of the file below the source directory.
        """
        candidates = [relative] + [parent for parent in relative.parents if parent != Path(".")]

        for candidate in candidates:
            for pattern in self.patterns:
                if fnmatch(candidate.as_posix(), pattern) or fnmatch(candidate.name, pattern):
                    return f"path {pattern}"

        return ""

    def _suffix_rule(self, relative: Path) -> str:
        """Returns the excluded extension of the file, empty when its suffix is kept."""
        suffix = relative.suffix.lower()

        return f"suffix {suffix}" if suffix in self.suffixes else ""

    def _size_rule(self, path: Path) -> str:
        """Returns the cap the file exceeds, empty when it fits or the size rule is off."""
        if not self.max_bytes:
            return ""

        return self._size_label() if path.stat().st_size > self.max_bytes else ""

    def _rule(self, path: Path, relative: Path) -> str:
        """Returns the first rule that leaves the file out, empty when all three keep it.

        Args:
            path     : File found under the source directory.
            relative : Path of that file below the source directory.
        """
        pattern = self._path_rule(relative)
        if pattern:
            return pattern

        suffix = self._suffix_rule(relative)
        if suffix:
            return suffix

        return self._size_rule(path)

    def _report(self) -> None:
        """Logs one row per rule that left something out, with the file count and the size saved."""
        if not self.dropped:
            return

        groups = {}
        for _relative, size, rule in self.dropped:
            count, total = groups.get(rule, (0, 0))
            groups[rule] = (count + 1, total + size)

        rows = [{"Rule": rule, "Files": str(count), "Size": f"{total / (1024 * 1024):,.1f} MB"} for rule, (count, total) in sorted(groups.items())]
        self.logger.metrics_table(rows, columns=["Rule", "Files", "Size"], title=f"Left out of {self.source_dir}")

    def _name_oversized(self) -> None:
        """Warns with the files the size cap left out, largest first, since no path or suffix rule named them."""
        oversized = sorted(((relative, size) for relative, size, rule in self.dropped if rule == self._size_label()), key=lambda item: item[1], reverse=True)
        if not oversized:
            return

        names = [f"{relative} ({size / (1024 * 1024):,.1f} MB)" for relative, size in oversized[:self.NAMED_LIMIT]]
        more  = f" and {len(oversized) - self.NAMED_LIMIT} more" if len(oversized) > self.NAMED_LIMIT else ""

        self.logger.warning(f"{len(oversized)} file(s) of {self.source_dir} left out for being {self._size_label()}, largest first: {', '.join(names)}{more}")

    def _size_label(self) -> str:
        """Returns the name of the size rule, the cap it enforces in megabytes."""
        return f"over {self.max_bytes / (1024 * 1024):,.1f} MB"

    def filter(self, paths: List[Path]) -> List[Path]:
        """Returns the files every rule keeps, recording and reporting the ones left out.

        Args:
            paths : Files found under the source directory, in gathering order.

        Returns:
            The kept files, in the order they were given.
        """
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
    """Gathers one source directory's artifacts and packs them into zip volumes.

    Attributes:
        config     : Gathering config naming the source, the staging directory, the gathered paths and the exclusion rules.
        logger     : Logger for the gathered-content table and the per-volume progress.
        source_dir : Source directory whose artifacts are packed.
        excluded   : Filter dropping the gathered files the exclusion rules leave out.
    """

    def __init__(self, config: ArtifactMailConfig, logger) -> None:
        """Initializes the bundle with its gathering config and logger."""
        self.config = config
        self.logger = logger

        self.source_dir = Path(config.source_dir)
        self.excluded   = ExcludedContent(self.source_dir, config.excluded_paths, config.excluded_suffixes, config.max_file_bytes, logger)

    def _required_roots(self) -> List[Path]:
        """Returns the required paths, every one of which must exist.

        Raises:
            FileNotFoundError : If a required path is missing from the source directory.
        """
        roots = []
        for relative in self.config.required_paths:
            path = self.source_dir / relative

            if not path.exists():
                raise FileNotFoundError(f"Required artifact '{relative}' is missing from {self.source_dir}")

            roots.append(path)

        return roots

    def _optional_roots(self) -> List[Path]:
        """Returns the optional paths that exist, warning once for each that does not."""
        roots = []
        for relative in self.config.optional_paths:
            path = self.source_dir / relative

            if path.exists():
                roots.append(path)
            else:
                self.logger.warning(f"No '{relative}' under {self.source_dir}; it is not attached")

        return roots

    def _analysis_roots(self) -> List[Path]:
        """Returns the analysis paths that exist, naming the analysis entry when none do."""
        roots = [self.source_dir / relative for relative in self.config.analysis_paths]
        found = [path for path in roots if path.exists()]

        if not found and roots:
            self.logger.warning(f"No analysis output under {self.source_dir}; run {self.config.analysis_entry} on it to attach the analysis artifacts")
            return []

        for path in roots:
            if not path.exists():
                self.logger.warning(f"No '{path.relative_to(self.source_dir).as_posix()}' under {self.source_dir}; it is not attached")

        return found

    def _entries(self, roots: List[Path]) -> List[Tuple[Path, str]]:
        """Expands the gathered roots into the files they hold and their archive paths.

        Args:
            roots : Existing files and directories under the source directory.

        Returns:
            One (source path, path inside the archive) pair per file the exclusion
            rules keep, deduplicated and sorted, every archive path nested under the
            bundle label so unpacking several volumes into one folder rebuilds the
            source tree.
        """
        files = []
        for root in roots:
            files += [root] if root.is_file() else [path for path in sorted(root.rglob("*")) if path.is_file()]

        ordered = self.excluded.filter(list(dict.fromkeys(files)))
        return [(path, f"{self.config.label}/{path.relative_to(self.source_dir).as_posix()}") for path in ordered]

    def _report(self, entries: List[Tuple[Path, str]]) -> None:
        """Logs the gathered content grouped by the top-level directory it came from.

        Args:
            entries : Files to pack, as (source path, archive path) pairs.
        """
        groups = {}
        for path, _arcname in entries:
            key          = path.relative_to(self.source_dir).parts[0]
            count, total = groups.get(key, (0, 0))
            groups[key]  = (count + 1, total + path.stat().st_size)

        rows = [{"Path": key, "Files": str(count), "Size": f"{total / (1024 * 1024):,.1f} MB"} for key, (count, total) in sorted(groups.items())]
        self.logger.metrics_table(rows, columns=["Path", "Files", "Size"], title=f"Gathered from {self.source_dir}")

    def stage(self) -> List[Path]:
        """Gathers the source's artifacts and packs them into the staged archives.

        Returns:
            The written archive paths, in volume order.
        """
        self.logger.subsection(f"Gathering artifacts: {self.config.label}")

        roots   = self._required_roots() + self._optional_roots() + self._analysis_roots()
        entries = self._entries(roots)

        self._report(entries)

        return ArchivePacker(entries, self.config.staging_dir, self.config.archive_stem, self.config.max_attachment_bytes, self.logger).pack()


class RunContentBundle(ArtifactBundle):
    """Gathers everything one training run holds, minus what the exclusion rules drop.

    A run is mailed for what it recorded rather than for a known set of files,
    so its whole tree is the single gathered root and the exclusion rules of the
    config are what keep the checkpoints, the reconstructed cubes and any other
    artifact too heavy to mail out of the archives.
    """

    def _required_roots(self) -> List[Path]:
        """Returns the run directory itself, the one root the whole tree is gathered from.

        Raises:
            FileNotFoundError : If the run directory does not exist.
        """
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {self.source_dir}")

        return [self.source_dir]
