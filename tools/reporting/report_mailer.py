"""Mailing of the result bundles of experiments, training runs, datasets and parameter runs.

Four targets share one mailer. The experiment target takes any path, finds the
experiments at or below it by the trial runs they hold, whatever they are
called: a trial sweep directory, a cross-validation run, a benchmark run, a
sequence workspace holding several of them, or a whole runs tree. For each
experiment it writes the seed-aggregate report of every seeded trial and the
trial comparison when they are missing, the cross-fold report of a
cross-validation run as well, and gathers the seed-aggregate reports, the
comparison reports and the configuration the experiment ran with into one
message. The run target mails one training run as it stands, everything under
it but the checkpoints, the reconstructed cubes and whatever else the exclusion
rules leave out. The dataset and parameter-run targets gather everything a
generated dataset or a fitted parameter run holds, plus the artifacts the
parameter analysis entry wrote. The last three targets pack into size-capped
zip volumes, and each source becomes one message, or one message per volume
when its artifacts do not fit into a single attachment.
"""

from __future__ import annotations

import os
import smtplib
import time

from email.message import EmailMessage
from pathlib       import Path
from typing        import List

from configuration.comparison                  import TrialComparisonConfig
from configuration.diagnostics                 import ExperimentReportMailConfig, ReportMailEntryConfig
from pipelines.shared.inference.metadata       import CompletedInference
from pipelines.shared.inference.run_classifier import InferenceLayout, RunDirectoryWalk
from pipelines.shared.training.seed_sweep      import SeedSet
from tools.data.io                             import FileIO
from tools.reporting.artifact_bundles          import ArtifactBundle, DatasetSelector, ParamRunSelector, RunContentBundle, RunContentSelector
from tools.reporting.report_collector          import ReportImageRewriter
from tools.runtime.completion                  import CompletionMarker
from tools.runtime.run_selector                import RunSelector
from tools.runtime.run_tag                     import RunTag


class TrialTree:
    """Recognises trial runs and the directories holding them by the run metadata they carry.

    A run is a directory holding a meta directory; a trial is a run or a
    directory whose children are ``seed<N>`` runs; an experiment directory is
    one whose children include trials.

    Attributes:
        SKIP : Directory names that hold a run's or an experiment's own artifacts and never a trial.
    """

    SKIP = ("logs", "batch_train_logs", "pipeline", "reports", "comparison", "inference", "meta", "docs", "tensorboard", "cubes", "figures", "animations", "data", "max_batch", "pretrain", "ddp")

    @staticmethod
    def is_run(directory: Path) -> bool:
        """Returns whether a directory is a training run."""
        return (Path(directory) / RunDirectoryWalk.MARKER).is_dir()

    @classmethod
    def seed_runs(cls, directory: Path) -> list[Path]:
        """Returns the ``seed<N>`` runs nested inside a directory, sorted by name."""
        directory = Path(directory)
        if not directory.is_dir():
            return []

        return sorted(child for child in directory.iterdir() if child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) and cls.is_run(child))

    @classmethod
    def is_trial(cls, directory: Path) -> bool:
        """Returns whether a directory is a trial: a run, or a sweep of seed runs."""
        return cls.is_run(directory) or bool(cls.seed_runs(directory))

    @classmethod
    def candidates(cls, directory: Path) -> list[Path]:
        """Returns the sub-directories of a directory that may hold trials or experiments, sorted by name."""
        directory = Path(directory)
        if not directory.is_dir():
            return []

        return sorted(child for child in directory.iterdir() if child.is_dir() and not child.name.startswith((".", "_")) and child.name not in cls.SKIP)

    @classmethod
    def trials(cls, directory: Path) -> list[Path]:
        """Returns the trials directly under a directory, sorted by name."""
        return [child for child in cls.candidates(directory) if cls.is_trial(child)]

    @classmethod
    def runs(cls, trial: Path) -> list[Path]:
        """Returns the runs of a trial: its seed runs, or the trial itself when it is a single run."""
        seeds = cls.seed_runs(trial)
        return seeds if seeds else [Path(trial)]

    @staticmethod
    def inference_stamps(inference_dir: Path, split: str | None) -> list[Path]:
        """Returns the completed inference stamps of a run, ordered by stamp name.

        Spatial families stamp directly under the inference directory and the
        profile and image autoencoder families under their family
        sub-directory, so both layouts are listed.

        Args:
            inference_dir : Inference directory of the run.
            split         : Split the stamps must have scored, or None for every split.
        """
        bases = [Path(inference_dir)] + [Path(inference_dir) / family for family in InferenceLayout.SUBDIRS.values()]

        if split is None:
            stamps = [stamp for base in bases if base.is_dir() for stamp in base.iterdir() if stamp.is_dir() and CompletionMarker.is_complete(stamp)]
        else:
            stamps = [stamp for base in bases for stamp in CompletedInference.stamps(base, split)]

        return sorted(stamps, key=lambda stamp: stamp.name)


class ReportFreshness:
    """Names every reused report and warns when the runs it summarises have moved on since it was written.

    A reused report is stale when one of its runs completed an inference after
    the report was written, or when the experiment now holds trials or seeds
    the report does not cover. It is still mailed; each warning names what is
    newer or missing.

    Attributes:
        inference_dirname : Directory of a run holding its inference stamps.
        logger            : Logger receiving the reuse lines and the staleness warnings.
        HINT              : Closing advice of every staleness warning.
    """

    HINT = "set regenerate_reports=True to rewrite it"

    def __init__(self, inference_dirname: str, logger) -> None:
        """Binds the check to the runs' inference directory name and the logger."""
        self.inference_dirname = inference_dirname
        self.logger            = logger

    @staticmethod
    def _written_at(stamp: Path) -> float:
        """Returns when a report stamp was written: the modification time of the newest file directly inside it, in seconds since the epoch.

        Raises:
            FileNotFoundError : If the stamp holds no file.
        """
        times = [path.stat().st_mtime for path in stamp.iterdir() if path.is_file()]
        if not times:
            raise FileNotFoundError(f"Report stamp {stamp} holds no file")

        return max(times)

    def _newer_inference(self, runs: list[Path], split: str | None, since: float) -> list[Path]:
        """Returns the completed inference stamps of the runs whose completion marker was written after a time, oldest first.

        Args:
            runs  : Runs whose inference stamps are listed.
            split : Split the stamps must have scored, or None for every split.
            since : Time in seconds since the epoch the markers are compared with.
        """
        stamps = [stamp for run in runs for stamp in TrialTree.inference_stamps(run / self.inference_dirname, split)]
        newer  = [stamp for stamp in stamps if CompletionMarker.path(stamp).stat().st_mtime > since]

        return sorted(newer, key=lambda stamp: CompletionMarker.path(stamp).stat().st_mtime)

    def check(self, kind: str, stamp: Path, runs: list[Path], split: str | None, uncovered: list[str]) -> None:
        """Logs the reuse of a report stamp and warns about the newer inference of its runs and the trials or seeds it does not cover.

        Args:
            kind      : Name of the report in the log lines, such as 'trial comparison'.
            stamp     : Reused report stamp directory.
            runs      : Runs whose inference the report summarises.
            split     : Split whose inference counts as newer, or None for every split.
            uncovered : Trials or seeds the experiment holds that the report does not cover.
        """
        self.logger.info(f"Reusing {kind} {stamp}")

        newer = self._newer_inference(runs, split, self._written_at(stamp))
        if newer:
            self.logger.warning(f"Reused {kind} {stamp.name} predates {len(newer)} completed inference(s) of its runs, the newest {newer[-1]}; {self.HINT}")

        if uncovered:
            self.logger.warning(f"Reused {kind} {stamp.name} does not cover {', '.join(uncovered)}; {self.HINT}")


class Experiment:
    """One mailable experiment: a directory of trial runs, their comparison and the configuration they ran with.

    Attributes:
        KIND             : Name of the experiment kind in tables and messages.
        SEARCH_BELOW     : Whether the scan goes on below the experiment: a sweep directory may hold further experiments beside its trials, a run's tree holds none.
        SUMMARY_FILENAME : JSON file of a comparison stamp recording the split and the trials it compared.
        root             : Directory identifying the experiment.
        trials_dir       : Directory holding the trial runs.
        selected         : Trials the experiment is restricted to, or None for every trial under the trials directory.
    """

    KIND             = "sweep"
    SEARCH_BELOW     = True
    SUMMARY_FILENAME = "comparison_summary.json"

    def __init__(self, root: Path, trials_dir: Path, selected: list[Path] | None = None) -> None:
        """Binds the experiment to its root and the directory holding its trials."""
        self.root       = Path(root)
        self.trials_dir = Path(trials_dir)
        self.selected   = list(selected) if selected is not None else None

    @property
    def label(self) -> str:
        """Returns the mailed identity of the experiment, its own name prefixed by its parent directory."""
        return f"{self.root.parent.name}/{self.root.name}"

    def trials(self) -> list[Path]:
        """Returns the trials of the experiment, sorted by name.

        Raises:
            FileNotFoundError : If the trials directory holds no trial.
        """
        trials = self.selected if self.selected is not None else TrialTree.trials(self.trials_dir)
        if not trials:
            raise FileNotFoundError(f"No trial run under {self.trials_dir}: expected directories holding a meta directory, or seed<N> runs")

        return trials

    def inference_split(self) -> str:
        """Returns the split the trial comparison scores and every mailed trial report is picked from, the trial-comparison default."""
        return TrialComparisonConfig.inference_split

    def comparison_root(self, comparison_dirname: str) -> Path:
        """Returns the directory holding the timestamped comparison stamps."""
        return self.trials_dir / comparison_dirname

    def comparison_report_names(self, configured: list[str]) -> list[str]:
        """Returns the report files gathered from a comparison stamp."""
        return list(configured)

    def stamp_coverage(self, stamp: Path) -> tuple[str, list[str]] | None:
        """Returns the split and the trial names a comparison stamp recorded, or None for a stamp written before comparisons carried a summary.

        Args:
            stamp : Trial-comparison stamp directory.
        """
        path = stamp / self.SUMMARY_FILENAME
        if not path.is_file():
            return None

        payload = FileIO.load_json(path)

        return payload["inference_split"], [trial["name"] for trial in payload["trials"]]

    def write_comparison(self, comparison_dirname: str, logger) -> Path:
        """Compares the trials as the trial-comparison entry does and returns the written stamp.

        Args:
            comparison_dirname : Directory under the trials directory receiving the stamp.
            logger             : Logger for the collection and the written files.
        """
        from pipelines.comparison.comparison_report import TrialComparisonReport
        from pipelines.comparison.trial_collector   import TrialCollector

        config  = TrialComparisonConfig(runs_dir=self.trials_dir, run_tags=[trial.name for trial in self.trials()], inference_split=self.inference_split())
        out_dir = self.comparison_root(comparison_dirname) / RunTag.now()

        collector = TrialCollector(runs_dir=self.trials_dir, run_tags=config.run_tags, logger=logger, inference_split=config.inference_split)
        records   = collector.collect()

        TrialComparisonReport(records=records, out_dir=out_dir, compare_images=config.compare_images, compare_gifs=config.compare_gifs, embed_images=config.embed_images, inference_split=config.inference_split, logger=logger, seed_dispersion=collector.seed_dispersion).write_all()

        return out_dir

    def extra_reports(self, regenerate: bool, logger) -> list[tuple[Path, str]]:
        """Returns further (path, staged name) report pairs of the experiment; none for a plain sweep."""
        return []

    def config_files(self, logger) -> list[Path]:
        """Returns the files describing what the experiment ran, warning for each expected one that is absent.

        A sweep records its configuration and its scheduler results under
        ``batch_train_logs``, and every run its resolved entry configuration
        under ``docs``; the first run stands for the sweep.

        Args:
            logger : Logger the warnings are written to.
        """
        from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
        from pipelines.shared.training.resume_source       import SweepRecording

        first = TrialTree.runs(self.trials()[0])[0]

        candidates = [
            SweepRecording.path(self.trials_dir),
            self.trials_dir / SweepRecording.DIRNAME / "train_scheduler_results.json",
            ResolvedEntryConfigWriter.path_for(first),
        ]

        return self._existing(candidates, logger)

    def _existing(self, candidates: list[Path], logger) -> list[Path]:
        """Returns the candidate files that exist, warning once for each that does not."""
        found = []
        for path in candidates:
            if path.is_file():
                found.append(path)
            else:
                logger.warning(f"No '{path.relative_to(self.root) if path.is_relative_to(self.root) else path}' under {self.root}; it is not attached")

        return found


class CrossValidationExperiment(Experiment):
    """A cross-validation run: its folds are the trials and its cross-fold report travels with them.

    Attributes:
        FOLDS_DIRNAME     : Directory of the run holding the fold runs.
        REPORTS_DIRNAME   : Directory of the run holding one timestamped cross-fold report per stamp.
        AGGREGATE_REPORTS : Files gathered from the newest cross-fold report stamp itself.
        SPLIT_REPORTS     : Files gathered from every split directory of that stamp.
        PIPELINE_FILES    : Paths inside the run describing what it ran with.
    """

    KIND         = "cross-validation"
    SEARCH_BELOW = False

    FOLDS_DIRNAME     = "folds"
    REPORTS_DIRNAME   = "reports"
    AGGREGATE_REPORTS = ("cv_aggregate_report.md", "cv_summary.json")
    SPLIT_REPORTS     = ("benchmark_overview.md", "metrics_comparison.md")
    PIPELINE_FILES    = ("pipeline/resolved_config.json", "pipeline/state.json")

    def __init__(self, root: Path) -> None:
        """Binds the experiment to a cross-validation run directory."""
        super().__init__(root, Path(root) / self.FOLDS_DIRNAME)

    @classmethod
    def is_run(cls, directory: Path) -> bool:
        """Returns whether a directory is a cross-validation run: it holds a folds directory with trials inside."""
        return bool(TrialTree.trials(Path(directory) / cls.FOLDS_DIRNAME))

    def _stamps(self) -> list[Path]:
        """Returns the sorted cross-fold report stamps of the run, empty without a reports directory."""
        reports = self.root / self.REPORTS_DIRNAME
        if not reports.is_dir():
            return []

        return sorted(child for child in reports.iterdir() if child.is_dir())

    def _report_stamp(self, regenerate: bool, logger) -> Path:
        """Returns the cross-fold report stamp to gather, writing one from the fold tree when missing or asked for.

        A reused stamp is named in the log and checked against the folds'
        completed inference of every split, since the report covers them all.
        """
        from pipelines.cross_validation.stages import CrossValidationReportRebuilder

        stamps = self._stamps()
        if stamps and not regenerate:
            runs = [run for trial in self.trials() for run in TrialTree.runs(trial)]
            ReportFreshness(InferenceLayout.DIRNAME, logger).check("cross-fold report", stamps[-1], runs, None, [])
            return stamps[-1]

        logger.info(f"{'Regenerating' if stamps else 'Writing'} the cross-fold report of {self.root} from its fold tree")

        return CrossValidationReportRebuilder(self.root, logger).rebuild()

    def extra_reports(self, regenerate: bool, logger) -> list[tuple[Path, str]]:
        """Returns the cross-fold aggregate files and every split's comparison pages of the newest report stamp.

        Raises:
            FileNotFoundError : If a required file is missing from the stamp.
        """
        stamp = self._report_stamp(regenerate, logger)
        pairs = []

        for name in self.AGGREGATE_REPORTS:
            path = stamp / name
            if not path.is_file():
                raise FileNotFoundError(f"Aggregate report '{name}' is missing from {stamp}")
            pairs.append((path, name))

        split_dirs = sorted(child for child in stamp.iterdir() if child.is_dir())
        if not split_dirs:
            logger.warning(f"No split directory under {stamp}; the run compared its folds on training checkpoints alone, so no per-split comparison is attached")

        for split_dir in split_dirs:
            for name in self.SPLIT_REPORTS:
                path = split_dir / name
                if not path.is_file():
                    raise FileNotFoundError(f"Split report '{name}' is missing from {split_dir}")
                pairs.append((path, f"{split_dir.name}_{name}"))

        return pairs

    def config_files(self, logger) -> list[Path]:
        """Returns the resolved configuration and stage state the run wrote, warning for each that is absent."""
        return self._existing([self.root / relative for relative in self.PIPELINE_FILES], logger)


class BenchmarkExperiment(Experiment):
    """A benchmark run: its training units are the trials and its own comparison stage writes the comparison.

    Attributes:
        TRAINING_DIRNAME   : Directory of the run holding the unit runs.
        COMPARISON_DIRNAME : Directory of the run holding one timestamped comparison per stamp.
        COMPARISON_REPORTS : Files gathered from the newest comparison stamp.
        PIPELINE_FILES     : Paths inside the run describing what it ran with.
    """

    KIND         = "benchmark"
    SEARCH_BELOW = False

    TRAINING_DIRNAME   = "training"
    COMPARISON_DIRNAME = "comparison"
    COMPARISON_REPORTS = ("benchmark_overview.md", "metrics_comparison.md")
    PIPELINE_FILES     = ("pipeline/resolved_config.json", "pipeline/state.json")

    def __init__(self, root: Path) -> None:
        """Binds the experiment to a benchmark run directory."""
        super().__init__(root, Path(root) / self.TRAINING_DIRNAME)

    @classmethod
    def is_run(cls, directory: Path) -> bool:
        """Returns whether a directory is a benchmark run: it holds a pipeline directory and a training directory with trials inside."""
        directory = Path(directory)
        return (directory / "pipeline").is_dir() and bool(TrialTree.trials(directory / cls.TRAINING_DIRNAME))

    def _recorded_config(self):
        """Returns the benchmark configuration the run recorded in its pipeline directory."""
        from configuration.benchmark                    import BenchmarkConfig
        from pipelines.shared.orchestration.resumed_run import StagedRunLocator
        from tools.runtime.config_cli                   import ConfigCli

        return ConfigCli.load_resolved(BenchmarkConfig(), StagedRunLocator.config_path(self.root))

    def inference_split(self) -> str:
        """Returns the split the run's inference stage scored, as its recorded configuration names it."""
        return self._recorded_config().inference.split

    def comparison_root(self, comparison_dirname: str) -> Path:
        """Returns the run's own comparison directory, whatever the sweep comparison directory is called."""
        return self.root / self.COMPARISON_DIRNAME

    def comparison_report_names(self, configured: list[str]) -> list[str]:
        """Returns the benchmark comparison's own report files."""
        return list(self.COMPARISON_REPORTS)

    def stamp_coverage(self, stamp: Path) -> tuple[str, list[str]] | None:
        """Returns the recorded split and the unit names a benchmark comparison stamp compared, or None for a stamp without its summary.

        Args:
            stamp : Benchmark comparison stamp directory, whose summary lists one entry per unit and was written on the recorded split.
        """
        path = stamp / self.SUMMARY_FILENAME
        if not path.is_file():
            return None

        return self.inference_split(), [entry["name"] for entry in FileIO.load_json(path)]

    def write_comparison(self, comparison_dirname: str, logger) -> Path:
        """Runs the benchmark comparison stage over the run with the configuration and split it recorded and returns the stamp.

        Args:
            comparison_dirname : Ignored; the benchmark writes under its own comparison directory.
            logger             : Logger for the stage.
        """
        from pipelines.benchmark.stages import ComparisonStage

        config = self._recorded_config()
        stage  = ComparisonStage(config=config, run_tag=self.root.name, logger=logger, reference_model=config.size_match.reference_model, embed_images=config.comparison.embed_images, inference_split=config.inference.split, run_dir=self.root)

        return stage.run()

    def config_files(self, logger) -> list[Path]:
        """Returns the resolved configuration and stage state the run wrote, warning for each that is absent."""
        return self._existing([self.root / relative for relative in self.PIPELINE_FILES], logger)


class ExperimentScanner:
    """Finds the experiments at or below a root by the trial runs they hold.

    The walk classifies every directory it reaches: a cross-validation run by
    its folds, a benchmark run by its pipeline and training directories, and
    any other directory with trial children as a sweep. It stops at each
    trial and at each run-shaped experiment, so run trees are never entered,
    goes on below a sweep, whose standalone runs may sit beside further
    sweeps and runs, and gives up below ``MAX_DEPTH``. The root may also be a
    trial itself, which is then mailed as an experiment of one, or the folds
    or training directory of a run, which resolves to that run.

    Attributes:
        root      : Directory the scan starts at, itself a candidate.
        MAX_DEPTH : Deepest level below the root an experiment is looked for at.
    """

    MAX_DEPTH = 4

    def __init__(self, root: Path) -> None:
        """Binds the scanner to its root."""
        self.root = Path(root)

    @staticmethod
    def classify(directory: Path) -> Experiment | None:
        """Returns the experiment a directory is, or None when it is none.

        Args:
            directory : Directory to classify.
        """
        directory = Path(directory)

        if CrossValidationExperiment.is_run(directory):
            return CrossValidationExperiment(directory)
        if BenchmarkExperiment.is_run(directory):
            return BenchmarkExperiment(directory)
        if directory.name == CrossValidationExperiment.FOLDS_DIRNAME and CrossValidationExperiment.is_run(directory.parent):
            return CrossValidationExperiment(directory.parent)
        if directory.name == BenchmarkExperiment.TRAINING_DIRNAME and BenchmarkExperiment.is_run(directory.parent):
            return BenchmarkExperiment(directory.parent)
        if not TrialTree.is_trial(directory) and TrialTree.trials(directory):
            return Experiment(directory, directory)

        return None

    def name(self, experiment: Experiment) -> str:
        """Returns the experiment's path relative to the root, or its own name when it is the root or above it."""
        root = experiment.root
        if root == self.root or not root.is_relative_to(self.root):
            return root.name

        return root.relative_to(self.root).as_posix()

    def _walk(self, directory: Path, depth: int):
        """Yields the experiments under a directory, stopping at each trial, at each run-shaped experiment and at the depth cap."""
        if TrialTree.is_trial(directory):
            return

        experiment = self.classify(directory)
        if experiment is not None:
            yield experiment

        if (experiment is not None and not experiment.SEARCH_BELOW) or depth >= self.MAX_DEPTH:
            return

        for child in TrialTree.candidates(directory):
            yield from self._walk(child, depth + 1)

    def scan(self) -> list[Experiment]:
        """Returns every experiment found at or below the root, sorted by path.

        Raises:
            NotADirectoryError : If the root is not a directory.
        """
        if not self.root.is_dir():
            raise NotADirectoryError(f"Experiments directory does not exist: {self.root}")

        if self.classify(self.root) is None and TrialTree.is_trial(self.root):
            return [Experiment(self.root, self.root.parent, selected=[self.root])]

        return sorted(self._walk(self.root, 0), key=lambda experiment: experiment.root)


class ExperimentSelector(RunSelector):
    """Selects experiments at or below a directory by the trial runs they hold.

    Attributes:
        scanner     : Scanner finding the experiments.
        experiments : Experiment per root directory, filled by discovery.
    """

    def __init__(self, experiments_dir: Path, comparison_dirname: str, logger) -> None:
        """Initializes the selector for report mailing.

        Args:
            experiments_dir    : Directory searched for experiments: a sweep directory, a cross-validation or benchmark run, a sequence workspace or a whole runs tree.
            comparison_dirname : Directory inside a sweep holding its trial-comparison stamps, counted in the table.
            logger             : Logger for the experiment table and selection messages.
        """
        super().__init__(experiments_dir, comparison_dirname, logger, action="mail")
        self.scanner     = ExperimentScanner(self.runs_dir)
        self.experiments = {}

    def _discover(self) -> list[Path]:
        """Returns the root of every experiment found, sorted.

        Raises:
            FileNotFoundError : If no experiment is found at or below the directory.
        """
        found = self.scanner.scan()
        if not found:
            raise FileNotFoundError(f"No experiment found at or below {self.runs_dir}: expected a directory of trial runs, a cross-validation run, a benchmark run, or a workspace holding them")

        self.experiments = {experiment.root: experiment for experiment in found}

        return list(self.experiments)

    def _present(self, run_dirs: list[Path]) -> None:
        """Logs a numbered table of the experiments with their kind, trial and comparison counts."""
        rows = []
        for index, root in enumerate(run_dirs, start=1):
            experiment = self.experiments[root]
            stamps     = experiment.comparison_root(self.marker)
            rows.append({
                "#"           : index,
                "Experiment"  : self.scanner.name(experiment),
                "Kind"        : experiment.KIND,
                "Trials"      : str(len(experiment.trials())),
                "Comparisons" : str(sum(1 for child in stamps.iterdir() if child.is_dir()) if stamps.is_dir() else 0),
            })

        self.logger.metrics_table(rows, columns=["#", "Experiment", "Kind", "Trials", "Comparisons"], title=f"Experiments at or below {self.runs_dir}")

    def select_experiments(self, experiment_filter: list[str] | None) -> list[Experiment]:
        """Returns the experiments the filter names, or the prompt or all of them as the selector resolves."""
        return [self.experiments[root] for root in self.resolve(experiment_filter)]


class ReportBundle:
    """Staging shared by the report bundles: rewritten reports and verbatim copies.

    Attributes:
        staging_dir  : Directory receiving the staged files.
        embed_images : Whether rewritten reports embed their figures instead of linking them by absolute path.
        logger       : Logger for per-file progress.
    """

    def __init__(self, staging_dir: Path, embed_images: bool, logger) -> None:
        """Initializes the staging directory, the image mode and the logger."""
        self.staging_dir  = Path(staging_dir)
        self.embed_images = embed_images
        self.logger       = logger

    def _stage_report(self, report_path: Path, output_name: str) -> Path:
        """Rewrites one report's image links and writes it into the staging directory.

        Args:
            report_path : Source report inside the run's outputs.
            output_name : Filename of the staged copy.

        Returns:
            Path of the staged copy.
        """
        text        = report_path.read_text(encoding="utf-8")
        rewritten   = ReportImageRewriter(report_path, self.embed_images).rewrite(text)
        destination = self.staging_dir / output_name

        destination.write_text(rewritten, encoding="utf-8")
        self.logger.ok(f"Staged {output_name} from {report_path}")
        return destination

    def _stage_copy(self, path: Path, output_name: str | None = None) -> Path:
        """Copies one file verbatim into the staging directory and returns the copy.

        Args:
            path        : Source file inside the run's outputs.
            output_name : Filename of the staged copy, the source name when omitted.

        Returns:
            Path of the staged copy.
        """
        destination = self.staging_dir / (output_name or path.name)

        destination.write_bytes(path.read_bytes())
        self.logger.ok(f"Staged {destination.name} from {path}")
        return destination


class ExperimentReportBundle(ReportBundle):
    """Gathers one experiment's reports and configuration into a staging directory, writing what is missing first.

    Attributes:
        config     : Gathering config naming the staging directory, the comparison layout and the regeneration switch.
        experiment : Experiment being gathered.
        freshness  : Check naming every reused seed aggregate and comparison and warning when it is stale.
    """

    def __init__(self, experiment: Experiment, config: ExperimentReportMailConfig, logger) -> None:
        """Initializes the bundle with the experiment, its gathering config and logger."""
        super().__init__(config.staging_dir, config.embed_images, logger)
        self.config     = config
        self.experiment = experiment
        self.freshness  = ReportFreshness(config.inference_dirname, logger)

    def _run_report(self, run_dir: Path, split: str) -> Path:
        """Returns the report of a single run's newest completed inference on the split.

        Args:
            run_dir : Run directory whose inference stamps are searched, directly and under the family sub-directories.
            split   : Split the inference must have scored, the one the trial comparison reads.

        Raises:
            FileNotFoundError : If no completed inference on the split holds a report, naming the splits whose stamps do.
        """
        inference = run_dir / self.config.inference_dirname
        reports   = [stamp / self.config.report_filename for stamp in TrialTree.inference_stamps(inference, split)]
        reports   = [path for path in reports if path.is_file()]

        if reports:
            return reports[-1]

        others = sorted({CompletionMarker.payload(stamp)["split"] for stamp in TrialTree.inference_stamps(inference, None) if (stamp / self.config.report_filename).is_file()})
        found  = f"; reports exist for split(s) {', '.join(others)}" if others else ""

        raise FileNotFoundError(f"No '{self.config.report_filename}' of a completed '{split}' inference under {inference}{found}; run inference on the '{split}' split of this trial before mailing it")

    def _seed_aggregates(self, trial_dir: Path) -> list[Path]:
        """Returns the seed-aggregate reports of a seeded trial, oldest stamp first.

        Args:
            trial_dir : Trial directory whose inference directory holds one stamp per seed comparison.
        """
        inference = trial_dir / self.config.inference_dirname
        if not inference.is_dir():
            return []

        return sorted(path for path in inference.glob(f"*/{self.config.report_filename}") if path.is_file())

    def _check_reused_aggregate(self, trial_dir: Path, report: Path, split: str) -> None:
        """Names a reused seed aggregate and warns when the trial holds seeds it did not average or a seed completed an inference after it.

        Args:
            trial_dir : Seeded trial directory.
            report    : Reused aggregate report.
            split     : Split whose newer inference makes the aggregate stale.
        """
        averaged  = self._aggregate_record(report)["seed_runs"]
        seeds     = TrialTree.seed_runs(trial_dir)
        uncovered = [f"{trial_dir.name}/{seed.name}" for seed in seeds if seed.name not in averaged]

        self.freshness.check(f"seed aggregate of {trial_dir.name}", report.parent, seeds, split, uncovered)

    def _aggregate_seeds(self, trial_dir: Path) -> None:
        """Writes the seed-aggregate report of a seeded trial as the seed comparison entry does, logging through the mail run's logger.

        Args:
            trial_dir : Trial directory holding the seed runs.
        """
        from configuration.comparison                              import SeedComparisonConfig
        from pipelines.backbone.inference.analysis.seed_comparison import SeedComparison

        self.logger.info(f"{trial_dir.name}: writing the seed-aggregate report over {len(TrialTree.seed_runs(trial_dir))} seed run(s)")

        reports = SeedComparison(SeedComparisonConfig(runs_dir=trial_dir.parent, group_tags=[trial_dir.name]), self.logger).run()

        self.logger.ok(f"{trial_dir.name}: seed-aggregate report written to {reports[0]}")

    def _check_aggregate_split(self, trial_dir: Path, report: Path, split: str) -> None:
        """Warns when a seed aggregate averaged an inference of another split, or an unfinished one, for any seed.

        The seed comparison averages each seed's newest inference whatever
        split it scored, while the trial comparison reads one split.

        Args:
            trial_dir : Seeded trial directory.
            report    : Mailed aggregate report.
            split     : Split the trial comparison reads.
        """
        outside = []
        for seed, record in self._aggregate_record(report)["seed_inference"].items():
            stamp_name = record["stamp"]
            stamp      = trial_dir / seed / self.config.inference_dirname / stamp_name

            if not CompletionMarker.is_complete(stamp):
                outside.append(f"{seed} ({stamp_name}, unfinished)")
            elif CompletionMarker.payload(stamp)["split"] != split:
                outside.append(f"{seed} ({stamp_name}, split {CompletionMarker.payload(stamp)['split']})")

        if outside:
            self.logger.warning(f"Seed aggregate {report.parent.name} of {trial_dir.name} averages inference outside the '{split}' split the comparison reads: {', '.join(outside)}")

    def _aggregate_record(self, report: Path) -> dict:
        """Returns the metrics file of a seed aggregate, which records the seed runs it averaged and the inference stamp read for each."""
        from configuration.comparison import SeedComparisonConfig

        return FileIO.load_json(report.parent / SeedComparisonConfig.metrics_filename)

    def _trial_report(self, trial_dir: Path, split: str) -> Path:
        """Returns the report mailed for one trial: a single run's newest report on the split, or a seeded trial's newest seed aggregate, written first when missing or asked for.

        Args:
            trial_dir : Trial directory whose report is located.
            split     : Split a single run's report must have scored, the one the trial comparison reads.

        Raises:
            FileNotFoundError : If a single-run trial holds no report of a completed inference on the split.
        """
        if not TrialTree.seed_runs(trial_dir):
            return self._run_report(trial_dir, split)

        aggregates = self._seed_aggregates(trial_dir)
        if aggregates and not self.config.regenerate_reports:
            self._check_reused_aggregate(trial_dir, aggregates[-1], split)
        else:
            self._aggregate_seeds(trial_dir)
            aggregates = self._seed_aggregates(trial_dir)

        self._check_aggregate_split(trial_dir, aggregates[-1], split)

        return aggregates[-1]

    def _check_reused_comparison(self, stamp: Path, split: str) -> None:
        """Names a reused comparison stamp and warns when it misses a trial, predates an inference of its runs, scored another split or carries no summary to tell.

        Args:
            stamp : Reused comparison stamp directory.
            split : Split the mailed trial reports are taken from.
        """
        trials   = self.experiment.trials()
        runs     = [run for trial in trials for run in TrialTree.runs(trial)]
        coverage = self.experiment.stamp_coverage(stamp)

        uncovered = [trial.name for trial in trials if trial.name not in coverage[1]] if coverage is not None else []

        self.freshness.check("trial comparison", stamp, runs, split, uncovered)

        if coverage is None:
            self.logger.warning(f"Reused trial comparison {stamp.name} holds no {self.experiment.SUMMARY_FILENAME}, so the split and the trials it covers cannot be checked; {ReportFreshness.HINT}")
        elif coverage[0] != split:
            self.logger.warning(f"Reused trial comparison {stamp.name} scored split '{coverage[0]}' while the trial reports are taken from '{split}'; {ReportFreshness.HINT}")

    def _comparison_stamp(self, split: str) -> Path | None:
        """Returns the comparison stamp to gather, writing one when missing or asked for; None for an experiment of one trial.

        Args:
            split : Split the mailed trial reports are taken from, which a reused stamp is checked against.
        """
        if len(self.experiment.trials()) < 2:
            self.logger.warning(f"{self.experiment.root.name}: a single trial, no trial comparison is written or attached")
            return None

        root   = self.experiment.comparison_root(self.config.comparison_dirname)
        stamps = sorted(child for child in root.iterdir() if child.is_dir()) if root.is_dir() else []

        if stamps and not self.config.regenerate_reports:
            self._check_reused_comparison(stamps[-1], split)
            return stamps[-1]

        self.logger.info(f"{'Regenerating' if stamps else 'Writing'} the trial comparison of {self.experiment.root}")

        return self.experiment.write_comparison(self.config.comparison_dirname, self.logger)

    def _comparison_reports(self, stamp: Path | None) -> list[Path]:
        """Returns the report files of the comparison stamp.

        Raises:
            FileNotFoundError : If a report is missing from the stamp.
        """
        if stamp is None:
            return []

        paths = []
        for name in self.experiment.comparison_report_names(self.config.comparison_reports):
            path = stamp / name
            if not path.is_file():
                raise FileNotFoundError(f"Comparison report '{name}' is missing from {stamp}")
            paths.append(path)

        return paths

    def _stage_file(self, path: Path, output_name: str) -> Path:
        """Stages one gathered file, rewriting the image links of a markdown report and copying anything else verbatim."""
        if path.suffix == ".md":
            return self._stage_report(path, output_name)

        return self._stage_copy(path, output_name)

    def stage(self) -> List[Path]:
        """Writes what is missing, then gathers every report and configuration file of the experiment.

        Returns:
            The staged file paths: one report per trial, the comparison reports, the experiment's further reports, then the configuration files.
        """
        self.logger.subsection(f"Gathering {self.experiment.KIND} experiment: {self.experiment.root.name}")
        FileIO.ensure_dirs(self.staging_dir)

        trials = self.experiment.trials()
        split  = self.experiment.inference_split()

        staged  = [self._stage_report(self._trial_report(trial, split), f"{trial.name}.md") for trial in trials]
        staged += [self._stage_file(path, f"comparison_{path.name}") for path in self._comparison_reports(self._comparison_stamp(split))]
        staged += [self._stage_file(path, name) for path, name in self.experiment.extra_reports(self.config.regenerate_reports, self.logger)]
        staged += [self._stage_copy(path) for path in self.experiment.config_files(self.logger)]

        return staged


class ReportMailer:
    """Sends one email with a set of staged files attached.

    Attributes:
        entry_config     : Entry config carrying the SMTP endpoint, the addresses and the dry-run flag.
        logger           : Logger for submission messages.
        ATTACHMENT_TYPES : MIME type per attachment file extension.
    """

    ATTACHMENT_TYPES = {
        ".md"   : ("text", "markdown"),
        ".json" : ("application", "json"),
        ".zip"  : ("application", "zip"),
    }

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        """Initializes the mailer with its entry config and logger."""
        self.entry_config = entry_config
        self.logger       = logger

    def _password(self) -> str:
        """Returns the SMTP password read from the configured environment variable.

        Raises:
            EnvironmentError : If the variable is unset or empty.
        """
        value = os.environ.get(self.entry_config.password_env, "")

        if not value:
            raise EnvironmentError(f"SMTP password not found: export {self.entry_config.password_env}=<app password> before running, or set dry_run=True to stage without sending")

        return value

    def _message(self, subject: str, body: str, attachments: List[Path]) -> EmailMessage:
        """Builds the email with every staged file attached.

        Args:
            subject     : Message subject line.
            body        : Plain-text message body.
            attachments : Files attached under their staged names.

        Returns:
            The assembled message.
        """
        message            = EmailMessage()
        message["From"]    = self.entry_config.sender
        message["To"]      = self.entry_config.recipient
        message["Subject"] = subject
        message.set_content(body)

        for path in attachments:
            maintype, subtype = self.ATTACHMENT_TYPES.get(path.suffix, ("application", "octet-stream"))
            message.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)

        return message

    def send(self, subject: str, body: str, attachments: List[Path]) -> None:
        """Assembles and submits one message, or only logs it on a dry run.

        The submission is announced with the message size before the
        connection opens and timed until it closes, and every blocking SMTP
        call gives up after the configured timeout, so a stall ends in an
        error instead of a silent hang.

        Args:
            subject     : Message subject line.
            body        : Plain-text message body.
            attachments : Files attached to the message.

        Raises:
            EnvironmentError : If the password variable is unset, checked before any connection opens.
            OSError          : If the submission fails, including a connect, login or upload stalled past the timeout, raised as a TimeoutError or as the SMTPServerDisconnected smtplib turns it into; the failure is logged with its duration first.
        """
        message = self._message(subject, body, attachments)
        size_mb = len(message.as_bytes()) / (1024 * 1024)

        if self.entry_config.dry_run:
            self.logger.info(f"Dry run: not sending '{subject}' with {len(attachments)} attachment(s), {size_mb:.1f} MB encoded")
            return

        self.logger.info(f"Sending '{subject}' via {self.entry_config.smtp_host}:{self.entry_config.smtp_port}: {len(attachments)} attachment(s), {size_mb:.1f} MB encoded, timeout {self.entry_config.smtp_timeout_s:g} s")

        password = self._password()
        start    = time.monotonic()

        try:
            with smtplib.SMTP_SSL(self.entry_config.smtp_host, self.entry_config.smtp_port, timeout=self.entry_config.smtp_timeout_s) as smtp:
                smtp.login(self.entry_config.sender, password)
                smtp.send_message(message)
        except OSError as error:
            self.logger.error(f"SMTP submission of '{subject}' failed after {time.monotonic() - start:.1f} s: {type(error).__name__}: {error}")
            raise

        self.logger.ok(f"Sent '{subject}' to {self.entry_config.recipient} with {len(attachments)} attachment(s) in {time.monotonic() - start:.1f} s")



class MailTarget:
    """One kind of source the mailer gathers and sends bundles for.

    Subclasses supply the selector that discovers the sources, the label that
    identifies one of them, and the gathering that turns it into the attachment
    groups, one group per message.

    Attributes:
        entry_config : Entry config carrying the roots, the filters and the gathered paths.
        logger       : Logger passed to the selector and the bundle.
        NOUN         : Name of one source, used in the log lines and the message body.
        BUNDLE_LABEL : Opening phrase of the message body, completed by the source label.
    """

    NOUN         = "source"
    BUNDLE_LABEL = "Bundle of"

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        """Initializes the target with its entry config and logger."""
        self.entry_config = entry_config
        self.logger       = logger

    def settings(self) -> dict:
        """Returns the target's own settings for the run's configuration table: its root, its filter and what shapes its bundles."""
        raise NotImplementedError

    def select(self) -> List[Path]:
        """Returns the source directories chosen by this target's filter."""
        raise NotImplementedError

    def label(self, source_dir: Path) -> str:
        """Returns the mailed identity of one source directory."""
        raise NotImplementedError

    def stage(self, source_dir: Path) -> List[List[Path]]:
        """Gathers one source and returns its attachments grouped by message."""
        raise NotImplementedError


class ExperimentMailTarget(MailTarget):
    """Mails one message per experiment with its gathered reports attached."""

    NOUN         = "experiment"
    BUNDLE_LABEL = "Report bundle of experiment"

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        """Initializes the target and the selector that finds the experiments."""
        super().__init__(entry_config, logger)
        self.selector = ExperimentSelector(entry_config.experiments_dir, entry_config.comparison_dirname, logger)

    def settings(self) -> dict:
        """Returns the experiments root, the filter, the regeneration switch and the image mode."""
        return {
            "Experiments dir"    : str(self.entry_config.experiments_dir),
            "Filter"             : ", ".join(self.entry_config.experiment_filter) or "all",
            "Regenerate reports" : self.entry_config.regenerate_reports,
            "Embed images"       : self.entry_config.embed_images,
        }

    def select(self) -> List[Path]:
        """Returns the root of every experiment chosen by the configured filter."""
        return [experiment.root for experiment in self.selector.select_experiments(self.entry_config.experiment_filter)]

    def label(self, source_dir: Path) -> str:
        """Returns the experiment's label, its own name prefixed by its parent directory."""
        return self.selector.experiments[Path(source_dir)].label

    def stage(self, source_dir: Path) -> List[List[Path]]:
        """Writes the missing reports of the experiment and stages its bundle as the single message's attachments."""
        experiment = self.selector.experiments[Path(source_dir)]
        config     = self.entry_config.to_config(experiment.label)

        return [ExperimentReportBundle(experiment, config, self.logger).stage()]


class RunContentMailTarget(MailTarget):
    """Mails everything one training run holds, minus the excluded content, as zip volumes."""

    NOUN         = "run"
    BUNDLE_LABEL = "Content bundle of run"

    def settings(self) -> dict:
        """Returns the runs root, the filter, the per-file cap and the excluded paths."""
        return {
            "Runs dir"       : str(self.entry_config.runs_dir),
            "Filter"         : ", ".join(self.entry_config.run_filter) or "all",
            "Max file MB"    : self.entry_config.run_max_file_mb,
            "Excluded paths" : ", ".join(self.entry_config.run_excluded_paths) or "none",
        }

    def select(self) -> List[Path]:
        """Returns the run directories chosen by the configured filter."""
        return RunContentSelector(self.entry_config.runs_dir, self.logger).resolve(self.entry_config.run_filter)

    def label(self, source_dir: Path) -> str:
        """Returns the run's label, its path below the runs root."""
        return self.entry_config.run_label(source_dir)

    def stage(self, source_dir: Path) -> List[List[Path]]:
        """Packs the run's content and returns one attachment group per volume."""
        config = self.entry_config.to_run_config(source_dir)
        return [[archive] for archive in RunContentBundle(config, self.logger).stage()]


class DatasetMailTarget(MailTarget):
    """Mails the generation artifacts of one dataset as zip volumes."""

    NOUN         = "dataset"
    BUNDLE_LABEL = "Artifact bundle of dataset"

    def settings(self) -> dict:
        """Returns the dataset root and the filter."""
        return {
            "Datasets dir" : str(self.entry_config.datasets_dir),
            "Filter"       : ", ".join(self.entry_config.dataset_filter) or "all",
        }

    def select(self) -> List[Path]:
        """Returns the dataset directories chosen by the configured filter."""
        return DatasetSelector(self.entry_config.datasets_dir, self.logger).resolve(self.entry_config.dataset_filter)

    def label(self, source_dir: Path) -> str:
        """Returns the dataset's label, its own name prefixed by the dataset root."""
        return self.entry_config.dataset_label(source_dir)

    def stage(self, source_dir: Path) -> List[List[Path]]:
        """Packs the dataset's artifacts and returns one attachment group per volume."""
        config = self.entry_config.to_dataset_config(source_dir)
        return [[archive] for archive in ArtifactBundle(config, self.logger).stage()]


class ParamRunMailTarget(MailTarget):
    """Mails the extraction and analysis artifacts of one parameter run as zip volumes."""

    NOUN         = "parameter run"
    BUNDLE_LABEL = "Artifact bundle of parameter run"

    def settings(self) -> dict:
        """Returns the parameter-run root and the filter."""
        return {
            "Params dir" : str(self.entry_config.params_dir),
            "Filter"     : ", ".join(self.entry_config.param_run_filter) or "all",
        }

    def select(self) -> List[Path]:
        """Returns the parameter-run directories chosen by the configured filter."""
        return ParamRunSelector(self.entry_config.params_dir, self.logger).resolve(self.entry_config.param_run_filter)

    def label(self, source_dir: Path) -> str:
        """Returns the run's label, its own name prefixed by the dataset it was fitted on."""
        return self.entry_config.param_run_label(source_dir)

    def stage(self, source_dir: Path) -> List[List[Path]]:
        """Packs the run's artifacts and returns one attachment group per volume."""
        config = self.entry_config.to_param_config(source_dir)
        return [[archive] for archive in ArtifactBundle(config, self.logger).stage()]


class ReportMailBatch:
    """Mails the bundle of every selected source of the configured target.

    Attributes:
        entry_config : Entry config naming the target, its root and filter, and the mail settings.
        logger       : Logger for selection and summary messages.
        mailer       : Mailer submitting the messages.
        target       : Target resolved from the entry config, supplying selection and gathering.
        TARGETS      : Target class per accepted ``target`` value.
    """

    TARGETS = {
        "experiments"      : ExperimentMailTarget,
        "runs"             : RunContentMailTarget,
        "datasets"         : DatasetMailTarget,
        "param_extraction" : ParamRunMailTarget,
    }

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        """Initializes the batch with its entry config and logger.

        Raises:
            ValueError : If the configured target is not one of the accepted values.
        """
        if entry_config.target not in self.TARGETS:
            raise ValueError(f"Unknown mail target '{entry_config.target}'; choose one of {', '.join(self.TARGETS)}")

        self.entry_config = entry_config
        self.logger       = logger
        self.mailer       = ReportMailer(entry_config, logger)
        self.target       = self.TARGETS[entry_config.target](entry_config, logger)

    def _subject(self, label: str, index: int, total: int) -> str:
        """Returns the subject of one message, numbered only when the bundle is split.

        Args:
            label : Mailed identity of the source.
            index : Position of the message in the bundle, starting at one.
            total : Number of messages the bundle is sent as.

        Returns:
            The subject line.
        """
        if total == 1:
            return f"{self.entry_config.subject_prefix}: {label}"

        return f"{self.entry_config.subject_prefix}: {label} [{index}/{total}]"

    def _body(self, source_dir: Path, label: str, attachments: List[Path], index: int, total: int) -> str:
        """Returns the plain-text body naming the source and listing the attached files.

        Args:
            source_dir  : Source the message reports on.
            label       : Mailed identity of the source.
            attachments : Files attached to this message.
            index       : Position of the message in the bundle, starting at one.
            total       : Number of messages the bundle is sent as.

        Returns:
            The message body.
        """
        lines  = [f"{self.target.BUNDLE_LABEL} '{label}'.", ""]
        lines += [f"Source directory: {source_dir}", ""]

        if total > 1:
            lines += [f"Part {index} of {total}; unpack every part into one directory to rebuild the run tree.", ""]

        lines += ["Attached files:"]
        lines += [f"  - {path.name} ({path.stat().st_size / (1024 * 1024):,.1f} MB)" for path in attachments]

        return "\n".join(lines) + "\n"

    def _mail_source(self, source_dir: Path) -> dict:
        """Gathers and mails one source's bundle, as one message per attachment group.

        Args:
            source_dir : Source whose artifacts are gathered and sent.

        Returns:
            Mapping with the source directory, the label, the staged paths and the subjects used.
        """
        label  = self.target.label(source_dir)
        groups = self.target.stage(source_dir)

        subjects = []
        for index, attachments in enumerate(groups, start=1):
            subject = self._subject(label, index, len(groups))
            self.mailer.send(subject, self._body(source_dir, label, attachments, index, len(groups)), attachments)
            subjects.append(subject)

        return {
            "target"        : self.entry_config.target,
            "source_dir"    : str(source_dir),
            "label"         : label,
            "n_messages"    : len(groups),
            "n_attachments" : sum(len(group) for group in groups),
            "staged_paths"  : [path for group in groups for path in group],
            "subjects"      : subjects,
        }

    def _log_settings(self) -> None:
        """Logs the effective settings of the run: the target with its root and filter, the staging, the SMTP endpoint, the addresses and the dry-run flag."""
        self.logger.kv_table({
            "Target"            : self.entry_config.target,
            **self.target.settings(),
            "Staging dir"       : str(self.entry_config.staging_dir),
            "Max attachment MB" : self.entry_config.max_attachment_mb,
            "SMTP"              : f"{self.entry_config.smtp_host}:{self.entry_config.smtp_port}, timeout {self.entry_config.smtp_timeout_s:g} s",
            "Sender"            : self.entry_config.sender,
            "Recipient"         : self.entry_config.recipient,
            "Dry run"           : self.entry_config.dry_run,
        }, title="Configuration")

    def _write_record(self, results: list[dict], started_at: str) -> Path:
        """Writes the run's mailing record, one summary per source, next to the staged files.

        Args:
            results    : One mailing summary per source, in selection order.
            started_at : Timestamp the run started at.

        Returns:
            Path of the written ``mail_<target>_<tag>.json`` file in the staging directory.
        """
        payload = {
            "target"      : self.entry_config.target,
            "recipient"   : self.entry_config.recipient,
            "dry_run"     : self.entry_config.dry_run,
            "started_at"  : started_at,
            "finished_at" : RunTag.timestamp(),
            "sources"     : results,
        }

        return FileIO.save_json(payload, Path(self.entry_config.staging_dir) / f"mail_{self.entry_config.target}_{RunTag.now()}.json", indent=2)

    def run(self) -> list[dict]:
        """Selects the sources of the configured target, gathers and mails each one and records what was sent.

        Returns:
            One mailing summary per source, in selection order.
        """
        started_at = RunTag.timestamp()

        if self.entry_config.dry_run:
            self.logger.section(f"Staging {self.target.NOUN} results for {self.entry_config.recipient} (dry run, nothing is sent)")
        else:
            self.logger.section(f"Mailing {self.target.NOUN} results to {self.entry_config.recipient}")

        self._log_settings()

        source_dirs = self.target.select()
        results     = [self._mail_source(source_dir) for source_dir in source_dirs]

        record     = self._write_record(results, started_at)
        n_files    = sum(result["n_attachments"] for result in results)
        n_messages = sum(result["n_messages"] for result in results)

        self.logger.info(f"Mailing record written to {record}")

        if self.entry_config.dry_run:
            self.logger.ok(f"Dry run: staged {n_files} file(s) for {n_messages} message(s) across {len(results)} {self.target.NOUN}(s) under {self.entry_config.staging_dir}; nothing was sent")
        else:
            self.logger.ok(f"Mailed {n_files} file(s) in {n_messages} message(s) across {len(results)} {self.target.NOUN}(s), staged under {self.entry_config.staging_dir}")

        return results
