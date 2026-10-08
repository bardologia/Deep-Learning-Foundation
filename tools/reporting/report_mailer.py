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

from email.message import EmailMessage
from pathlib       import Path
from typing        import List

from configuration.diagnostics                 import ExperimentReportMailConfig, ReportMailEntryConfig
from pipelines.shared.inference.run_classifier import RunDirectoryWalk
from pipelines.shared.training.seed_sweep      import SeedSet
from tools.data.io                             import FileIO
from tools.reporting.artifact_bundles          import ArtifactBundle, DatasetSelector, ParamRunSelector, RunContentBundle, RunContentSelector
from tools.reporting.report_collector          import ReportImageRewriter
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


class Experiment:
    """One mailable experiment: a directory of trial runs, their comparison and the configuration they ran with.

    Attributes:
        KIND       : Name of the experiment kind in tables and messages.
        root       : Directory identifying the experiment.
        trials_dir : Directory holding the trial runs.
        selected   : Trials the experiment is restricted to, or None for every trial under the trials directory.
    """

    KIND = "sweep"

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

    def comparison_root(self, comparison_dirname: str) -> Path:
        """Returns the directory holding the timestamped comparison stamps."""
        return self.trials_dir / comparison_dirname

    def comparison_report_names(self, configured: list[str]) -> list[str]:
        """Returns the report files gathered from a comparison stamp."""
        return list(configured)

    def write_comparison(self, comparison_dirname: str, logger) -> Path:
        """Compares the trials as the trial-comparison entry does and returns the written stamp.

        Args:
            comparison_dirname : Directory under the trials directory receiving the stamp.
            logger             : Logger for the collection and the written files.
        """
        from configuration.comparison               import TrialComparisonConfig
        from pipelines.comparison.comparison_report import TrialComparisonReport
        from pipelines.comparison.trial_collector   import TrialCollector

        config  = TrialComparisonConfig(runs_dir=self.trials_dir, run_tags=[trial.name for trial in self.trials()])
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

    KIND = "cross-validation"

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
        """Returns the cross-fold report stamp to gather, writing one from the fold tree when missing or asked for."""
        from pipelines.cross_validation.stages import CrossValidationReportRebuilder

        stamps = self._stamps()
        if stamps and not regenerate:
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

    KIND = "benchmark"

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

    def comparison_root(self, comparison_dirname: str) -> Path:
        """Returns the run's own comparison directory, whatever the sweep comparison directory is called."""
        return self.root / self.COMPARISON_DIRNAME

    def comparison_report_names(self, configured: list[str]) -> list[str]:
        """Returns the benchmark comparison's own report files."""
        return list(self.COMPARISON_REPORTS)

    def write_comparison(self, comparison_dirname: str, logger) -> Path:
        """Runs the benchmark comparison stage over the run with the configuration it recorded and returns the stamp.

        Args:
            comparison_dirname : Ignored; the benchmark writes under its own comparison directory.
            logger             : Logger for the stage.
        """
        from configuration.benchmark                    import BenchmarkConfig
        from pipelines.benchmark.stages                 import ComparisonStage
        from pipelines.shared.orchestration.resumed_run import StagedRunLocator
        from tools.runtime.config_cli                   import ConfigCli

        config = ConfigCli.load_resolved(BenchmarkConfig(), StagedRunLocator.config_path(self.root))
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
    experiment and at each trial, so run trees are never entered, and gives up
    below ``MAX_DEPTH``. The root may also be a trial itself, which is then
    mailed as an experiment of one, or the folds or training directory of a
    run, which resolves to that run.

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
        """Yields the experiments under a directory, stopping at each trial, each experiment and the depth cap."""
        if TrialTree.is_trial(directory):
            return

        experiment = self.classify(directory)
        if experiment is not None:
            yield experiment
            return

        if depth >= self.MAX_DEPTH:
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
        self.logger.ok(f"Staged {output_name}")
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
        self.logger.ok(f"Staged {destination.name}")
        return destination


class ExperimentReportBundle(ReportBundle):
    """Gathers one experiment's reports and configuration into a staging directory, writing what is missing first.

    Attributes:
        config     : Gathering config naming the staging directory, the comparison layout and the regeneration switch.
        experiment : Experiment being gathered.
    """

    def __init__(self, experiment: Experiment, config: ExperimentReportMailConfig, logger) -> None:
        """Initializes the bundle with the experiment, its gathering config and logger."""
        super().__init__(config.staging_dir, config.embed_images, logger)
        self.config     = config
        self.experiment = experiment

    def _reports_under(self, run_dir: Path) -> list[Path]:
        """Returns every report under a run's inference directory, oldest stamp first.

        Spatial families stamp directly under the inference directory and the
        other families one level deeper, so both depths are searched.

        Args:
            run_dir : Run or trial directory whose inference reports are listed.
        """
        inference = run_dir / self.config.inference_dirname
        if not inference.is_dir():
            return []

        reports = list(inference.glob(f"*/{self.config.report_filename}")) + list(inference.glob(f"*/*/{self.config.report_filename}"))

        return sorted((path for path in reports if path.is_file()), key=lambda path: path.parent.name)

    def _aggregate_seeds(self, trial_dir: Path) -> None:
        """Writes the seed-aggregate report of a seeded trial, as the seed comparison entry does.

        Args:
            trial_dir : Trial directory holding the seed runs.
        """
        from configuration.comparison                              import SeedComparisonConfig
        from pipelines.backbone.inference.analysis.seed_comparison import SeedComparison

        self.logger.info(f"{trial_dir.name}: writing the seed-aggregate report over {len(TrialTree.seed_runs(trial_dir))} seed run(s)")

        SeedComparison(SeedComparisonConfig(runs_dir=trial_dir.parent, group_tags=[trial_dir.name]), self.logger).run()

    def _trial_report(self, trial_dir: Path) -> Path:
        """Returns the newest trial-level report, writing the seed aggregate of a seeded trial when missing or asked for.

        Args:
            trial_dir : Trial directory whose report is located.

        Raises:
            FileNotFoundError : If a single-run trial holds no inference report, so there is nothing to mail for it.
        """
        seeded = bool(TrialTree.seed_runs(trial_dir))

        if seeded and (self.config.regenerate_reports or not self._reports_under(trial_dir)):
            self._aggregate_seeds(trial_dir)

        reports = self._reports_under(trial_dir)
        if not reports:
            raise FileNotFoundError(f"No '{self.config.report_filename}' under {trial_dir / self.config.inference_dirname}; run inference on this trial before mailing it")

        return reports[-1]

    def _comparison_stamp(self) -> Path | None:
        """Returns the comparison stamp to gather, writing one when missing or asked for; None for an experiment of one trial."""
        if len(self.experiment.trials()) < 2:
            self.logger.warning(f"{self.experiment.root.name}: a single trial, no trial comparison is written or attached")
            return None

        root   = self.experiment.comparison_root(self.config.comparison_dirname)
        stamps = sorted(child for child in root.iterdir() if child.is_dir()) if root.is_dir() else []

        if stamps and not self.config.regenerate_reports:
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

        staged  = [self._stage_report(self._trial_report(trial), f"{trial.name}.md") for trial in trials]
        staged += [self._stage_file(path, f"comparison_{path.name}") for path in self._comparison_reports(self._comparison_stamp())]
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

        Args:
            subject     : Message subject line.
            body        : Plain-text message body.
            attachments : Files attached to the message.
        """
        message = self._message(subject, body, attachments)

        if self.entry_config.dry_run:
            self.logger.info(f"Dry run: not sending '{subject}' with {len(attachments)} attachment(s)")
            return

        with smtplib.SMTP_SSL(self.entry_config.smtp_host, self.entry_config.smtp_port) as smtp:
            smtp.login(self.entry_config.sender, self._password())
            smtp.send_message(message)

        self.logger.ok(f"Sent '{subject}' to {self.entry_config.recipient} with {len(attachments)} attachment(s)")



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

    def run(self) -> list[dict]:
        """Selects the sources of the configured target and gathers and mails each one.

        Returns:
            One mailing summary per source, in selection order.
        """
        self.logger.section(f"Mailing {self.target.NOUN} results to {self.entry_config.recipient}")

        source_dirs = self.target.select()
        results     = [self._mail_source(source_dir) for source_dir in source_dirs]

        self.logger.ok(f"Mailed {sum(result['n_attachments'] for result in results)} file(s) in {sum(result['n_messages'] for result in results)} message(s) across {len(results)} {self.target.NOUN}(s)")
        return results
