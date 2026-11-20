
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

    SKIP = ("logs", "batch_train_logs", "pipeline", "reports", "comparison", "inference", "meta", "docs", "tensorboard", "cubes", "figures", "animations", "data", "ddp")

    @staticmethod
    def is_run(directory: Path) -> bool:
        return (Path(directory) / RunDirectoryWalk.MARKER).is_dir()

    @classmethod
    def seed_runs(cls, directory: Path) -> list[Path]:
        directory = Path(directory)
        if not directory.is_dir():
            return []

        return sorted(child for child in directory.iterdir() if child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) and cls.is_run(child))

    @classmethod
    def is_trial(cls, directory: Path) -> bool:
        return cls.is_run(directory) or bool(cls.seed_runs(directory))

    @classmethod
    def candidates(cls, directory: Path) -> list[Path]:
        directory = Path(directory)
        if not directory.is_dir():
            return []

        return sorted(child for child in directory.iterdir() if child.is_dir() and not child.name.startswith((".", "_")) and child.name not in cls.SKIP)

    @classmethod
    def trials(cls, directory: Path) -> list[Path]:
        return [child for child in cls.candidates(directory) if cls.is_trial(child)]

    @classmethod
    def runs(cls, trial: Path) -> list[Path]:
        seeds = cls.seed_runs(trial)
        return seeds if seeds else [Path(trial)]

    @staticmethod
    def inference_stamps(inference_dir: Path, split: str | None) -> list[Path]:
        base = Path(inference_dir)

        if split is None:
            stamps = [stamp for stamp in base.iterdir() if stamp.is_dir() and CompletionMarker.is_complete(stamp)] if base.is_dir() else []
        else:
            stamps = CompletedInference.stamps(base, split)

        return sorted(stamps, key=lambda stamp: stamp.name)


class ReportFreshness:

    HINT = "set regenerate_reports=True to rewrite it"

    def __init__(self, inference_dirname: str, logger) -> None:
        self.inference_dirname = inference_dirname
        self.logger            = logger

    @staticmethod
    def _written_at(stamp: Path) -> float:
        times = [path.stat().st_mtime for path in stamp.iterdir() if path.is_file()]
        return max(times)

    def _newer_inference(self, runs: list[Path], split: str | None, since: float) -> list[Path]:
        stamps = [stamp for run in runs for stamp in TrialTree.inference_stamps(run / self.inference_dirname, split)]
        newer  = [stamp for stamp in stamps if CompletionMarker.path(stamp).stat().st_mtime > since]

        return sorted(newer, key=lambda stamp: CompletionMarker.path(stamp).stat().st_mtime)

    def check(self, kind: str, stamp: Path, runs: list[Path], split: str | None, uncovered: list[str]) -> None:
        self.logger.info(f"Reusing {kind} {stamp}")

        newer = self._newer_inference(runs, split, self._written_at(stamp))
        if newer:
            self.logger.warning(f"Reused {kind} {stamp.name} predates {len(newer)} completed inference(s) of its runs, the newest {newer[-1]}; {self.HINT}")

        if uncovered:
            self.logger.warning(f"Reused {kind} {stamp.name} does not cover {', '.join(uncovered)}; {self.HINT}")


class Experiment:

    KIND             = "sweep"
    SEARCH_BELOW     = True
    SUMMARY_FILENAME = "comparison_summary.json"

    def __init__(self, root: Path, trials_dir: Path, selected: list[Path] | None = None) -> None:
        self.root       = Path(root)
        self.trials_dir = Path(trials_dir)
        self.selected   = list(selected) if selected is not None else None

    @property
    def label(self) -> str:
        return f"{self.root.parent.name}/{self.root.name}"

    def trials(self) -> list[Path]:
        trials = self.selected if self.selected is not None else TrialTree.trials(self.trials_dir)
        return trials

    def inference_split(self) -> str:
        return TrialComparisonConfig.inference_split

    def comparison_root(self, comparison_dirname: str) -> Path:
        return self.trials_dir / comparison_dirname

    def comparison_report_names(self, configured: list[str]) -> list[str]:
        return list(configured)

    def stamp_coverage(self, stamp: Path) -> tuple[str, list[str]] | None:
        path = stamp / self.SUMMARY_FILENAME
        if not path.is_file():
            return None

        payload = FileIO.load_json(path)

        return payload["inference_split"], [trial["name"] for trial in payload["trials"]]

    def write_comparison(self, comparison_dirname: str, logger) -> Path:
        from pipelines.comparison.comparison_report import TrialComparisonReport
        from pipelines.comparison.trial_collector   import TrialCollector

        config  = TrialComparisonConfig(runs_dir=self.trials_dir, run_tags=[trial.name for trial in self.trials()], inference_split=self.inference_split())
        out_dir = self.comparison_root(comparison_dirname) / RunTag.now()

        collector = TrialCollector(runs_dir=self.trials_dir, run_tags=config.run_tags, logger=logger, inference_split=config.inference_split)
        records   = collector.collect()

        TrialComparisonReport(records=records, out_dir=out_dir, compare_images=config.compare_images, compare_gifs=config.compare_gifs, embed_images=config.embed_images, inference_split=config.inference_split, logger=logger, seed_dispersion=collector.seed_dispersion).write_all()

        return out_dir

    def extra_reports(self, regenerate: bool, logger) -> list[tuple[Path, str]]:
        return []

    def config_files(self, logger) -> list[Path]:
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
        found = []
        for path in candidates:
            if path.is_file():
                found.append(path)
            else:
                logger.warning(f"No '{path.relative_to(self.root) if path.is_relative_to(self.root) else path}' under {self.root}; it is not attached")

        return found


class CrossValidationExperiment(Experiment):

    KIND         = "cross-validation"
    SEARCH_BELOW = False

    FOLDS_DIRNAME     = "folds"
    REPORTS_DIRNAME   = "reports"
    AGGREGATE_REPORTS = ("cv_aggregate_report.md", "cv_summary.json")
    SPLIT_REPORTS     = ("benchmark_overview.md", "metrics_comparison.md")
    PIPELINE_FILES    = ("pipeline/resolved_config.json", "pipeline/state.json")

    def __init__(self, root: Path) -> None:
        super().__init__(root, Path(root) / self.FOLDS_DIRNAME)

    @classmethod
    def is_run(cls, directory: Path) -> bool:
        return bool(TrialTree.trials(Path(directory) / cls.FOLDS_DIRNAME))

    def _stamps(self) -> list[Path]:
        reports = self.root / self.REPORTS_DIRNAME
        if not reports.is_dir():
            return []

        return sorted(child for child in reports.iterdir() if child.is_dir())

    def _report_stamp(self, regenerate: bool, logger) -> Path:
        from pipelines.cross_validation.stages import CrossValidationReportRebuilder

        stamps = self._stamps()
        if stamps and not regenerate:
            runs = [run for trial in self.trials() for run in TrialTree.runs(trial)]
            ReportFreshness(InferenceLayout.DIRNAME, logger).check("cross-fold report", stamps[-1], runs, None, [])
            return stamps[-1]

        logger.info(f"{'Regenerating' if stamps else 'Writing'} the cross-fold report of {self.root} from its fold tree")

        return CrossValidationReportRebuilder(self.root, logger).rebuild()

    def extra_reports(self, regenerate: bool, logger) -> list[tuple[Path, str]]:
        stamp = self._report_stamp(regenerate, logger)
        pairs = []

        for name in self.AGGREGATE_REPORTS:
            path = stamp / name
            pairs.append((path, name))

        split_dirs = sorted(child for child in stamp.iterdir() if child.is_dir())
        if not split_dirs:
            logger.warning(f"No split directory under {stamp}; the run compared its folds on training checkpoints alone, so no per-split comparison is attached")

        for split_dir in split_dirs:
            for name in self.SPLIT_REPORTS:
                path = split_dir / name
                pairs.append((path, f"{split_dir.name}_{name}"))

        return pairs

    def config_files(self, logger) -> list[Path]:
        return self._existing([self.root / relative for relative in self.PIPELINE_FILES], logger)


class BenchmarkExperiment(Experiment):

    KIND         = "benchmark"
    SEARCH_BELOW = False

    TRAINING_DIRNAME   = "training"
    COMPARISON_DIRNAME = "comparison"
    COMPARISON_REPORTS = ("benchmark_overview.md", "metrics_comparison.md")
    PIPELINE_FILES     = ("pipeline/resolved_config.json", "pipeline/state.json")

    def __init__(self, root: Path) -> None:
        super().__init__(root, Path(root) / self.TRAINING_DIRNAME)

    @classmethod
    def is_run(cls, directory: Path) -> bool:
        directory = Path(directory)
        return (directory / "pipeline").is_dir() and bool(TrialTree.trials(directory / cls.TRAINING_DIRNAME))

    def _recorded_config(self):
        from configuration.benchmark                    import BenchmarkConfig
        from pipelines.shared.orchestration.resumed_run import StagedRunLocator
        from tools.runtime.config_cli                   import ConfigCli

        return ConfigCli.load_resolved(BenchmarkConfig(), StagedRunLocator.config_path(self.root))

    def inference_split(self) -> str:
        return self._recorded_config().inference.split

    def comparison_root(self, comparison_dirname: str) -> Path:
        return self.root / self.COMPARISON_DIRNAME

    def comparison_report_names(self, configured: list[str]) -> list[str]:
        return list(self.COMPARISON_REPORTS)

    def stamp_coverage(self, stamp: Path) -> tuple[str, list[str]] | None:
        path = stamp / self.SUMMARY_FILENAME
        if not path.is_file():
            return None

        return self.inference_split(), [entry["name"] for entry in FileIO.load_json(path)]

    def write_comparison(self, comparison_dirname: str, logger) -> Path:
        from pipelines.benchmark.stages import ComparisonStage

        config = self._recorded_config()
        stage  = ComparisonStage(config=config, run_tag=self.root.name, logger=logger, embed_images=config.comparison.embed_images, inference_split=config.inference.split, run_dir=self.root)

        return stage.run()

    def config_files(self, logger) -> list[Path]:
        return self._existing([self.root / relative for relative in self.PIPELINE_FILES], logger)


class ExperimentScanner:

    MAX_DEPTH = 4

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def classify(directory: Path) -> Experiment | None:
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
        root = experiment.root
        if root == self.root or not root.is_relative_to(self.root):
            return root.name

        return root.relative_to(self.root).as_posix()

    def _walk(self, directory: Path, depth: int):
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
        if self.classify(self.root) is None and TrialTree.is_trial(self.root):
            return [Experiment(self.root, self.root.parent, selected=[self.root])]

        return sorted(self._walk(self.root, 0), key=lambda experiment: experiment.root)


class ExperimentSelector(RunSelector):

    def __init__(self, experiments_dir: Path, comparison_dirname: str, logger) -> None:
        super().__init__(experiments_dir, comparison_dirname, logger, action="mail")
        self.scanner     = ExperimentScanner(self.runs_dir)
        self.experiments = {}

    def _discover(self) -> list[Path]:
        found = self.scanner.scan()
        self.experiments = {experiment.root: experiment for experiment in found}

        return list(self.experiments)

    def _present(self, run_dirs: list[Path]) -> None:
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
        return [self.experiments[root] for root in self.resolve(experiment_filter)]


class ReportBundle:

    def __init__(self, staging_dir: Path, embed_images: bool, logger) -> None:
        self.staging_dir  = Path(staging_dir)
        self.embed_images = embed_images
        self.logger       = logger

    def _stage_report(self, report_path: Path, output_name: str) -> Path:
        text        = report_path.read_text(encoding="utf-8")
        rewritten   = ReportImageRewriter(report_path, self.embed_images).rewrite(text)
        destination = self.staging_dir / output_name

        destination.write_text(rewritten, encoding="utf-8")
        self.logger.ok(f"Staged {output_name} from {report_path}")
        return destination

    def _stage_copy(self, path: Path, output_name: str | None = None) -> Path:
        destination = self.staging_dir / (output_name or path.name)

        destination.write_bytes(path.read_bytes())
        self.logger.ok(f"Staged {destination.name} from {path}")
        return destination


class ExperimentReportBundle(ReportBundle):

    def __init__(self, experiment: Experiment, config: ExperimentReportMailConfig, logger) -> None:
        super().__init__(config.staging_dir, config.embed_images, logger)
        self.config     = config
        self.experiment = experiment
        self.freshness  = ReportFreshness(config.inference_dirname, logger)

    def _run_report(self, run_dir: Path, split: str) -> Path:
        inference = run_dir / self.config.inference_dirname
        reports   = [stamp / self.config.report_filename for stamp in TrialTree.inference_stamps(inference, split)]
        reports   = [path for path in reports if path.is_file()]

        return reports[-1]

    def _seed_aggregates(self, trial_dir: Path) -> list[Path]:
        inference = trial_dir / self.config.inference_dirname
        if not inference.is_dir():
            return []

        return sorted(path for path in inference.glob(f"*/{self.config.report_filename}") if path.is_file())

    def _check_reused_aggregate(self, trial_dir: Path, report: Path, split: str) -> None:
        averaged  = self._aggregate_record(report)["seed_runs"]
        seeds     = TrialTree.seed_runs(trial_dir)
        uncovered = [f"{trial_dir.name}/{seed.name}" for seed in seeds if seed.name not in averaged]

        self.freshness.check(f"seed aggregate of {trial_dir.name}", report.parent, seeds, split, uncovered)

    def _aggregate_seeds(self, trial_dir: Path, split: str) -> None:
        from configuration.comparison                              import SeedComparisonConfig
        from pipelines.backbone.inference.analysis.seed_comparison import SeedComparison

        self.logger.info(f"{trial_dir.name}: writing the '{split}' seed-aggregate report over {len(TrialTree.seed_runs(trial_dir))} seed run(s)")

        reports = SeedComparison(SeedComparisonConfig(runs_dir=trial_dir.parent, group_tags=[trial_dir.name], split=split), self.logger).run()

        self.logger.ok(f"{trial_dir.name}: seed-aggregate report written to {reports[0]}")

    def _check_aggregate_split(self, trial_dir: Path, report: Path, split: str) -> None:
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
        from configuration.comparison import SeedComparisonConfig

        return FileIO.load_json(report.parent / SeedComparisonConfig.metrics_filename)

    def _trial_report(self, trial_dir: Path, split: str) -> Path:
        if not TrialTree.seed_runs(trial_dir):
            return self._run_report(trial_dir, split)

        aggregates = self._seed_aggregates(trial_dir)
        if aggregates and not self.config.regenerate_reports:
            self._check_reused_aggregate(trial_dir, aggregates[-1], split)
        else:
            self._aggregate_seeds(trial_dir, split)
            aggregates = self._seed_aggregates(trial_dir)

        self._check_aggregate_split(trial_dir, aggregates[-1], split)

        return aggregates[-1]

    def _check_reused_comparison(self, stamp: Path, split: str) -> None:
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
        if stamp is None:
            return []

        paths = []
        for name in self.experiment.comparison_report_names(self.config.comparison_reports):
            path = stamp / name
            paths.append(path)

        return paths

    def _stage_file(self, path: Path, output_name: str) -> Path:
        if path.suffix == ".md":
            return self._stage_report(path, output_name)

        return self._stage_copy(path, output_name)

    def stage(self) -> List[Path]:
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

    ATTACHMENT_TYPES = {
        ".md"   : ("text", "markdown"),
        ".json" : ("application", "json"),
        ".zip"  : ("application", "zip"),
    }

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger

    def _password(self) -> str:
        value = os.environ.get(self.entry_config.password_env, "")

        return value

    def _message(self, subject: str, body: str, attachments: List[Path]) -> EmailMessage:
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

    NOUN         = "source"
    BUNDLE_LABEL = "Bundle of"

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger


class ExperimentMailTarget(MailTarget):

    NOUN         = "experiment"
    BUNDLE_LABEL = "Report bundle of experiment"

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        super().__init__(entry_config, logger)
        self.selector = ExperimentSelector(entry_config.experiments_dir, entry_config.comparison_dirname, logger)

    def settings(self) -> dict:
        return {
            "Experiments dir"    : str(self.entry_config.experiments_dir),
            "Filter"             : ", ".join(self.entry_config.experiment_filter) or "all",
            "Regenerate reports" : self.entry_config.regenerate_reports,
            "Embed images"       : self.entry_config.embed_images,
        }

    def select(self) -> List[Path]:
        return [experiment.root for experiment in self.selector.select_experiments(self.entry_config.experiment_filter)]

    def label(self, source_dir: Path) -> str:
        return self.selector.experiments[Path(source_dir)].label

    def stage(self, source_dir: Path) -> List[List[Path]]:
        experiment = self.selector.experiments[Path(source_dir)]
        config     = self.entry_config.to_config(experiment.label)

        return [ExperimentReportBundle(experiment, config, self.logger).stage()]


class RunContentMailTarget(MailTarget):

    NOUN         = "run"
    BUNDLE_LABEL = "Content bundle of run"

    def settings(self) -> dict:
        return {
            "Runs dir"       : str(self.entry_config.runs_dir),
            "Filter"         : ", ".join(self.entry_config.run_filter) or "all",
            "Max file MB"    : self.entry_config.run_max_file_mb,
            "Excluded paths" : ", ".join(self.entry_config.run_excluded_paths) or "none",
        }

    def select(self) -> List[Path]:
        return RunContentSelector(self.entry_config.runs_dir, self.logger).resolve(self.entry_config.run_filter)

    def label(self, source_dir: Path) -> str:
        return self.entry_config.run_label(source_dir)

    def stage(self, source_dir: Path) -> List[List[Path]]:
        config = self.entry_config.to_run_config(source_dir)
        return [[archive] for archive in RunContentBundle(config, self.logger).stage()]


class DatasetMailTarget(MailTarget):

    NOUN         = "dataset"
    BUNDLE_LABEL = "Artifact bundle of dataset"

    def settings(self) -> dict:
        return {
            "Datasets dir" : str(self.entry_config.datasets_dir),
            "Filter"       : ", ".join(self.entry_config.dataset_filter) or "all",
        }

    def select(self) -> List[Path]:
        return DatasetSelector(self.entry_config.datasets_dir, self.logger).resolve(self.entry_config.dataset_filter)

    def label(self, source_dir: Path) -> str:
        return self.entry_config.dataset_label(source_dir)

    def stage(self, source_dir: Path) -> List[List[Path]]:
        config = self.entry_config.to_dataset_config(source_dir)
        return [[archive] for archive in ArtifactBundle(config, self.logger).stage()]


class ParamRunMailTarget(MailTarget):

    NOUN         = "parameter run"
    BUNDLE_LABEL = "Artifact bundle of parameter run"

    def settings(self) -> dict:
        return {
            "Params dir" : str(self.entry_config.params_dir),
            "Filter"     : ", ".join(self.entry_config.param_run_filter) or "all",
        }

    def select(self) -> List[Path]:
        return ParamRunSelector(self.entry_config.params_dir, self.logger).resolve(self.entry_config.param_run_filter)

    def label(self, source_dir: Path) -> str:
        return self.entry_config.param_run_label(source_dir)

    def stage(self, source_dir: Path) -> List[List[Path]]:
        config = self.entry_config.to_param_config(source_dir)
        return [[archive] for archive in ArtifactBundle(config, self.logger).stage()]


class ReportMailBatch:

    TARGETS = {
        "experiments"      : ExperimentMailTarget,
        "runs"             : RunContentMailTarget,
        "datasets"         : DatasetMailTarget,
        "param_extraction" : ParamRunMailTarget,
    }

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger
        self.mailer       = ReportMailer(entry_config, logger)
        self.target       = self.TARGETS[entry_config.target](entry_config, logger)

    def _log_settings(self) -> None:
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

    def _subject(self, label: str, index: int, total: int) -> str:
        if total == 1:
            return f"{self.entry_config.subject_prefix}: {label}"

        return f"{self.entry_config.subject_prefix}: {label} [{index}/{total}]"

    def _body(self, source_dir: Path, label: str, attachments: List[Path], index: int, total: int) -> str:
        lines  = [f"{self.target.BUNDLE_LABEL} '{label}'.", ""]
        lines += [f"Source directory: {source_dir}", ""]

        if total > 1:
            lines += [f"Part {index} of {total}; unpack every part into one directory to rebuild the run tree.", ""]

        lines += ["Attached files:"]
        lines += [f"  - {path.name} ({path.stat().st_size / (1024 * 1024):,.1f} MB)" for path in attachments]

        return "\n".join(lines) + "\n"

    def _mail_source(self, source_dir: Path) -> dict:
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

    def _write_record(self, results: list[dict], started_at: str) -> Path:
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
