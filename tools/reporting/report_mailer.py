"""Mailing of experiment report bundles gathered from a sequence-run workspace.

Discovers the experiment folders inside an experiments directory, gathers each
experiment's trial seed-aggregate reports, its newest trial-comparison reports
and the configuration files the experiment ran with, and sends one email per
experiment with the gathered files attached.
"""

from __future__ import annotations

import os
import re
import smtplib

from email.message import EmailMessage
from pathlib       import Path
from typing        import List

from configuration.diagnostics        import ExperimentReportMailConfig, ReportCollectionConfig, ReportMailEntryConfig
from tools.data.io                    import FileIO
from tools.reporting.report_collector import ReportImageRewriter, RunReportLocator
from tools.runtime.run_selector       import RunSelector


class ExperimentSelector(RunSelector):
    """Selects experiment directories by the comparison folder they contain."""

    def __init__(self, experiments_dir: Path, comparison_dirname: str, logger) -> None:
        """Initializes the selector for report mailing.

        Args:
            experiments_dir    : Root directory holding one folder per experiment.
            comparison_dirname : Directory inside an experiment marking finished trial comparisons.
            logger             : Logger for the experiment table and selection messages.
        """
        super().__init__(experiments_dir, comparison_dirname, logger, action="mail")

    def _discover(self) -> list[Path]:
        """Returns the sorted direct children holding a comparison directory.

        Raises:
            FileNotFoundError : If the experiments root is missing or no child holds a comparison directory.
        """
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Experiments directory does not exist: {self.runs_dir}")

        experiments = sorted(child for child in self.runs_dir.iterdir() if child.is_dir() and (child / self.marker).is_dir())

        if not experiments:
            raise FileNotFoundError(f"No experiment with a '{self.marker}' directory found under {self.runs_dir}")

        return experiments

    def _present(self, run_dirs: list[Path]) -> None:
        """Logs a numbered table of the experiments with trial and comparison counts."""
        rows = []
        for index, experiment_dir in enumerate(run_dirs, start=1):
            trials = [child for child in experiment_dir.iterdir() if child.is_dir() and child.name != self.marker and not child.name.startswith("_")]
            stamps = [child for child in (experiment_dir / self.marker).iterdir() if child.is_dir()]
            rows.append({
                "#"           : index,
                "Experiment"  : str(experiment_dir.relative_to(self.runs_dir)),
                "Trials"      : str(len(trials)),
                "Comparisons" : str(len(stamps)),
            })

        self.logger.metrics_table(rows, columns=["#", "Experiment", "Trials", "Comparisons"], title=f"Experiments under {self.runs_dir}")


class ExperimentReportBundle:
    """Gathers one experiment's reports and configuration into a staging directory.

    Attributes:
        config       : Gathering config naming the experiment, the staging directory and the filenames.
        logger       : Logger for per-file progress.
        SEED_PATTERN : Directory name pattern identifying a seed run inside a trial.
    """

    SEED_PATTERN = re.compile(r"seed\d+")

    def __init__(self, config: ExperimentReportMailConfig, logger) -> None:
        """Initializes the bundle with its gathering config and logger."""
        self.config = config
        self.logger = logger

        self.experiment_dir = Path(config.experiment_dir)
        self.staging_dir    = Path(config.staging_dir)

    def _trial_dirs(self) -> List[Path]:
        """Returns the experiment's trial directories, sorted by name.

        Returns:
            Direct children holding either seed runs or their own inference outputs.

        Raises:
            FileNotFoundError : If no child of the experiment looks like a trial.
        """
        skip     = {self.config.comparison_dirname, "logs", "batch_train_logs"}
        children = sorted(child for child in self.experiment_dir.iterdir() if child.is_dir() and child.name not in skip and not child.name.startswith("_"))
        trials   = [child for child in children if self._seed_dirs(child) or (child / self.config.inference_dirname).is_dir()]

        if not trials:
            raise FileNotFoundError(f"No trial directory with seed runs or '{self.config.inference_dirname}' outputs under {self.experiment_dir}")

        return trials

    def _trial_report(self, trial_dir: Path) -> Path:
        """Returns the newest trial-level report, the seed aggregate for seeded trials.

        Args:
            trial_dir : Trial directory whose report is located.

        Returns:
            Path of the newest report under the trial's own inference directory.

        Raises:
            FileNotFoundError : If a seeded trial carries no aggregated report, pointing at compare_seeds.
        """
        collection = ReportCollectionConfig(
            run_directory     = trial_dir,
            collector_dir     = self.staging_dir,
            inference_dirname = self.config.inference_dirname,
            report_filename   = self.config.report_filename,
            latest_only       = True,
            embed_images      = self.config.embed_images,
        )

        try:
            return RunReportLocator(collection).locate()[-1]
        except FileNotFoundError as error:
            if self._seed_dirs(trial_dir):
                raise FileNotFoundError(f"Trial {trial_dir} holds seed runs but no aggregated report under its own '{self.config.inference_dirname}' directory; run compare_seeds on it first") from error
            raise

    def _comparison_reports(self) -> List[Path]:
        """Returns the configured reports of the newest comparison stamp.

        Raises:
            FileNotFoundError : If the comparison directory, a stamp, or a configured report is missing.
        """
        root = self.experiment_dir / self.config.comparison_dirname

        if not root.is_dir():
            raise FileNotFoundError(f"No '{self.config.comparison_dirname}' directory under {self.experiment_dir}; run compare_trials on the experiment first")

        stamps = sorted(child for child in root.iterdir() if child.is_dir())
        if not stamps:
            raise FileNotFoundError(f"No comparison stamp directory under {root}; run compare_trials on the experiment first")

        latest = stamps[-1]

        paths = []
        for name in self.config.comparison_reports:
            path = latest / name
            if not path.is_file():
                raise FileNotFoundError(f"Comparison report '{name}' is missing from {latest}")
            paths.append(path)

        return paths

    def _config_files(self, trial_dirs: List[Path]) -> List[Path]:
        """Returns the configuration files describing what the experiment ran.

        Args:
            trial_dirs : Trial directories of the experiment, first one providing the representative run.

        Returns:
            The experiment-level scheduler summary and one run's resolved entry configuration, each when present.
        """
        paths = []

        scheduler = self.experiment_dir / self.config.scheduler_results_filename
        if scheduler.is_file():
            paths.append(scheduler)
        else:
            self.logger.warning(f"No '{self.config.scheduler_results_filename}' at {self.experiment_dir}; the experiment-level configuration is not attached")

        seeds    = self._seed_dirs(trial_dirs[0])
        run_dir  = seeds[0] if seeds else trial_dirs[0]
        resolved = run_dir / self.config.resolved_config_relpath

        if resolved.is_file():
            paths.append(resolved)
        else:
            self.logger.warning(f"No '{self.config.resolved_config_relpath}' under {run_dir}; the resolved run configuration is not attached")

        return paths

    def _seed_dirs(self, trial_dir: Path) -> List[Path]:
        """Returns the seed run directories nested inside a trial, sorted by name."""
        return sorted(child for child in trial_dir.iterdir() if child.is_dir() and self.SEED_PATTERN.fullmatch(child.name))

    def _stage_report(self, report_path: Path, output_name: str) -> Path:
        """Rewrites one report's image links and writes it into the staging directory.

        Args:
            report_path : Source report inside the experiment's outputs.
            output_name : Filename of the staged copy.

        Returns:
            Path of the staged copy.
        """
        text        = report_path.read_text(encoding="utf-8")
        rewritten   = ReportImageRewriter(report_path, self.config.embed_images).rewrite(text)
        destination = self.staging_dir / output_name

        destination.write_text(rewritten, encoding="utf-8")
        self.logger.ok(f"Staged {output_name}")
        return destination

    def _stage_copy(self, path: Path) -> Path:
        """Copies one file verbatim into the staging directory and returns the copy."""
        destination = self.staging_dir / path.name

        destination.write_bytes(path.read_bytes())
        self.logger.ok(f"Staged {path.name}")
        return destination

    def stage(self) -> List[Path]:
        """Gathers every report and configuration file of the experiment.

        Returns:
            The staged file paths: one report per trial, the comparison reports, then the configuration files.
        """
        self.logger.subsection(f"Gathering experiment: {self.experiment_dir.name}")
        FileIO.ensure_dirs(self.staging_dir)

        trial_dirs = self._trial_dirs()

        staged  = [self._stage_report(self._trial_report(trial_dir), f"{trial_dir.name}.md") for trial_dir in trial_dirs]
        staged += [self._stage_report(path, f"comparison_{path.name}") for path in self._comparison_reports()]
        staged += [self._stage_copy(path) for path in self._config_files(trial_dirs)]

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


class ExperimentReportMailBatch:
    """Mails the report bundle of every selected experiment, one message each.

    Attributes:
        entry_config : Entry config naming the experiments directory, the filter and the mail settings.
        logger       : Logger for selection and summary messages.
        mailer       : Mailer submitting one message per experiment.
    """

    def __init__(self, entry_config: ReportMailEntryConfig, logger) -> None:
        """Initializes the batch with its entry config and logger."""
        self.entry_config = entry_config
        self.logger       = logger
        self.mailer       = ReportMailer(entry_config, logger)

    def _select_experiments(self) -> list[Path]:
        """Returns the experiment directories chosen by the configured filter."""
        selector = ExperimentSelector(self.entry_config.experiments_dir, self.entry_config.comparison_dirname, self.logger)
        return selector.resolve(self.entry_config.experiment_filter)

    def _body(self, experiment_dir: Path, attachments: List[Path]) -> str:
        """Returns the plain-text body listing the experiment and its attached files.

        Args:
            experiment_dir : Experiment the message reports on.
            attachments    : Files attached to the message.

        Returns:
            The message body.
        """
        lines  = [f"Report bundle of experiment '{self.entry_config.experiment_label(experiment_dir)}'.", ""]
        lines += [f"Experiment directory: {experiment_dir}", ""]
        lines += ["Attached files:"]
        lines += [f"  - {path.name}" for path in attachments]

        return "\n".join(lines) + "\n"

    def _mail_experiment(self, experiment_dir: Path) -> dict:
        """Gathers and mails one experiment's bundle.

        Args:
            experiment_dir : Experiment whose reports are gathered and sent.

        Returns:
            Mapping with the experiment directory, the staged paths and the subject used.
        """
        config  = self.entry_config.to_config(experiment_dir)
        staged  = ExperimentReportBundle(config, self.logger).stage()
        subject = f"{self.entry_config.subject_prefix}: {self.entry_config.experiment_label(experiment_dir)}"

        self.mailer.send(subject, self._body(experiment_dir, staged), staged)

        return {
            "experiment_dir" : str(experiment_dir),
            "staging_dir"    : str(config.staging_dir),
            "n_attachments"  : len(staged),
            "staged_paths"   : staged,
            "subject"        : subject,
        }

    def run(self) -> list[dict]:
        """Selects the experiments and gathers and mails each one.

        Returns:
            One mailing summary per experiment, in selection order.
        """
        self.logger.section(f"Report mailing to {self.entry_config.recipient}")

        experiment_dirs = self._select_experiments()
        results         = [self._mail_experiment(experiment_dir) for experiment_dir in experiment_dirs]

        self.logger.ok(f"Mailed {sum(result['n_attachments'] for result in results)} file(s) across {len(results)} experiment message(s)")
        return results
