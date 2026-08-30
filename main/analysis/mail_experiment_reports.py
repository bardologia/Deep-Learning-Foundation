"""Entry point mailing each experiment's gathered reports to the configured recipient.

Thin CLI wrapper: parses a ``ReportMailEntryConfig`` and hands it to
``ExperimentReportMailBatch``, which scans the experiments directory, gathers
each experiment's trial and comparison reports plus its run configuration, and
sends one email per experiment with the files attached.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the report-mailing config and runs the mailing batch."""
    EnvironmentPinner.threads()

    from configuration.diagnostics     import ReportMailEntryConfig
    from tools.reporting.report_mailer import ExperimentReportMailBatch
    from tools.runtime.config_cli      import ConfigCli
    from tools.monitoring.logger       import Logger

    entry  = ConfigCli(ReportMailEntryConfig(), description="Scan an experiments directory as written by a sequence run, select one or more experiments, and email each experiment's trial seed-aggregate reports, its newest trial-comparison reports and the configuration it ran with as attachments, one message per experiment; the SMTP password is read from an environment variable").apply()

    logger = Logger(log_dir="logs", name="mail_experiment_reports")
    ExperimentReportMailBatch(entry, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
