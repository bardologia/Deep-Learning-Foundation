"""Entry point mailing gathered results to the configured recipient.

Thin CLI wrapper: parses a ``ReportMailEntryConfig`` and hands it to
``ReportMailBatch``, which picks the configured target, scans its root
directory, and sends one message per selected experiment, training run,
dataset or parameter-extraction run with that source's files attached,
writing an experiment's missing seed aggregates and comparisons first.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the report-mailing config and runs the mailing batch."""
    EnvironmentPinner.threads()

    from configuration.diagnostics     import ReportMailEntryConfig
    from tools.reporting.report_mailer import ReportMailBatch
    from tools.runtime.config_cli      import ConfigCli
    from tools.monitoring.logger       import Logger

    entry  = ConfigCli(ReportMailEntryConfig(), description="Email gathered results, one message per source: experiments found at or below any path (a sweep of trial runs, a cross-validation run, a benchmark run, a sequence workspace or a runs tree), each mailed with the seed-aggregate report of every trial, the trial-comparison reports, a cross-validation run's cross-fold report and the configuration it ran with, the missing seed aggregates and comparisons written first; training runs, each one mailed whole except for the checkpoints, the reconstructed cubes and the files above the per-file cap; generated datasets (the layout and metadata generate_dataset wrote), or parameter-extraction runs (the extraction outputs plus the analyze_param_extraction metrics and figures), the last three packed into size-capped zip volumes; the SMTP password is read from an environment variable").apply()

    with Logger(log_dir="logs", name="mail_experiment_reports") as logger:
        ReportMailBatch(entry, logger).run()


if __name__ == "__main__":
    main()
