
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics     import ReportMailEntryConfig
    from tools.reporting.report_mailer import ReportMailBatch
    from tools.runtime.config_cli      import ConfigCli
    from tools.monitoring.logger       import Logger

    entry  = ConfigCli(ReportMailEntryConfig(), description="Email gathered results, one message per source: experiments found at or below any path (a sweep of trial runs, a cross-validation run, a benchmark run, a sequence workspace or a runs tree), each mailed with the seed-aggregate report of every trial, the trial-comparison reports, a cross-validation run's cross-fold report and the configuration it ran with, the missing seed aggregates and comparisons written first; training runs, each one mailed whole except for the checkpoints, the saved prediction maps and the files above the per-file cap; or generated datasets (the dataset.json and logs generate_dataset wrote), the last two packed into size-capped zip volumes; the SMTP password is read from an environment variable").apply()

    with Logger(log_dir="logs", name="mail_experiment_reports") as logger:
        ReportMailBatch(entry, logger).run()


if __name__ == "__main__":
    main()
