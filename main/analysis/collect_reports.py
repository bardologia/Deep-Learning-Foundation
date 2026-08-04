from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics        import ReportCollectionEntryConfig
    from tools.reporting.report_collector import ReportCollectionBatch
    from tools.runtime.config_cli         import ConfigCli
    from tools.monitoring.logger          import Logger

    entry  = ConfigCli(ReportCollectionEntryConfig(), description="Scan a runs directory for training runs with inference reports, select one or more, and gather each run's report into a single collector directory renamed after the run; image links are re-pointed at the original figures or embedded as self-contained data").apply()

    logger = Logger(log_dir="logs", name="collect_reports")
    ReportCollectionBatch(entry, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
