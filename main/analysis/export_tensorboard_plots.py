from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics           import TensorboardExportEntryConfig
    from tools.diagnostics.tensorboard_export import TensorboardExportBatch
    from tools.runtime.config_cli             import ConfigCli
    from tools.monitoring.logger              import Logger

    entry  = ConfigCli(TensorboardExportEntryConfig(), description="Scan a runs directory for training runs with tensorboard event logs, select one or more, and export every scalar series as a publication-quality figure inside each run directory; train and validation series of the same metric share one figure, and sibling seed runs of one trial additionally get per-metric seed-overlay figures in the trial directory").apply()

    logger = Logger(log_dir="logs", name="export_tensorboard_plots")
    TensorboardExportBatch(entry, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
