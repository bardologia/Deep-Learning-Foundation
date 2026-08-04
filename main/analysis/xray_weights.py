from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics import WeightXrayEntryConfig
    from tools.diagnostics.weight_xray               import WeightXrayBatch
    from tools.runtime.config_cli                    import ConfigCli
    from tools.monitoring.logger                     import Logger

    entry  = ConfigCli(WeightXrayEntryConfig(), description="Scan a runs directory for model checkpoints, select one or more, and x-ray each for dead weights, uniform layers, rank collapse, and other structural pathologies; results are written inside each run directory").apply()

    logger = Logger(log_dir="logs", name="xray_weights")
    WeightXrayBatch(entry, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
