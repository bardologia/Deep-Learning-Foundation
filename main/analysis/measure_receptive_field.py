from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                             import ReceptiveFieldConfig
    from pipelines.backbone.inference.analysis.receptive_field import ReceptiveFieldBatch
    from tools.runtime.config_cli                              import ConfigCli
    from tools.monitoring.logger                               import Logger

    config = ConfigCli(ReceptiveFieldConfig(), description="Measure the effective receptive field of trained backbone runs on real data: probe pixels across the split region, gradient of the centre output w.r.t. the input window, ERF sigma per axis, cumulative gradient-mass ladder, heatmap and report written into each run directory").apply()

    logger = Logger(log_dir="logs", name="measure_receptive_field")
    ReceptiveFieldBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
