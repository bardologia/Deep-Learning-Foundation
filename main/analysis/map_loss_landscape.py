from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                            import LossLandscapeConfig
    from pipelines.backbone.inference.analysis.loss_landscape import LossLandscapeBatch
    from tools.runtime.config_cli                             import ConfigCli
    from tools.monitoring.logger                              import Logger

    config = ConfigCli(LossLandscapeConfig(), description="Map the curve-MSE landscape around the trained weights of backbone runs along two filter-normalized random directions: 1D cuts, a 2D log contour and a sharpness scalar per direction, written into each run directory; the physical curve objective keeps runs with different training losses comparable").apply()

    logger = Logger(log_dir="logs", name="map_loss_landscape")
    LossLandscapeBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
