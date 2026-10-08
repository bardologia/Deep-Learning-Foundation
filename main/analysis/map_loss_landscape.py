"""Entry point mapping the curve-MSE loss landscape around trained backbone weights.

Perturbs the weights along two filter-normalized random directions and writes 1D cuts, a
2D log contour and a per-direction sharpness scalar into each run directory.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the landscape config and maps the loss surface of the selected runs."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                            import LossLandscapeConfig
    from pipelines.backbone.inference.analysis.loss_landscape import LossLandscapeBatch
    from tools.runtime.config_cli                             import ConfigCli
    from tools.monitoring.logger                              import Logger

    cli    = ConfigCli(LossLandscapeConfig(), description="Map the curve-MSE landscape around the trained weights of backbone runs along two filter-normalized random directions: 1D cuts, a 2D log contour and a sharpness scalar per direction, written into each run directory; the physical curve objective keeps runs with different training losses comparable")
    config = cli.apply()

    with Logger(log_dir="logs", name="map_loss_landscape") as logger:
        LossLandscapeBatch(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
