"""Entry point fitting linear layer probes on the features of trained backbone runs.

Ridge readouts on sampled real pixels predict the ground-truth active Gaussian count and
the dominant scatterer elevation, giving held-out R-squared as a function of depth.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the layer-probe config and runs the probes over the selected runs."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                          import LayerProbeConfig
    from pipelines.backbone.inference.analysis.layer_probes import LayerProbeBatch
    from tools.runtime.config_cli                           import ConfigCli
    from tools.monitoring.logger                            import Logger

    config = ConfigCli(LayerProbeConfig(), description="Probe trained backbone runs layer by layer: ridge readouts on sampled real pixels predict the GT active Gaussian count and dominant component position along z from each layer's features, and the held-out R² by depth shows where each quantity becomes linearly decodable; figures, JSON and a markdown report land inside each run directory").apply()

    logger = Logger(log_dir="logs", name="probe_layers")
    LayerProbeBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
