from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                          import LayerProbeConfig
    from pipelines.backbone.inference.analysis.layer_probes import LayerProbeBatch
    from tools.runtime.config_cli                           import ConfigCli
    from tools.monitoring.logger                            import Logger

    config = ConfigCli(LayerProbeConfig(), description="Probe trained backbone runs layer by layer: ridge readouts on sampled real pixels predict the GT active Gaussian count and dominant scatterer elevation from each layer's features, and the held-out R² by depth shows where each quantity becomes linearly decodable; figures, JSON and a markdown report land inside each run directory").apply()

    logger = Logger(log_dir="logs", name="probe_layers")
    LayerProbeBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
