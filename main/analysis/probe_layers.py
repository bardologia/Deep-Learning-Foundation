"""Entry point fitting linear layer probes on the features of trained backbone runs.

Ridge readouts on sampled real pixels predict the ground-truth active Gaussian count and
the dominant component z, giving held-out R-squared as a function of depth.
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

    cli    = ConfigCli(LayerProbeConfig(), description="Probe trained backbone runs layer by layer: ridge readouts on sampled real pixels predict the GT active Gaussian count and dominant component position along z from each layer's features, and the held-out R² by depth shows where each quantity becomes linearly decodable; figures, JSON and a markdown report land inside each run directory")
    config = cli.apply()

    with Logger(log_dir="logs", name="probe_layers") as logger:
        LayerProbeBatch(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
