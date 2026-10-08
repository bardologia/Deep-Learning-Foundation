"""Entry point x-raying the activations of trained backbone runs on real data.

Hooks every leaf module over a few batches and reports dead layers, dead channels and
exploding or constant activations as depth profiles, histograms, JSON and markdown.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the activation-xray config and analyses the activations of the selected runs."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                             import ActivationXrayConfig
    from pipelines.backbone.inference.analysis.activation_xray import ActivationXrayBatch
    from tools.runtime.config_cli                              import ConfigCli
    from tools.monitoring.logger                               import Logger

    cli    = ConfigCli(ActivationXrayConfig(), description="X-ray the activations of trained backbone runs on real data: hook every leaf module, run a few batches, and report dead layers, dead channels, exploding or constant activations with depth profiles, per-layer histograms, a JSON dump and a markdown report inside each run directory")
    config = cli.apply()

    with Logger(log_dir="logs", name="xray_activations") as logger:
        ActivationXrayBatch(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
