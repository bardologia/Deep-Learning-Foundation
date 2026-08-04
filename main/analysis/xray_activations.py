from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                             import ActivationXrayConfig
    from pipelines.backbone.inference.analysis.activation_xray import ActivationXrayBatch
    from tools.runtime.config_cli                              import ConfigCli
    from tools.monitoring.logger                               import Logger

    config = ConfigCli(ActivationXrayConfig(), description="X-ray the activations of trained backbone runs on real data: hook every leaf module, run a few batches, and report dead layers, dead channels, exploding or constant activations with depth profiles, per-layer histograms, a JSON dump and a markdown report inside each run directory").apply()

    logger = Logger(log_dir="logs", name="xray_activations")
    ActivationXrayBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
