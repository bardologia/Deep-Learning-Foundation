
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                               import InputAttributionConfig
    from pipelines.backbone.inference.analysis.input_attribution import InputAttributionBatch
    from tools.runtime.config_cli                                import ConfigCli
    from tools.monitoring.logger                                 import Logger

    cli    = ConfigCli(InputAttributionConfig(), description="Attribute the predictions of trained backbone runs to their input channels: gradient attribution per target channel on probe windows, plus an occlusion pass that zeroes one normalized channel at a time and measures the prediction shift; figures, JSON and a markdown report are written into each run directory")
    config = cli.apply()

    with Logger(log_dir="logs", name="attribute_inputs") as logger:
        InputAttributionBatch(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
