
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                        import RobustnessConfig
    from pipelines.backbone.inference.analysis.robustness import RobustnessBatch
    from tools.runtime.config_cli                         import ConfigCli
    from tools.monitoring.logger                          import Logger

    cli    = ConfigCli(RobustnessConfig(), description="Stress trained backbone runs with controlled input degradation: weighted MSE against severity under Gaussian noise (all channels and each input channel), global gain error, input-channel dropout (averaged over random subsets) and row shift-equivariance error, with the noise-tolerance sigma at which the loss doubles; curves, JSON and a markdown report land inside each run directory")
    config = cli.apply()

    with Logger(log_dir="logs", name="stress_inputs") as logger:
        RobustnessBatch(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
