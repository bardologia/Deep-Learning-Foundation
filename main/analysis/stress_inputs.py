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

    config = ConfigCli(RobustnessConfig(), description="Stress trained backbone runs with controlled input degradation: curve-MSE-vs-severity under gaussian noise on the normalized inputs and under whole-track dropout (secondary + interferogram channels zeroed, averaged over random subsets); curves, JSON and a markdown report land inside each run directory").apply()

    logger = Logger(log_dir="logs", name="stress_inputs")
    RobustnessBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
