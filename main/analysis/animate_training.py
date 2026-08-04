from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                                import TrainingEvolutionConfig
    from pipelines.backbone.inference.analysis.training_evolution import TrainingEvolutionBatch
    from tools.runtime.config_cli                                 import ConfigCli
    from tools.monitoring.logger                                  import Logger

    config = ConfigCli(TrainingEvolutionConfig(), description="Animate how a backbone run learned: re-predict probe pixels from every epoch snapshot saved during training (snapshot_every_n_epochs > 0) and render one GIF per pixel of the prediction converging onto the fixed ground truth; GIFs, JSON and a markdown report land inside each run directory").apply()

    logger = Logger(log_dir="logs", name="animate_training")
    TrainingEvolutionBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
