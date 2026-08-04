from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                           import PaperFigurePackConfig
    from pipelines.backbone.inference.analysis.paper_figures import PaperFigurePack
    from tools.runtime.config_cli                            import ConfigCli
    from tools.monitoring.logger                             import Logger

    config = ConfigCli(PaperFigurePackConfig(), description="Pack inference figures from selected runs into the publication figures directory under stable names (<run>__<figure-path>.png) with a JSON manifest of every source; render the figures in paper style first by running inference with figure_style=paper").apply()

    logger = Logger(log_dir="logs", name="export_paper_figures")
    PaperFigurePack(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
