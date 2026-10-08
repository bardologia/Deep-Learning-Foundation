"""Entry point packing inference figures of selected runs into the publication figure set.

Copies existing figures under stable ``<run>__<figure-path>.png`` names and writes a JSON
manifest of their sources; paper styling itself comes from the inference run.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the figure-pack config and copies the selected figures with a manifest."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                           import PaperFigurePackConfig
    from pipelines.backbone.inference.analysis.paper_figures import PaperFigurePack
    from tools.runtime.config_cli                            import ConfigCli
    from tools.monitoring.logger                             import Logger

    cli    = ConfigCli(PaperFigurePackConfig(), description="Pack inference figures from selected runs into the publication figures directory under stable names (<run>__<figure-path>.png) with a JSON manifest of every source; render the figures in paper style first by running inference with figure_style=paper")
    config = cli.apply()

    with Logger(log_dir="logs", name="export_paper_figures") as logger:
        PaperFigurePack(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
