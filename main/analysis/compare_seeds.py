"""Entry point aggregating existing per-seed inference results into seed-comparison reports.

Reads only what previous inference runs left on disk: no model is re-evaluated.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the seed-comparison config and writes the per-group seed reports."""
    EnvironmentPinner.threads()

    from configuration.comparison                              import SeedComparisonConfig
    from pipelines.backbone.inference.analysis.seed_comparison import SeedComparison
    from tools.monitoring.logger                               import Logger
    from tools.runtime.config_cli                              import ConfigCli

    config = ConfigCli(SeedComparisonConfig(), description="Aggregate the existing inference results of the seed runs nested inside each selected group directory into a per-group seed-comparison report with the across-seed mean ± std of every scalar metric, plus per-pixel seed-disagreement maps written into each seed's cube directory; no inference is re-run").apply()

    with Logger(log_dir="logs", name="compare_seeds") as logger:
        SeedComparison(config, logger).run()


if __name__ == "__main__":
    main()
