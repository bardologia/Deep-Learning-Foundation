from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.comparison import SeedComparisonConfig
    from pipelines.backbone.inference.analysis.seed_comparison import SeedComparison
    from tools.runtime.config_cli import ConfigCli

    config = ConfigCli(SeedComparisonConfig(), description="Aggregate the existing inference results of the seed runs nested inside each selected group directory into a per-group seed-comparison report with the across-seed mean ± std of every scalar metric, plus per-pixel seed-disagreement maps written into each seed's cube directory; no inference is re-run").apply()

    SeedComparison(config).run()


if __name__ == "__main__":
    main()
