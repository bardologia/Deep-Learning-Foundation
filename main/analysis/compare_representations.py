"""Entry point comparing internal representations of trained backbone runs with linear CKA.

Thin CLI wrapper around ``CkaComparison``, which samples identical pixels across the
selected runs and writes cross-layer heatmaps plus a run-by-run alignment matrix.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the CKA config and runs the cross-run representation comparison."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                                import CkaConfig
    from pipelines.backbone.inference.analysis.representation_cka import CkaComparison
    from tools.runtime.config_cli                                 import ConfigCli
    from tools.monitoring.logger                                  import Logger

    cli    = ConfigCli(CkaConfig(), description="Compare the internal representations of two or more trained backbone runs with linear CKA on identical sampled pixels: per-pair cross-layer heatmaps plus a run-by-run alignment matrix showing whether different architectures or seeds converge to similar features; requires runs sharing the split region and patch grid")
    config = cli.apply()

    with Logger(log_dir="logs", name="compare_representations") as logger:
        CkaComparison(config, logger, cli.profile_overrides).run()


if __name__ == "__main__":
    main()
