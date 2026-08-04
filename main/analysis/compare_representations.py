from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                                import CkaConfig
    from pipelines.backbone.inference.analysis.representation_cka import CkaComparison
    from tools.runtime.config_cli                                 import ConfigCli
    from tools.monitoring.logger                                  import Logger

    config = ConfigCli(CkaConfig(), description="Compare the internal representations of two or more trained backbone runs with linear CKA on identical sampled pixels: per-pair cross-layer heatmaps plus a run-by-run alignment matrix showing whether different architectures or seeds converge to similar features; requires runs sharing the split region and patch grid").apply()

    logger = Logger(log_dir="logs", name="compare_representations")
    CkaComparison(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
