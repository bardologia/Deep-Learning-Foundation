from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics                               import AttentionCaptureConfig
    from pipelines.backbone.inference.analysis.attention_capture import AttentionCaptureBatch
    from tools.runtime.config_cli                                import ConfigCli
    from tools.monitoring.logger                                 import Logger

    config = ConfigCli(AttentionCaptureConfig(), description="Capture the attention of trained attention-based backbone runs on one real batch: attention-gate maps, shared-block and torch multi-head attention weights with per-layer entropy and peak statistics, gate figures, JSON and a markdown report inside each run directory; refuses runs whose model holds no attention module").apply()

    logger = Logger(log_dir="logs", name="capture_attention")
    AttentionCaptureBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
