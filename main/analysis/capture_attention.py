"""Entry point capturing attention maps and statistics from attention-based backbone runs."""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the CLI overrides and runs the attention capture batch over the given runs."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                               import AttentionCaptureConfig
    from pipelines.backbone.inference.analysis.attention_capture import AttentionCaptureBatch
    from tools.runtime.config_cli                                import ConfigCli
    from tools.monitoring.logger                                 import Logger

    config = ConfigCli(AttentionCaptureConfig(), description="Capture the attention of trained attention-based backbone runs on real batches: attention-gate maps, shared-block and torch multi-head attention weights with per-layer and per-head entropy, head redundancy, attention distance against the uniform reference, gate and mean-attention figures, JSON and a markdown report inside each run directory; refuses runs whose model holds no attention module").apply()

    logger = Logger(log_dir="logs", name="capture_attention")
    AttentionCaptureBatch(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
