"""Entry point exporting trained backbone checkpoints as ONNX graphs.

Thin CLI wrapper: parses an ``OnnxExportEntryConfig`` and hands it to
``OnnxRunExporter``, which rebuilds each run's model from its persisted
configuration, loads the best checkpoint and writes model.onnx plus a
manifest under the run's export directory.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the export config and writes one ONNX graph per selected run."""
    EnvironmentPinner.threads()

    from configuration.diagnostics                  import OnnxExportEntryConfig
    from pipelines.backbone.inference.onnx_export   import OnnxRunExporter
    from tools.runtime.config_cli                   import ConfigCli
    from tools.monitoring.logger                    import Logger

    entry  = ConfigCli(OnnxExportEntryConfig(), description="Rebuild each selected backbone run's model from its persisted architecture config, load the best checkpoint and export the forward as an ONNX graph with a dynamic batch axis, writing model.onnx and a manifest (opset, input shape, checkpoint provenance, sha256) under the run's export directory").apply()

    logger = Logger(log_dir="logs", name="export_onnx")
    OnnxRunExporter(entry, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
