"""Entry point exporting scalar run metrics as CSV, LaTeX and markdown tables.

Thin CLI wrapper: parses a ``ResultTableEntryConfig`` and hands it to
``ResultTableExporter``, which does the run discovery, metric loading and
table rendering.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Parses the table-export config and writes the result tables."""
    EnvironmentPinner.threads()

    from configuration.diagnostics          import ResultTableEntryConfig
    from pipelines.comparison.result_tables import ResultTableExporter
    from tools.runtime.config_cli           import ConfigCli
    from tools.monitoring.logger            import Logger

    entry  = ConfigCli(ResultTableEntryConfig(), description="Scan a runs directory for finished inference stamps and export one table row per run: a CSV with every scalar metric at full precision, plus LaTeX (booktabs) and markdown tables with the selected metric columns at fixed precision, ready for a paper").apply()

    with Logger(log_dir="logs", name="export_result_tables") as logger:
        ResultTableExporter(entry, logger).run()


if __name__ == "__main__":
    main()
