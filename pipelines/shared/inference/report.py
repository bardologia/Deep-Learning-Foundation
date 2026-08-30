"""Shared skeleton of the per-run markdown inference report."""

from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

from tools.reporting.markdown  import MarkdownDoc, ScalarFormatter
from tools.reporting.reporting import ReportAssets


class InferenceReportBase:
    """Assembles a run summary, metric tables and figure sections into one report.

    Subclasses declare which metrics and figures belong to which section and
    supply the run summary rows.

    Attributes:
        METRIC_GROUPS   : (section title, metric keys) pairs rendered as tables.
        FIGURE_SECTIONS : (section title, figure group keys) pairs rendered as images.
        DOC_TITLE       : Title of the generated document.
        output_dir      : Stamp directory image links are made relative to.
        run             : Run metadata object described in the summary table.
        config          : Inference configuration the report describes.
        metrics         : Scalar metrics keyed by metric name.
        figures         : Figure paths grouped by figure key.
        report_path     : Destination of the rendered markdown file.
        assets          : Path resolver for relative image links.
    """

    METRIC_GROUPS   : list = []
    FIGURE_SECTIONS : list = []
    DOC_TITLE       : str  = "Inference Report"

    def __init__(self, output_dir: Path, run, config, metrics: dict, figures: Dict[str, List[Path]], report_path: Path) -> None:
        """Stores the report inputs and prepares the relative-path resolver.

        Args:
            output_dir  : Stamp directory image links are made relative to.
            run         : Run metadata object described in the summary table.
            config      : Inference configuration the report describes.
            metrics     : Scalar metrics keyed by metric name.
            figures     : Figure paths grouped by figure key.
            report_path : Destination of the rendered markdown file.
        """
        self.output_dir  = Path(output_dir)
        self.run         = run
        self.config      = config
        self.metrics     = metrics
        self.figures     = figures
        self.report_path = Path(report_path)
        self.assets      = ReportAssets(self.output_dir)

    def _summary(self) -> dict:
        """Returns the run summary rows of the report.

        Raises:
            NotImplementedError : Always; subclasses describe their own run.
        """
        raise NotImplementedError

    def _write_metrics(self, doc: MarkdownDoc) -> None:
        """Appends one metric table per configured group, skipping empty groups.

        Args:
            doc : Document being assembled.
        """
        for title, keys in self.METRIC_GROUPS:
            rows = [(key, ScalarFormatter.format_scalar(self.metrics[key], adaptive=True)) for key in keys if key in self.metrics]
            if not rows:
                continue

            doc.heading(title, level=3)
            doc.kv_table(rows, header=("Metric", "Value"))

    def _write_figures(self, doc: MarkdownDoc) -> None:
        """Appends one image section per configured figure group, skipping empty groups.

        Args:
            doc : Document being assembled.
        """
        for title, group_keys in self.FIGURE_SECTIONS:
            paths = [path for key in group_keys for path in self.figures.get(key, [])]
            if not paths:
                continue

            doc.heading(title, level=3)
            for path in paths:
                doc.image(Path(path).stem, self.assets.rel(Path(path)))

    def assemble(self) -> Path:
        """Renders the run, metrics and figures sections and returns the written report path."""
        doc = MarkdownDoc(self.DOC_TITLE)

        doc.heading("Run", level=2)
        doc.kv_table(self._summary())

        doc.heading("Metrics", level=2)
        self._write_metrics(doc)

        if self.figures:
            doc.heading("Figures", level=2)
            self._write_figures(doc)

        return doc.save(self.report_path)
