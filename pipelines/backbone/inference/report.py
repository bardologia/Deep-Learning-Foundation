from __future__ import annotations

from pathlib import Path

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


class Report:

    def __init__(self, run, metrics: dict, figure_paths: dict[str, dict[str, Path]], output_dir: Path) -> None:
        self.run          = run
        self.metrics      = metrics
        self.figure_paths = figure_paths
        self.output_dir   = Path(output_dir)

    def _summary(self, doc: MarkdownDoc) -> None:
        doc.heading("Run", level=2)
        doc.kv_table({
            "Run directory"    : str(self.metrics["run_directory"]),
            "Model"            : self.run.backbone_name,
            "Checkpoint"       : f"{self.metrics['checkpoint_name']} (epoch {self.metrics['checkpoint_epoch']})",
            "Dataset"          : str(self.run.dataset_config.dataset_directory),
            "Split"            : self.run.split_name,
            "Region"           : str(self.run.split_region.as_tuple()),
            "Input channels"   : ", ".join(self.run.input_channels),
            "Target channels"  : ", ".join(self.run.target_channels),
            "Scored pixels"    : f"{self.metrics['n_pixels']:,} per channel",
        }, code_keys=False)

    def _metrics(self, doc: MarkdownDoc) -> None:
        table = MarkdownTable(["Target channel", "RMSE", "MAE", "Bias", "R2"], align=["left", "right", "right", "right", "right"])

        for name, values in self.metrics["channels"].items():
            table.add_row(name, f"{values['rmse']:.5g}", f"{values['mae']:.5g}", f"{values['bias']:+.5g}", f"{values['r2']:.4f}")

        table.add_row("mean", f"{self.metrics['rmse_mean']:.5g}", f"{self.metrics['mae_mean']:.5g}", "", f"{self.metrics['r2_mean']:.4f}")

        doc.heading("Metrics", level=2)
        doc.paragraph("Errors are measured in the units of each target channel, on the stitched prediction of the whole region.")
        doc.table(table)

    def _figures(self, doc: MarkdownDoc) -> None:
        if not self.figure_paths:
            return

        doc.heading("Figures", level=2)

        for name, paths in self.figure_paths.items():
            doc.heading(name, level=3)
            for label, path in paths.items():
                doc.image(f"{name} {label}", Path(path).relative_to(self.output_dir).as_posix())

    def write(self, path: Path) -> Path:
        doc = MarkdownDoc(title="Inference report")

        self._summary(doc)
        self._metrics(doc)
        self._figures(doc)

        return doc.save(path)
