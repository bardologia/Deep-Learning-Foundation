"""Export of run metrics as publication-ready CSV, LaTeX and markdown tables.

Walks a runs directory for finished inference stamps, collects each run's
scalar metrics, and renders one row per run: the CSV carries every scalar key
for downstream analysis, while the LaTeX and markdown tables carry the
selected columns at a fixed precision for direct inclusion in a paper.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.backbone.inference.analysis.seed_comparison import SeedInferenceResolver
from tools.data.io                                         import FileIO
from tools.metrics.scoring                                 import FiniteScalar
from tools.monitoring.logger                               import Logger
from tools.reporting.markdown                              import MarkdownTable


class ResultTableExporter:
    """Collects run metrics under a root directory and writes them as tables.

    Attributes:
        entry    : Entry configuration naming the runs root, columns and output.
        logger   : Logger receiving the discovery and output lines.
        resolver : Resolver picking each run's latest finished inference stamp.
    """

    def __init__(self, entry, logger: Logger) -> None:
        """Stores the entry configuration and builds the inference resolver.

        Args:
            entry  : ResultTableEntryConfig instance.
            logger : Logger for progress and result lines.
        """
        self.entry    = entry
        self.logger   = logger
        self.resolver = SeedInferenceResolver(entry.inference_subdir, entry.metrics_filename)

    def _discover(self) -> list[tuple[str, Path]]:
        """Returns (run name, inference dir) pairs for every run with finished inference.

        The run name is the run directory path relative to the runs root. An
        empty run_filter admits every discovered run; otherwise only runs whose
        name contains one of the filter strings survive.

        Raises:
            FileNotFoundError : If the runs root does not exist or no run under it has finished inference.
        """
        root = Path(self.entry.runs_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Runs directory {root} does not exist; train and infer runs before exporting tables")

        rows = []
        for inference_root in sorted(root.rglob("inference")):
            if not inference_root.is_dir():
                continue

            run_dir = inference_root.parent
            name    = run_dir.relative_to(root).as_posix()

            if self.entry.run_filter and not any(token in name for token in self.entry.run_filter):
                continue

            try:
                rows.append((name, self.resolver.resolve(run_dir)))
            except FileNotFoundError:
                self.logger.warning(f"Skipping {name}: no finished inference with {self.entry.metrics_filename}")

        if not rows:
            raise FileNotFoundError(f"No run under {root} holds a finished inference with {self.entry.metrics_filename}")

        return rows

    def _load(self, rows: list[tuple[str, Path]]) -> list[tuple[str, dict]]:
        """Returns each run's name with its scalar metrics, non-scalar values dropped."""
        loaded = []
        for name, inference_dir in rows:
            metrics = FileIO.load_json(inference_dir / self.entry.metrics_filename)
            scalars = {key: FiniteScalar.coerce(value) for key, value in metrics.items()}
            loaded.append((name, {key: value for key, value in scalars.items() if value is not None}))

        return loaded

    def _columns(self, loaded: list[tuple[str, dict]]) -> list[str]:
        """Returns the table columns: the configured metrics, or every scalar key seen.

        Raises:
            KeyError : If a configured metric key appears in no run.
        """
        seen = sorted({key for _name, metrics in loaded for key in metrics})

        if not self.entry.metrics:
            return seen

        missing = [key for key in self.entry.metrics if key not in seen]
        if missing:
            raise KeyError(f"Configured metric keys {missing} appear in no discovered run; available keys include {seen[:20]}")

        return list(self.entry.metrics)

    def _sorted(self, loaded: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Returns the rows ordered by the configured sort key, discovery order when unset."""
        if not self.entry.sort_by:
            return loaded

        key = self.entry.sort_by
        return sorted(loaded, key=lambda row: (row[1].get(key) is None, row[1].get(key, 0.0)), reverse=not self.entry.ascending)

    def _fmt(self, value: float | None) -> str:
        """Returns a value at the configured precision, an empty cell for None."""
        if value is None:
            return ""

        return f"{value:.{self.entry.precision}g}"

    def _write_csv(self, loaded: list[tuple[str, dict]], out_dir: Path) -> Path:
        """Writes every scalar key of every run as results_table.csv, full precision."""
        keys  = sorted({key for _name, metrics in loaded for key in metrics})
        lines = [",".join(["run", *keys])]

        for name, metrics in loaded:
            cells = [repr(metrics[key]) if key in metrics else "" for key in keys]
            lines.append(",".join([name, *cells]))

        path = out_dir / "results_table.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return path

    @staticmethod
    def _latex_escape(text: str) -> str:
        """Returns the text with LaTeX-special characters escaped for a tabular cell."""
        for char in ("&", "%", "#", "_"):
            text = text.replace(char, f"\\{char}")

        return text

    def _write_latex(self, loaded: list[tuple[str, dict]], columns: list[str], out_dir: Path) -> Path:
        """Writes the selected columns as a booktabs tabular in results_table.tex."""
        lines = [
            f"\\begin{{tabular}}{{l{'r' * len(columns)}}}",
            "\\toprule",
            " & ".join(["Run", *[self._latex_escape(key) for key in columns]]) + " \\\\",
            "\\midrule",
        ]

        for name, metrics in loaded:
            cells = [self._fmt(metrics.get(key)) for key in columns]
            lines.append(" & ".join([self._latex_escape(name), *cells]) + " \\\\")

        lines += ["\\bottomrule", "\\end{tabular}"]

        path = out_dir / "results_table.tex"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return path

    def _write_markdown(self, loaded: list[tuple[str, dict]], columns: list[str], out_dir: Path) -> Path:
        """Writes the selected columns as a markdown table in results_table.md."""
        table = MarkdownTable(("Run", *columns))
        for name, metrics in loaded:
            table.add_row(f"`{name}`", *[self._fmt(metrics.get(key)) for key in columns])

        path = out_dir / "results_table.md"
        path.write_text("\n".join(table.render()) + "\n", encoding="utf-8")

        return path

    def run(self) -> dict[str, Path]:
        """Discovers the runs, loads their metrics and writes the three tables.

        Returns:
            Mapping with the written "csv", "tex" and "md" paths.
        """
        rows   = self._discover()
        loaded = self._sorted(self._load(rows))

        columns = self._columns(loaded)
        out_dir = FileIO.ensure_dir(Path(self.entry.output_dir))

        paths = {
            "csv" : self._write_csv(loaded, out_dir),
            "tex" : self._write_latex(loaded, columns, out_dir),
            "md"  : self._write_markdown(loaded, columns, out_dir),
        }

        self.logger.section("[Result tables]")
        self.logger.kv_table({
            "Runs exported" : len(loaded),
            "Columns"       : f"{len(columns)} ({'configured' if self.entry.metrics else 'every scalar key'})",
            "CSV"           : paths["csv"],
            "LaTeX"         : paths["tex"],
            "Markdown"      : paths["md"],
        })

        return paths
