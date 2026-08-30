"""Collection of inference figures from several runs into one publication figure pack.

Figures are copied as each run's inference rendered them, renamed to a stable
run-qualified name, and listed in a manifest next to them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from configuration.diagnostics                             import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.seed_comparison import SeedInferenceResolver
from tools.data.io                                         import FileIO
from tools.runtime.run_selector                            import ReportRunSelector


class PaperFigurePack:
    """Copies the figures matching a set of patterns out of several runs into one directory.

    Attributes:
        config     : Run selection, figure patterns and output directory of the pack.
        output_dir : Directory receiving the copied figures and the manifest.
    """

    MANIFEST = "figure_manifest.json"

    def __init__(self, config: PaperFigurePackConfig, logger) -> None:
        """Stores the pack configuration and resolves the output directory.

        Args:
            config : Figure-pack configuration.
            logger : Logger receiving the per-run progress.
        """
        self.config = config
        self.logger = logger

        self.output_dir = Path(config.output_dir)

    def _select_runs(self) -> list[Path]:
        """Returns the run directories that hold an inference report and match the filter."""
        selector = ReportRunSelector(self.config.runs_dir, "inference", self.config.report_filename, self.logger)
        return selector.resolve(self.config.run_filter)

    def _run_label(self, run_dir: Path) -> str:
        """Returns a flat label for a run, joining its path under the runs directory."""
        relative = run_dir.relative_to(self.config.runs_dir) if run_dir.is_relative_to(self.config.runs_dir) else Path(run_dir.name)
        return "__".join(relative.parts)

    def _stable_name(self, run_dir: Path, figures_dir: Path, figure: Path) -> str:
        """Builds the target filename of a figure, qualified by its run and its subpath.

        Args:
            run_dir     : Run the figure comes from.
            figures_dir : Figures directory of that run's inference.
            figure      : Source figure path.

        Returns:
            A flat filename keeping the run label, the figure subpath and the extension.
        """
        relative = figure.relative_to(figures_dir)
        return f"{self._run_label(run_dir)}__{'__'.join(relative.with_suffix('').parts)}{figure.suffix}"

    def _collect_run(self, run_dir: Path, resolver: SeedInferenceResolver) -> list[dict]:
        """Copies every figure of one run that matches the configured patterns.

        Args:
            run_dir  : Run to collect from.
            resolver : Resolver picking the inference directory of the run, including seed-swept runs.

        Returns:
            One manifest entry per copied figure, holding the run label, the source path
            and the target filename.

        Raises:
            FileNotFoundError : If the run saved no figures directory, or if a configured pattern matches nothing in it.
        """
        inference_dir = resolver.resolve(run_dir)
        figures_dir   = inference_dir / self.config.figures_subdir

        if not figures_dir.is_dir():
            raise FileNotFoundError(f"{figures_dir} is missing; the inference saved no figures (save_plots disabled?)")

        entries = []
        for pattern in self.config.patterns:
            figures = [path for path in sorted(figures_dir.glob(pattern)) if path.is_file()]

            if not figures:
                raise FileNotFoundError(f"No figure under {figures_dir} matches '{pattern}'; that stage rendered nothing for this run, so adjust patterns or re-run inference with save_plots")

            for figure in figures:
                target = self.output_dir / self._stable_name(run_dir, figures_dir, figure)
                shutil.copy2(figure, target)
                entries.append({"run": self._run_label(run_dir), "source": str(figure), "target": target.name})

        return entries

    def run(self) -> dict:
        """Packs the figures of every selected run and writes the manifest.

        Returns:
            Payload with the output directory, the patterns used, a note on the figure
            formats and the manifest entries of the copied figures.
        """
        FileIO.ensure_dirs(self.output_dir)

        resolver = SeedInferenceResolver(self.config.inference_subdir, self.config.metrics_filename)
        manifest = []

        for run_dir in self._select_runs():
            self.logger.subsection(f"Run: {run_dir}")
            entries   = self._collect_run(run_dir, resolver)
            manifest += entries
            self.logger.ok(f"{run_dir.name}: {len(entries)} figures packed")

        payload = {
            "output_dir" : str(self.output_dir),
            "patterns"   : list(self.config.patterns),
            "note"       : "Figures are copied as rendered by each run's inference, whatever the extension: the paper style writes line and scatter figures as .pdf and keeps raster maps as .png. Re-run inference with figure_style=paper for publication styling before packing.",
            "figures"    : manifest,
        }
        FileIO.save_json(payload, self.output_dir / self.MANIFEST)

        self.logger.ok(f"Figure pack: {len(manifest)} figures -> {self.output_dir}")

        return payload
