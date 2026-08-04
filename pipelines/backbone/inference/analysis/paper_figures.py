from __future__ import annotations

import shutil
from pathlib import Path

from configuration.diagnostics                             import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.seed_comparison import SeedInferenceResolver
from tools.data.io                                         import FileIO
from tools.runtime.run_selector                            import ReportRunSelector


class PaperFigurePack:

    MANIFEST = "figure_manifest.json"

    def __init__(self, config: PaperFigurePackConfig, logger) -> None:
        self.config = config
        self.logger = logger

        self.output_dir = Path(config.output_dir)

    def _select_runs(self) -> list[Path]:
        selector = ReportRunSelector(self.config.runs_dir, "inference", self.config.report_filename, self.logger)
        return selector.resolve(self.config.run_filter)

    def _run_label(self, run_dir: Path) -> str:
        relative = run_dir.relative_to(self.config.runs_dir) if run_dir.is_relative_to(self.config.runs_dir) else Path(run_dir.name)
        return "__".join(relative.parts)

    def _stable_name(self, run_dir: Path, figures_dir: Path, figure: Path) -> str:
        relative = figure.relative_to(figures_dir)
        return f"{self._run_label(run_dir)}__{'__'.join(relative.with_suffix('').parts)}{figure.suffix}"

    def _collect_run(self, run_dir: Path, resolver: SeedInferenceResolver) -> list[dict]:
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
