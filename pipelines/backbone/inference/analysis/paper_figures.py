"""Collection of inference figures from several runs into one publication figure pack.

Figures are copied as each run's inference rendered them, renamed to a stable
run-qualified name, and listed in a manifest next to them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from configuration.diagnostics                             import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.run_batch       import AnalysisConfigTable
from pipelines.backbone.inference.analysis.seed_comparison import SeedInferenceResolver
from tools.data.io                                         import FileIO
from tools.runtime.run_selector                            import ReportRunSelector
from tools.runtime.run_tag                                 import RunTag


class PaperFigurePack:
    """Copies the figures matching a set of patterns out of several runs into one directory.

    Attributes:
        config            : Run selection, figure patterns and output directory of the pack.
        logger            : Logger receiving the per-run progress.
        profile_overrides : Leaf path to value of every default the machine profile rewrote.
        output_dir        : Directory receiving the copied figures and the manifest.
    """

    MANIFEST = "figure_manifest.json"

    def __init__(self, config: PaperFigurePackConfig, logger, profile_overrides: dict) -> None:
        """Stores the pack configuration and resolves the output directory.

        Args:
            config            : Figure-pack configuration.
            logger            : Logger receiving the per-run progress.
            profile_overrides : Leaf path to value of every default the machine profile rewrote, tagged in the configuration table.
        """
        self.config            = config
        self.logger            = logger
        self.profile_overrides = profile_overrides

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

    def _collect_run(self, run_dir: Path, resolver: SeedInferenceResolver, packed_at: str, overwritten: list[str]) -> list[dict]:
        """Copies every figure of one run that matches the configured patterns.

        Args:
            run_dir     : Run to collect from.
            resolver    : Resolver picking the inference directory of the run, including seed-swept runs.
            packed_at   : Timestamp of this pack, recorded on every entry.
            overwritten : Target filenames that already existed before this pack; the run's replaced targets are appended.

        Returns:
            One manifest entry per copied figure, holding the run label, the source path,
            the target filename, the inference stamp it came from, its split, its
            completion time and the time it was packed.

        Raises:
            FileNotFoundError : If the run saved no figures directory, or if a configured pattern matches nothing in it.
        """
        inference_dir = resolver.resolve(run_dir)
        record        = SeedInferenceResolver.record(inference_dir)
        figures_dir   = inference_dir / self.config.figures_subdir

        self.logger.info(f"{run_dir.name}: inference {record['stamp']} on split '{record['split']}', completed {record['completed_at']}")

        if not figures_dir.is_dir():
            raise FileNotFoundError(f"{figures_dir} is missing; the inference saved no figures (save_plots disabled?)")

        entries = []
        for pattern in self.config.patterns:
            figures = [path for path in sorted(figures_dir.glob(pattern)) if path.is_file()]

            if not figures:
                raise FileNotFoundError(f"No figure under {figures_dir} matches '{pattern}'; that stage rendered nothing for this run, so adjust patterns or re-run inference with save_plots")

            for figure in figures:
                target = self.output_dir / self._stable_name(run_dir, figures_dir, figure)
                if target.exists():
                    overwritten.append(target.name)

                shutil.copy2(figure, target)
                entries.append({"run": self._run_label(run_dir), "source": str(figure), "target": target.name, "inference": str(inference_dir), "split": record["split"], "completed_at": record["completed_at"], "packed_at": packed_at})

        return entries

    def _merge_manifest(self, entries: list[dict]) -> list[dict]:
        """Merges this pack's entries into the manifest already in the output directory.

        Earlier entries are kept for the targets this pack did not rewrite and that are
        still on disk, so figures left from an earlier pack keep their recorded source.

        Args:
            entries : Manifest entries of the figures copied by this pack.

        Returns:
            Every manifest entry, keyed uniquely by target and sorted by target.
        """
        manifest_path = self.output_dir / self.MANIFEST
        merged        = {}

        if manifest_path.is_file():
            merged = {entry["target"]: entry for entry in FileIO.load_json(manifest_path)["figures"] if (self.output_dir / entry["target"]).is_file()}

        kept = len(set(merged) - {entry["target"] for entry in entries})
        if kept:
            self.logger.info(f"{kept} figure(s) from earlier packs keep their manifest entries")

        merged.update({entry["target"]: entry for entry in entries})

        return [merged[target] for target in sorted(merged)]

    def _report_uncovered(self, manifest: list[dict]) -> None:
        """Warns about files in the output directory that no manifest entry records a source for."""
        covered   = {entry["target"] for entry in manifest} | {self.MANIFEST}
        uncovered = sorted(path.name for path in self.output_dir.iterdir() if path.is_file() and path.name not in covered)

        if uncovered:
            self.logger.warning(f"{len(uncovered)} file(s) in {self.output_dir} have no manifest entry and no recorded source: {', '.join(uncovered[:8])}" + (", ..." if len(uncovered) > 8 else ""))

    def run(self) -> dict:
        """Packs the figures of every selected run and writes the merged manifest.

        Returns:
            Payload with the output directory, the patterns used, a note on the figure
            formats and the merged manifest entries, earlier packs' entries included.
        """
        self.logger.section("Paper figure pack")
        AnalysisConfigTable.log(self.logger, self.config, self.profile_overrides)

        FileIO.ensure_dirs(self.output_dir)

        resolver    = SeedInferenceResolver(self.config.inference_subdir, self.config.split, self.config.metrics_filename, self.logger)
        packed_at   = RunTag.timestamp()
        packed      = []
        overwritten = []

        for run_dir in self._select_runs():
            self.logger.subsection(f"Run: {run_dir}")
            entries = self._collect_run(run_dir, resolver, packed_at, overwritten)
            packed += entries
            self.logger.ok(f"{run_dir.name}: {len(entries)} figures packed")

        if overwritten:
            self.logger.warning(f"{len(overwritten)} figure(s) overwritten in {self.output_dir}: {', '.join(overwritten[:8])}" + (", ..." if len(overwritten) > 8 else ""))

        manifest = self._merge_manifest(packed)
        self._report_uncovered(manifest)

        payload = {
            "output_dir" : str(self.output_dir),
            "split"      : self.config.split,
            "patterns"   : list(self.config.patterns),
            "note"       : "Figures are copied as rendered by each run's inference, whatever the extension: the paper style writes line and scatter figures as .pdf and keeps raster maps as .png. Re-run inference with figure_style=paper for publication styling before packing.",
            "figures"    : manifest,
        }
        FileIO.save_json(payload, self.output_dir / self.MANIFEST)

        self.logger.ok(f"Figure pack: {len(packed)} figures packed, {len(overwritten)} overwritten, {len(manifest)} in the manifest -> {self.output_dir}")

        return payload
