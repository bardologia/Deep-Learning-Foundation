from __future__ import annotations

import re
from pathlib import Path
from typing  import List

from configuration.diagnostics  import ReportCollectionConfig, ReportCollectionEntryConfig
from tools.data.io              import FileIO
from tools.reporting.reporting  import ReportAssets
from tools.runtime.run_selector import ReportRunSelector


class RunReportLocator:
    def __init__(self, config: ReportCollectionConfig) -> None:
        self.config = config

    def locate(self) -> List[Path]:
        inference_dir = self.config.inference_directory
        if not inference_dir.is_dir():
            raise FileNotFoundError(f"No '{self.config.inference_dirname}' directory at {inference_dir}; expected a training run directory holding inference outputs")

        reports = sorted(path for path in inference_dir.glob(f"*/{self.config.report_filename}") if path.is_file())
        if not reports:
            raise FileNotFoundError(f"No '{self.config.report_filename}' found in any inference output under {inference_dir}")

        if self.config.latest_only:
            return [reports[-1]]
        return reports


class ReportImageRewriter:
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
    PASSTHROUGH   = ("data:", "http://", "https://")

    def __init__(self, report_path: Path, embed_images: bool) -> None:
        self.report_dir   = Path(report_path).parent
        self.embed_images = embed_images
        self.assets       = ReportAssets(self.report_dir, embed_images=True)

    def _resolve(self, target: str) -> Path:
        path = Path(target)
        if not path.is_absolute():
            path = self.report_dir / path

        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Report references a missing image: {target} (resolved to {path})")

        return path

    def _substitute(self, match: re.Match) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(self.PASSTHROUGH):
            return match.group(0)

        path = self._resolve(target)
        src  = self.assets.src(path) if self.embed_images else path.as_posix()
        return f"![{label}]({src})"

    def rewrite(self, text: str) -> str:
        return self.IMAGE_PATTERN.sub(self._substitute, text)


class ReportCollection:
    def __init__(self, config: ReportCollectionConfig, logger) -> None:
        self.config = config
        self.logger = logger

    @staticmethod
    def run_label(run_dir: Path) -> str:
        run_dir = Path(run_dir)
        if re.fullmatch(r"seed\d+", run_dir.name):
            return f"{run_dir.parent.name}_{run_dir.name}"
        return run_dir.name

    def _output_name(self, report_path: Path) -> str:
        run_name = self.run_label(self.config.run_directory)
        if self.config.latest_only:
            return f"{run_name}.md"
        return f"{run_name}_{report_path.parent.name}.md"

    def _collect_one(self, report_path: Path) -> Path:
        text        = report_path.read_text(encoding="utf-8")
        rewritten   = ReportImageRewriter(report_path, self.config.embed_images).rewrite(text)
        destination = Path(self.config.collector_dir) / self._output_name(report_path)

        destination.write_text(rewritten, encoding="utf-8")
        self.logger.ok(f"{report_path.parent.name}: collected as {destination.name}")
        return destination

    def run(self) -> dict:
        self.logger.subsection(f"Collecting reports: {Path(self.config.run_directory).name}")

        reports   = RunReportLocator(self.config).locate()
        collected = [self._collect_one(report_path) for report_path in reports]

        return {
            "run_directory"   : str(self.config.run_directory),
            "collector_dir"   : str(self.config.collector_dir),
            "n_reports"       : len(collected),
            "collected_paths" : collected,
        }


class ReportCollectionBatch:
    def __init__(self, entry_config: ReportCollectionEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger

    def _select_runs(self) -> list[Path]:
        selector = ReportRunSelector(self.entry_config.runs_dir, self.entry_config.inference_dirname, self.entry_config.report_filename, self.logger)
        return selector.resolve(self.entry_config.run_filter)

    def _check_collisions(self, run_dirs: list[Path]) -> None:
        names      = [ReportCollection.run_label(run_dir) for run_dir in run_dirs]
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(f"Selected runs share a name, their collected reports would collide: {duplicates}")

    def _collect_run(self, run_dir: Path) -> dict:
        config = self.entry_config.to_config(run_dir)
        return ReportCollection(config, self.logger).run()

    def run(self) -> list[dict]:
        self.logger.section(f"Report collection into {self.entry_config.collector_dir}")

        run_dirs = self._select_runs()
        self._check_collisions(run_dirs)

        FileIO.ensure_dirs(Path(self.entry_config.collector_dir))
        results = [self._collect_run(run_dir) for run_dir in run_dirs]

        self.logger.ok(f"Collected {sum(result['n_reports'] for result in results)} report(s) from {len(results)} run(s) into {self.entry_config.collector_dir}")
        return results
