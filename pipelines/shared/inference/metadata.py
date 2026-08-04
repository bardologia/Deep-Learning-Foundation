from __future__ import annotations

from tools.data.io         import FileIO
from tools.runtime.run_tag import RunTag


class InferenceMetadata:
    SUBDIR: str

    def __init__(self, config) -> None:
        paths = config.paths

        base = config.run_directory / "inference" / self.SUBDIR

        self.output_dir   = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir  = self.output_dir / paths.figures_subdir
        self.logs_dir     = self.output_dir / paths.logs_subdir
        self.metrics_path = self.output_dir / paths.metrics_filename
        self.report_path  = self.output_dir / paths.report_filename

    def create_dirs(self) -> None:
        FileIO.ensure_dirs(self.output_dir, self.figures_dir, self.logs_dir)
