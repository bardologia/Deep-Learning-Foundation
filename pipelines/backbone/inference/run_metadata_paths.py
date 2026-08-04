from __future__ import annotations

from pathlib import Path

from configuration.inference import InferenceConfig
from tools.data.io           import FileIO
from tools.runtime.run_tag   import RunTag


class InferenceMetadata:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        paths       = config.paths

        base = config.run_directory / "inference"
        self.output_dir     = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir    = self.output_dir / paths.figures_subdir
        self.animations_dir = self.output_dir / paths.animations_subdir
        self.logs_dir       = self.output_dir / paths.logs_subdir
        self.cube_dir       = self.output_dir / paths.cubes_subdir
        self.metrics_path   = self.output_dir / paths.metrics_filename
        self.report_path    = self.output_dir / paths.report_filename

    def figure_path(self, name: str, ext: str = "png") -> Path:
        return self.figures_dir / f"{name}.{ext}"

    def create_dirs(self) -> None:
        FileIO.ensure_dirs(
            self.output_dir,
            self.figures_dir,
            self.animations_dir,
            self.logs_dir,
            self.cube_dir,
        )
