"""Output directory layout of a backbone inference run.

Derives the figure, animation, log, cube, metric and report paths from the inference
configuration, either under an explicit output subdirectory or under a fresh timestamp.
"""

from __future__ import annotations

from pathlib import Path

from configuration.inference           import InferenceConfig
from tools.data.io                     import FileIO
from tools.runtime.environment_capture import EnvironmentCapture
from tools.runtime.run_tag             import RunTag


class InferenceMetadata:
    """Resolved output paths of one inference run.

    Attributes:
        config         : Inference configuration the paths were derived from.
        output_dir     : Root directory of this inference run inside the training run directory.
        figures_dir    : Directory receiving the report figures.
        animations_dir : Directory receiving the cube animations.
        logs_dir       : Directory receiving the run logs.
        cube_dir       : Directory receiving the stitched cubes and metric maps.
        metrics_path   : Path of the serialised metric dictionary.
        report_path    : Path of the generated markdown report.
    """

    def __init__(self, config: InferenceConfig) -> None:
        """Derives every output path from the configuration, timestamping when no subdir is set."""
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
        """Returns the path of a named figure inside the figures directory."""
        return self.figures_dir / f"{name}.{ext}"

    def create_dirs(self) -> None:
        """Creates the output, figure, animation, log and cube directories and records the environment snapshot."""
        FileIO.ensure_dirs(
            self.output_dir,
            self.figures_dir,
            self.animations_dir,
            self.logs_dir,
            self.cube_dir,
        )
        FileIO.save_json(EnvironmentCapture.snapshot(), self.output_dir / "environment.json")
