
from __future__ import annotations

from pathlib import Path

from configuration.inference                   import InferenceConfig
from pipelines.shared.inference.metadata       import TargetDataset
from pipelines.shared.inference.run_classifier import InferenceLayout, RunType
from tools.data.io                             import FileIO
from tools.runtime.environment_capture         import EnvironmentCapture
from tools.runtime.run_tag                     import RunTag


class InferenceMetadata:

    CONFIG_FILENAME = "inference_config.json"

    def __init__(self, config: InferenceConfig, target: TargetDataset | None = None) -> None:
        self.config = config
        paths       = config.paths

        base = InferenceLayout.base(config.run_directory, RunType.BACKBONE) if target is None else InferenceLayout.target_base(config.run_directory, RunType.BACKBONE, target.name)
        self.output_dir     = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir    = self.output_dir / paths.figures_subdir
        self.animations_dir = self.output_dir / paths.animations_subdir
        self.logs_dir       = self.output_dir / paths.logs_subdir
        self.cube_dir       = self.output_dir / paths.cubes_subdir
        self.metrics_path   = self.output_dir / paths.metrics_filename
        self.report_path    = self.output_dir / paths.report_filename
        self.config_path    = self.output_dir / self.CONFIG_FILENAME

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
        FileIO.save_json(EnvironmentCapture.snapshot(), self.output_dir / "environment.json")
