
from __future__ import annotations

from configuration.inference                   import InferenceConfig
from pipelines.shared.inference.metadata       import TargetDataset
from pipelines.shared.inference.run_classifier import InferenceLayout
from tools.data.io                             import FileIO
from tools.runtime.environment_capture         import EnvironmentCapture
from tools.runtime.run_tag                     import RunTag


class InferenceMetadata:

    CONFIG_FILENAME = "inference_config.json"

    def __init__(self, config: InferenceConfig, target: TargetDataset | None = None) -> None:
        self.config = config
        paths       = config.paths

        base = InferenceLayout.base(config.run_directory) if target is None else InferenceLayout.target_base(config.run_directory, target.name)
        self.output_dir   = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir  = self.output_dir / paths.figures_subdir
        self.logs_dir     = self.output_dir / paths.logs_subdir
        self.maps_dir     = self.output_dir / paths.maps_subdir
        self.metrics_path = self.output_dir / paths.metrics_filename
        self.report_path  = self.output_dir / paths.report_filename
        self.config_path  = self.output_dir / self.CONFIG_FILENAME

    def create_dirs(self) -> None:
        FileIO.ensure_dirs(self.output_dir, self.figures_dir, self.logs_dir, self.maps_dir)
        FileIO.save_json(EnvironmentCapture.snapshot(), self.output_dir / "environment.json")
