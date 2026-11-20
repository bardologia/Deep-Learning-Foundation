from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from configuration.generation                      import GenerateDatasetEntryConfig, GenerationConfig
from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.dataset.dataset_files        import DatasetFiles
from tools.data.regions                            import CropRegion, SplitRegions
from tools.monitoring.logger                       import Logger
from tools.runtime.git_state                       import RunProvenance


class InputFields:

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng    = rng

    @property
    def names(self) -> list[str]:
        return [f"x{index}" for index in range(self.config.input_channels)]

    def smooth_field(self) -> np.ndarray:
        noise = self.rng.standard_normal((self.config.rows, self.config.cols))
        field = gaussian_filter(noise, sigma=self.config.smoothing_pixels, mode="reflect")

        return (field - field.mean()) / field.std()

    def build(self) -> np.ndarray:
        fields = np.stack([self.smooth_field() for _ in self.names])
        noise  = self.config.input_noise * self.rng.standard_normal(fields.shape)

        return (fields + noise).astype(np.float32)


class TargetFunction:

    NAMES = ["pointwise", "neighbourhood"]

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng    = rng

    def pointwise(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs[0] * inputs[1]) + 0.5 * inputs[2]

    def neighbourhood(self, inputs: np.ndarray) -> np.ndarray:
        size = self.config.neighbourhood

        return uniform_filter(inputs[3] ** 2, size=size, mode="reflect") - uniform_filter(inputs[0], size=size, mode="reflect")

    def apply(self, inputs: np.ndarray) -> np.ndarray:
        targets = np.stack([self.pointwise(inputs), self.neighbourhood(inputs)])
        noise   = self.config.target_noise * self.rng.standard_normal(targets.shape)

        return (targets + noise).astype(np.float32)


class RowBandSplits:

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def build(self) -> SplitRegions:
        rows, cols = self.config.rows, self.config.cols
        train_end  = round(rows * (1.0 - self.config.val_fraction - self.config.test_fraction))
        val_end    = round(rows * (1.0 - self.config.test_fraction))

        return SplitRegions(
            train = [CropRegion(0, train_end, 0, cols)],
            val   = [CropRegion(train_end, val_end, 0, cols)],
            test  = [CropRegion(val_end, rows, 0, cols)],
        )


class DatasetGenerator:

    def __init__(self, config: GenerationConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.rng    = np.random.default_rng(config.seed)

    def _provenance(self) -> dict:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self.config).items()}

        return {"config": config, **RunProvenance.collect()}

    def _log_dataset(self, files: DatasetFiles, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.logger.section("[Synthetic Dataset Written]")
        self.logger.kv_table({
            "Directory"       : str(files.directory),
            "Grid"            : f"{files.rows} rows x {files.cols} cols",
            "Input channels"  : ", ".join(files.input_channels),
            "Target channels" : ", ".join(files.target_channels),
            "Train rows"      : str([region.as_tuple() for region in files.splits.regions("train")]),
            "Val rows"        : str([region.as_tuple() for region in files.splits.regions("val")]),
            "Test rows"       : str([region.as_tuple() for region in files.splits.regions("test")]),
            "Seed"            : self.config.seed,
        })

        rows = []
        for name, values in [*zip(files.input_channels, inputs), *zip(files.target_channels, targets)]:
            rows.append({"Channel": name, "Mean": f"{values.mean():+.4f}", "Std": f"{values.std():.4f}", "Min": f"{values.min():+.4f}", "Max": f"{values.max():+.4f}"})

        self.logger.metrics_table(rows, ["Channel", "Mean", "Std", "Min", "Max"], title="Channel statistics over the whole grid")

    def run(self) -> DatasetFiles:
        fields  = InputFields(self.config, self.rng)
        inputs  = fields.build()
        targets = TargetFunction(self.config, self.rng).apply(inputs)

        files = DatasetFiles(
            directory       = Path(self.config.output_directory),
            name            = Path(self.config.output_directory).name,
            input_channels  = fields.names,
            target_channels = list(TargetFunction.NAMES),
            rows            = self.config.rows,
            cols            = self.config.cols,
            splits          = RowBandSplits(self.config).build(),
            generator       = self._provenance(),
        )

        files.write(inputs, targets)
        self._log_dataset(files, inputs, targets)

        return files


class GenerateDatasetSession:

    def __init__(self, entry_config: GenerateDatasetEntryConfig) -> None:
        self.entry_config = entry_config

    def execute(self) -> DatasetFiles:
        log_dir = Path(self.entry_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        with Logger(log_dir=str(log_dir), name="generate_dataset", level="INFO") as logger:
            files = DatasetGenerator(self.entry_config.generation, logger).run()

            ResolvedEntryConfigWriter.save(self.entry_config, files.directory, logger)

            return files
