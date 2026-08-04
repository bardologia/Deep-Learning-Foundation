from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.synthesis                    import SynthesisConfig
from pipelines.processing.synthesis.diagnostics import FitDiagnosticsBuilder
from pipelines.processing.synthesis.fields      import WhiteNoiseSampler
from pipelines.processing.synthesis.parameters  import ParameterFieldBuilder
from pipelines.processing.synthesis.scene       import ElevationBuilder, HeightAxis, PassBuilder, ProfileBuilder, TomogramBuilder
from pipelines.processing.synthesis.writer      import DatasetWriter
from tools.data.io                              import FileIO
from tools.monitoring.logger                    import Logger


class SyntheticDatasetPipeline:
    def __init__(self, config: SynthesisConfig, logger: Logger) -> None:
        config.validate()

        self.config      = config
        self.logger      = logger
        self.height_axis = HeightAxis.build(config)
        self.noise       = WhiteNoiseSampler(config.seed + 7919)
        self.writer      = DatasetWriter(config, logger=logger)

        self.logger.section("[Synthetic Dataset Pipeline Initialized]")
        self.logger.kv_table({
            "Output Directory" : str(config.output_directory),
            "Global Crop"      : config.global_crop,
            "Scene"            : f"{config.azimuth_size} azimuth x {config.range_size} range",
            "Profile Length"   : config.profile_length,
            "Height Range"     : config.height_range,
            "Components"       : config.k_max,
            "Parameter Run"    : config.params_run_name,
            "Pass Labels"      : ", ".join(config.pass_labels),
            "Seed"             : config.seed,
        }, title="Synthesis")

    def _stage_parameters(self) -> tuple:
        self.logger.section("[Stage 1] Sampling smooth mixture parameter fields")

        amps, mus, sigmas = ParameterFieldBuilder(self.config).build()
        occupancy         = (amps > self.config.activity_threshold).sum(axis=0)

        self.logger.kv_table({
            f"Pixels with {count} component(s)": f"{int((occupancy == count).sum()):,}" for count in range(self.config.k_max + 1)
        }, title="Component occupancy")

        return amps, mus, sigmas

    def _stage_scene(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        self.logger.section("[Stage 2] Evaluating profiles and observations")

        profiles                              = ProfileBuilder(self.height_axis).evaluate(amps, mus, sigmas)
        tomogram                              = TomogramBuilder(self.config, self.noise).build(profiles)
        primary, secondaries, interferograms  = PassBuilder(self.config, self.height_axis, self.noise).build(profiles)
        elevation                             = ElevationBuilder(self.config).build()

        del profiles

        return tomogram, primary, secondaries, interferograms, elevation

    def _stage_parameter_run(self, tomogram: np.ndarray, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        self.logger.section("[Stage 3] Scoring the mixture against the observed cube")

        parameters, diagnostics = FitDiagnosticsBuilder(self.config, self.height_axis).build(tomogram, amps, mus, sigmas)
        best_k                  = diagnostics["best_k_map"]

        self.logger.kv_table({
            f"Pixels retained at k={count}": f"{int((best_k == count).sum()):,}" for count in range(self.config.k_max + 1)
        }, title="Best-K distribution")

        return parameters, diagnostics

    def _stage_write(self, arrays: dict, parameters: np.ndarray, diagnostics: dict) -> dict:
        self.logger.section("[Stage 4] Writing the dataset")

        FileIO.ensure_dirs(self.config.data_directory, self.config.metadata_directory)

        self.writer.write_arrays(arrays)
        layout_path = self.writer.write_layout()
        state_path  = self.writer.write_state()
        outputs     = self.writer.write_parameters(parameters, diagnostics)

        return {"dataset_json": layout_path, "config_state": state_path, **outputs}

    def run(self) -> dict:
        amps, mus, sigmas = self._stage_parameters()

        tomogram, primary, secondaries, interferograms, elevation = self._stage_scene(amps, mus, sigmas)

        parameters, diagnostics = self._stage_parameter_run(tomogram, amps, mus, sigmas)

        arrays = {
            "primary"        : primary,
            "secondaries"    : secondaries,
            "interferograms" : interferograms,
            "dem_full"       : elevation,
            "tomogram_full"  : tomogram,
        }

        outputs = self._stage_write(arrays, parameters, diagnostics)

        self.logger.section("[Synthetic Dataset Complete]")
        self.logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

        return outputs


class SyntheticDatasetSession:
    def __init__(self, config: SynthesisConfig, log_dir: Path) -> None:
        self.config  = config
        self.log_dir = Path(log_dir)

    def execute(self) -> dict:
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger  = Logger(log_dir=str(self.log_dir), name="generate_dataset", level="INFO")
        outputs = SyntheticDatasetPipeline(self.config, logger=logger).run()

        logger.close()

        return outputs
