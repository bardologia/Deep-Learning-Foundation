"""Generates a synthetic Gaussian-mixture dataset with the layout the training and inference pipelines read.

The pipeline samples smooth parameter fields, evaluates the profiles, cube and
channels they imply, scores the mixture against the observed cube as a
parameter run would, and writes the dataset. The session refuses a directory
already holding a finished dataset and records an abort in its log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.synthesis                    import SynthesisConfig
from pipelines.processing.synthesis.diagnostics import FitDiagnosticsBuilder
from pipelines.processing.synthesis.fields      import WhiteNoiseSampler
from pipelines.processing.synthesis.parameters  import ParameterFieldBuilder
from pipelines.processing.synthesis.scene       import AuxFieldBuilder, ZAxis, ChannelBuilder, ProfileBuilder, CubeBuilder
from pipelines.processing.synthesis.writer      import DatasetDirectoryGuard, DatasetWriter
from tools.data.gaussians                       import GaussianActivity
from tools.data.io                              import FileIO
from tools.monitoring.logger                    import Logger


class SyntheticDatasetPipeline:
    """Runs the four generation stages and writes the dataset.

    Attributes:
        config : Synthesis configuration describing the scene and the output directory.
        logger : Logger for the stage report.
        z_axis : Z axis the profiles are sampled on, in metres.
        noise  : White-noise sampler shared by the cube and the channels.
        writer : Writer of the dataset files.
    """

    def __init__(self, config: SynthesisConfig, logger: Logger) -> None:
        """Validates the configuration and builds the axis, the noise sampler and the writer.

        Args:
            config : Synthesis configuration describing the scene and the output directory.
            logger : Logger for the stage report.
        """
        config.validate()

        self.config = config
        self.logger = logger
        self.z_axis = ZAxis.build(config)
        self.noise  = WhiteNoiseSampler(config.seed + 7919)
        self.writer = DatasetWriter(config, logger=logger)

        self.logger.section("[Synthetic Dataset Pipeline Initialized]")
        self.logger.kv_table({
            "Output Directory" : str(config.output_directory),
            "Global Crop"      : config.global_crop,
            "Scene"            : f"{config.row_size} rows x {config.col_size} cols",
            "Profile Length"   : config.profile_length,
            "Z Range"          : config.z_range,
            "Components"       : config.k_max,
            "Parameter Run"    : config.params_run_name,
            "Channel Labels"   : ", ".join(config.channel_labels),
            "Seed"             : config.seed,
        }, title="Synthesis")

    def _stage_parameters(self) -> tuple:
        """Samples the smooth amplitude, mean and sigma fields and logs the component occupancy.

        Returns:
            Tuple of amplitudes, means in metres and sigmas in metres, each of shape (k_max, rows, cols).
        """
        self.logger.section("[Stage 1] Sampling smooth mixture parameter fields")

        amps, mus, sigmas = ParameterFieldBuilder(self.config).build()
        occupancy         = GaussianActivity.is_active(amps, self.config.activity_threshold).sum(axis=0)

        self.logger.kv_table({
            f"Pixels with {count} component(s)": f"{int((occupancy == count).sum()):,}" for count in range(self.config.k_max + 1)
        }, title="Component occupancy")

        return amps, mus, sigmas

    def _stage_scene(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        """Evaluates the profiles and derives the cube, the channels and the auxiliary field from them.

        Args:
            amps   : Amplitudes of shape (k_max, rows, cols).
            mus    : Means of shape (k_max, rows, cols), in metres.
            sigmas : Sigmas of shape (k_max, rows, cols), in metres.

        Returns:
            Tuple of the cube, the primary, secondary and derived channels, and the auxiliary field.
        """
        self.logger.section("[Stage 2] Evaluating profiles and observations")

        profiles                      = ProfileBuilder(self.z_axis).evaluate(amps, mus, sigmas)
        cube                          = CubeBuilder(self.config, self.noise).build(profiles)
        primary, secondaries, derived = ChannelBuilder(self.config, self.z_axis, self.noise).build(profiles)
        z                             = AuxFieldBuilder(self.config).build()

        del profiles

        return cube, primary, secondaries, derived, z

    def _stage_parameter_run(self, cube: np.ndarray, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        """Scores the mixture against the observed cube as a fitted parameter run would.

        Args:
            cube   : Complex cube of shape (n_z, rows, cols).
            amps   : Amplitudes of shape (k_max, rows, cols).
            mus    : Means of shape (k_max, rows, cols), in metres.
            sigmas : Sigmas of shape (k_max, rows, cols), in metres.

        Returns:
            Tuple of the parameter stack of shape (3 * k_max, rows, cols) and its fit diagnostics.
        """
        self.logger.section("[Stage 3] Scoring the mixture against the observed cube")

        parameters, diagnostics = FitDiagnosticsBuilder(self.config, self.z_axis).build(cube, amps, mus, sigmas)
        best_k                  = diagnostics["best_k_map"]

        self.logger.kv_table({
            f"Pixels retained at k={count}": f"{int((best_k == count).sum()):,}" for count in range(self.config.k_max + 1)
        }, title="Best-K distribution")

        return parameters, diagnostics

    def _stage_write(self, arrays: dict, parameters: np.ndarray, diagnostics: dict) -> dict:
        """Writes the arrays, the configuration state and the parameter run, then the layout last.

        The layout marks a finished dataset for every reader and for the
        directory guard, so a generation that fails part-way leaves no layout
        behind.

        Args:
            arrays      : Mapping holding one array per ``DatasetWriter.ARTIFACTS`` key.
            parameters  : Parameter stack of shape (3 * k_max, rows, cols).
            diagnostics : Fit diagnostics arrays of the parameter run.

        Returns:
            Mapping from output name to the written path.
        """
        self.logger.section("[Stage 4] Writing the dataset")

        FileIO.ensure_dirs(self.config.data_directory, self.config.metadata_directory)

        self.writer.write_arrays(arrays)
        state_path  = self.writer.write_state()
        outputs     = self.writer.write_parameters(parameters, diagnostics)
        layout_path = self.writer.write_layout()

        return {"dataset_json": layout_path, "config_state": state_path, **outputs}

    def run(self) -> dict:
        """Runs every stage and writes the dataset.

        Returns:
            Mapping from output name to the written path.
        """
        amps, mus, sigmas = self._stage_parameters()

        cube, primary, secondaries, derived, z = self._stage_scene(amps, mus, sigmas)

        parameters, diagnostics = self._stage_parameter_run(cube, amps, mus, sigmas)

        arrays = {
            "primary"     : primary,
            "secondaries" : secondaries,
            "derived"     : derived,
            "aux_full"    : z,
            "cube_full"   : cube,
        }

        outputs = self._stage_write(arrays, parameters, diagnostics)

        self.logger.section("[Synthetic Dataset Complete]")
        self.logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

        return outputs


class SyntheticDatasetSession:
    """One generation of a synthetic dataset with its own log.

    Attributes:
        config  : Synthesis configuration describing the scene and the output directory.
        log_dir : Directory the session log is written into.
    """

    def __init__(self, config: SynthesisConfig, log_dir: Path) -> None:
        """Binds the session to a synthesis configuration and a log directory.

        Args:
            config  : Synthesis configuration describing the scene and the output directory.
            log_dir : Directory the session log is written into.
        """
        self.config  = config
        self.log_dir = Path(log_dir)

    def execute(self) -> dict:
        """Refuses a finished dataset directory, opens the session log and runs the pipeline.

        The directory is checked before the log is opened, and the log closes
        with the exception, its traceback and an ABORTED banner when a stage
        raises.

        Returns:
            The output mapping returned by ``SyntheticDatasetPipeline.run``.

        Raises:
            FileExistsError : If the output directory already holds a finished dataset.
        """
        guard     = DatasetDirectoryGuard(self.config.output_directory)
        leftovers = guard.leftovers()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        with Logger(log_dir=str(self.log_dir), name="generate_dataset", level="INFO") as logger:
            guard.warn(leftovers, logger)

            return SyntheticDatasetPipeline(self.config, logger=logger).run()
