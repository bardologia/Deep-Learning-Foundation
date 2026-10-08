"""Writes a synthetic dataset in the on-disk layout the training and inference pipelines read.

Holds the guard that refuses to generate into a directory already holding a
finished dataset and the writer of the arrays, the configuration state, the
parameter run and the dataset layout, which is written last and marks a
finished dataset.
"""

from __future__ import annotations

import dataclasses

from datetime import datetime
from pathlib  import Path

import numpy as np

from configuration.synthesis import SynthesisConfig
from tools.data.io           import FileIO
from tools.monitoring.logger import Logger
from tools.runtime.git_state import GitState


class DatasetDirectoryGuard:
    """Refuses to generate into a directory that already holds a finished dataset.

    A directory counts as a finished dataset when it carries the dataset layout
    or parameter runs under params/. Each of these means later stages have
    already read, or been fitted to, the arrays a rerun would replace in place,
    and nothing downstream could tell the old and the new arrays apart
    afterwards. A directory left behind by an unfinished run carries neither
    and is reused with a warning.

    Attributes:
        LAYOUT_RELPATH : Dataset layout the writer saves last, relative to the dataset directory.
        PARAMS_DIRNAME : Directory holding the dataset's parameter runs.
        output_dir     : Dataset directory the guard inspects.
    """

    LAYOUT_RELPATH = Path("data") / "dataset.json"
    PARAMS_DIRNAME = "params"

    def __init__(self, output_dir: Path) -> None:
        """Binds the guard to a dataset directory.

        Args:
            output_dir : Dataset directory a generation is about to write into.
        """
        self.output_dir = Path(output_dir)

    def parameter_runs(self) -> list[str]:
        """Returns the names of the parameter runs held under params/, sorted."""
        params_dir = self.output_dir / self.PARAMS_DIRNAME

        if not params_dir.is_dir():
            return []

        return sorted(child.name for child in params_dir.iterdir() if child.is_dir())

    def leftovers(self) -> list[str]:
        """Refuses a finished dataset directory and returns what an unfinished run left in it.

        Called before a session creates anything in the directory, so the
        returned names are an earlier run's, never the session's own.

        Returns:
            Sorted names of the directory's entries; empty for an absent or empty directory.

        Raises:
            FileExistsError : If the directory holds a dataset layout or parameter runs.
        """
        if not self.output_dir.exists():
            return []

        markers = [str(self.LAYOUT_RELPATH)] if (self.output_dir / self.LAYOUT_RELPATH).is_file() else []
        runs    = [f"{self.PARAMS_DIRNAME}/{run}" for run in self.parameter_runs()]

        if markers or runs:
            held = ", ".join(markers + runs)
            raise FileExistsError(f"Dataset directory {self.output_dir} already holds a finished dataset ({held}); generation refuses to overwrite it because its parameter runs and training runs were built on the arrays in it. Choose another output_directory or remove the directory by hand.")

        return sorted(child.name for child in self.output_dir.iterdir())

    def warn(self, contents: list[str], logger: Logger) -> None:
        """Warns that an unfinished run's leftovers are overwritten; silent when there are none.

        Args:
            contents : Entries the directory held before the session started, as leftovers returns them.
            logger   : Logger receiving the warning.
        """
        if contents:
            logger.warning(f"Dataset directory {self.output_dir} exists without a finished dataset (contents: {contents}); the leftovers of an unfinished run are overwritten")


class DatasetWriter:
    """Saves the arrays, configuration state, parameter run and layout of a synthetic dataset.

    Attributes:
        ARTIFACTS : Array name to the file it is saved under in data/.
        SOURCE    : Metadata ``source`` value marking a parameter run written by the generator.
        config    : Synthesis configuration naming the output directories.
        logger    : Logger reporting every written file.
    """

    ARTIFACTS = {
        "primary"     : "primary.npy",
        "secondaries" : "secondaries.npy",
        "derived"     : "derived.npy",
        "aux_full"    : "aux_full.npy",
        "cube_full"   : "cube_full.npy",
    }

    SOURCE = "synthesis"

    def __init__(self, config: SynthesisConfig, logger: Logger) -> None:
        """Binds the writer to a synthesis configuration.

        Args:
            config : Synthesis configuration naming the output directories.
            logger : Logger reporting every written file.
        """
        self.config = config
        self.logger = logger

    def _save_array(self, name: str, array: np.ndarray) -> Path:
        """Saves one array under its artifact name and logs its shape, dtype and size."""
        path = self.config.data_directory / self.ARTIFACTS[name]
        np.save(str(path), np.ascontiguousarray(array), allow_pickle=False)

        self.logger.subsection(f"-> {path}  {array.shape}  {array.dtype}  ({array.nbytes / 1e6:.1f} MB)")
        return path

    def write_arrays(self, arrays: dict) -> None:
        """Saves every artifact array into data/.

        Args:
            arrays : Mapping holding one array per ``ARTIFACTS`` key.
        """
        FileIO.ensure_dir(self.config.data_directory)

        for name in self.ARTIFACTS:
            self._save_array(name, arrays[name])

    def write_state(self, provenance: dict) -> Path:
        """Writes meta/config_state.json with the crop, the cube z range, the synthesis settings and the run's provenance.

        Args:
            provenance : Provenance block of the generating process, written under the 'provenance' key.

        Returns:
            Path of the written configuration state file.
        """
        crop    = self.config.global_crop
        payload = {
            "crop" : {
                "row_start" : int(crop[0]),
                "row_end"   : int(crop[1]),
                "col_start" : int(crop[2]),
                "col_end"   : int(crop[3]),
            },
            "cube_config" : {
                "z_range" : list(self.config.z_range),
                "n_z"     : int(self.config.profile_length),
            },
            "synthesis_config" : {key: str(value) if isinstance(value, Path) else value for key, value in dataclasses.asdict(self.config).items()},
            "provenance"       : provenance,
        }

        return FileIO.save_json(payload, self.config.metadata_directory / "config_state.json")

    def write_parameters(self, parameters: np.ndarray, diagnostics: dict) -> dict:
        """Saves the parameter run: the parameter stack, its diagnostics and its metadata.

        Args:
            parameters  : Parameter stack of shape (3 * k_max, rows, cols).
            diagnostics : Fit diagnostics arrays saved into fit_diagnostics.npz.

        Returns:
            Mapping with the parameter, diagnostics and metadata paths.
        """
        directory = self.config.parameters_directory
        FileIO.ensure_dir(directory)

        parameters_path  = directory / "parameters.npy"
        diagnostics_path = directory / "fit_diagnostics.npz"

        np.save(str(parameters_path), np.ascontiguousarray(parameters), allow_pickle=False)
        np.savez(str(diagnostics_path), **diagnostics)

        meta_path = FileIO.save_json(self._parameter_meta(parameters_path, diagnostics_path), directory / "param_extraction_meta.json")

        self.logger.subsection(f"-> {parameters_path}  {parameters.shape}  {parameters.dtype}")
        self.logger.subsection(f"-> {diagnostics_path}")
        self.logger.subsection(f"-> {meta_path}")

        return {"parameters": parameters_path, "diagnostics": diagnostics_path, "metadata": meta_path}

    def _parameter_meta(self, parameters_path: Path, diagnostics_path: Path) -> dict:
        """Returns the parameter run metadata in the form a fitted run records it, carrying the generating commit."""
        config = self.config

        return {
            "timestamp"           : datetime.now().isoformat(timespec="seconds"),
            "git_commit"          : GitState.commit(),
            "source"              : self.SOURCE,
            "processed_data_path" : str(config.output_directory),
            "source_cube"         : str(config.data_directory / self.ARTIFACTS["cube_full"]),
            "output_directory"    : str(config.parameters_directory),
            "parameters_npy"      : parameters_path.name,
            "diagnostics_npz"     : diagnostics_path.name,
            "z_range"             : list(config.z_range),
            "k_max"               : int(config.k_max),
            "lambda_k"            : float(config.lambda_k),
            "activity_threshold"  : float(config.activity_threshold),
            "threshold_factor"    : float(config.threshold_factor),
            "truncation_index"    : int(config.truncation_index),
        }

    def write_layout(self, provenance: dict) -> Path:
        """Writes data/dataset.json, the layout downstream pipelines read the dataset through and the mark of a finished dataset.

        Args:
            provenance : Provenance block of the generating process, written under the 'provenance' key.

        Returns:
            Path of the written layout file.
        """
        payload = {
            "global_crop"        : list(self.config.global_crop),
            "dataset_type"       : self.config.dataset_type,
            "cube_tag"           : self.config.cube_tag,
            "parameter_tag"      : self.config.parameter_tag,
            "max_amplitude_clip" : self.config.max_amplitude_clip,
            "channel_labels"     : self.config.channel_labels,
            "artifacts"          : dict(self.ARTIFACTS),
            "provenance"         : provenance,
        }

        return FileIO.save_json(payload, self.config.data_directory / "dataset.json")
