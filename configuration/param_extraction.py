"""Configuration for fitting Gaussian mixtures to volumetric z profiles.

Covers the fit hyperparameters and Adam settings of the extraction itself, the
sweep entry point that runs it over a dataset tree, the analysis entry point
that reads the resulting parameter runs, and the injection of externally
produced parameter cubes into the same layout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional, Tuple


@dataclass
class FitConfig:
    """Hyperparameters of the per-pixel Gaussian mixture fit.

    Attributes:
        threshold_factor   : Fraction of the profile maximum below which samples are ignored.
        truncation_index   : Z bin above which the profile is truncated.
        k_max              : Maximum number of Gaussian components fitted per pixel.
        lambda_k           : Penalty weight discouraging additional components.
        prominence_frac    : Peak prominence, as a fraction of the profile range, required to seed a component.
        sigma_init_divisor : Divisor turning a peak's half-width into the initial sigma in metres.
        activity_threshold : Profile peak at or below which a pixel is skipped by the fit and keeps zero parameters; recorded in the run metadata as the amplitude above which a slot counts as active in every later stage.
        fit_sigma          : Whether component widths are free parameters.
        fit_amplitude      : Whether component amplitudes are free parameters.
        fit_mean           : Whether component means are free parameters.
    """

    threshold_factor   : float = 0.25
    truncation_index   : int   = 170
    k_max              : int   = 5
    lambda_k           : float = 1e-2
    prominence_frac    : float = 0.05
    sigma_init_divisor : float = 4.0
    activity_threshold : float = 1e-3
    fit_sigma          : bool  = True
    fit_amplitude      : bool  = False
    fit_mean           : bool  = False


class FitMode:
    """Parser for the underscore-joined fit mode naming the free components.

    Attributes:
        COMPONENTS : Component names in the order the returned flags follow.
    """

    COMPONENTS = ("sigma", "amp", "mu")

    @classmethod
    def free_flags(cls, mode: str) -> Tuple[bool, bool, bool]:
        """Returns which of sigma, amplitude and mean are free in a fit mode.

        Args:
            mode : Distinct component names joined with "_", e.g. "sigma_amp_mu".

        Returns:
            One boolean per entry of `COMPONENTS`, True when that component is free.

        Raises:
            ValueError : If the mode is empty, repeats a component, or names an unknown one.
        """
        parts = mode.split("_") if mode else []

        if not parts or len(set(parts)) != len(parts) or any(part not in cls.COMPONENTS for part in parts):
            raise ValueError(f"Unknown fit mode '{mode}'. A mode joins distinct components from {', '.join(cls.COMPONENTS)} with '_', e.g. 'sigma', 'amp_mu', 'sigma_amp_mu'")

        return tuple(component in parts for component in cls.COMPONENTS)


class ChannelOrder:
    """Parser for the component ordering of an external parameter cube.

    Attributes:
        COMPONENTS : Component names in the canonical internal channel order.
    """

    COMPONENTS = ("amp", "mu", "sigma")

    @classmethod
    def positions(cls, order: str) -> Tuple[int, int, int]:
        """Returns where each canonical component sits in a foreign channel order.

        Args:
            order : Permutation of the component names joined with "_", e.g. "mu_sigma_amp".

        Returns:
            The source channel index of amplitude, mean and sigma, in that order.

        Raises:
            ValueError : If the string is not a permutation of `COMPONENTS`.
        """
        parts = order.split("_") if order else []

        if sorted(parts) != sorted(cls.COMPONENTS):
            raise ValueError(f"Unknown channel order '{order}'. An order is a permutation of {', '.join(cls.COMPONENTS)} joined with '_', e.g. 'mu_sigma_amp'")

        return tuple(parts.index(component) for component in cls.COMPONENTS)


@dataclass
class FitSettings:
    """Wrapper exposing the fit mode implied by the fit toggles.

    Attributes:
        fit_config : Hyperparameters of the per-pixel Gaussian mixture fit.
    """

    fit_config : FitConfig = field(default_factory=FitConfig)

    @property
    def free_parameters(self) -> Tuple[str, ...]:
        """Names of the free components, in canonical order.

        Raises:
            ValueError : If no component is free.
        """
        cfg   = self.fit_config
        flags = {"sigma": cfg.fit_sigma, "amp": cfg.fit_amplitude, "mu": cfg.fit_mean}
        names = tuple(name for name in FitMode.COMPONENTS if flags[name])

        if not names:
            raise ValueError("No free parameters: enable at least one of fit_sigma, fit_amplitude, fit_mean")

        return names

    @property
    def fitting_method(self) -> str:
        """Identifier of the fit, joining the free components with the optimizer."""
        return "_".join(self.free_parameters) + "_adam"


@dataclass
class ExtractionConfig:
    """Settings for one parameter-extraction run over a processed dataset.

    The output directory name encodes the fit hyperparameters so that different
    extractions of the same cube stay side by side.

    Attributes:
        processed_data_path  : Root of the processed dataset holding data/ and meta/.
        output_prefix        : Leading token of the parameter run directory name.
        output_suffix        : Explicit suffix, or None to derive one from the fit settings.
        z_range              : Z axis extent in metres, or None to read it from the dataset metadata.
        fit_settings         : Fit hyperparameters and the implied fit mode.
        parameter_workers    : CPU worker processes writing the parameter chunks.
        col_batch_size       : Number of cols loaded per batch.
        gpu_pixel_batch_size : Number of pixels fitted per GPU batch.
        adam_steps           : Adam iterations per pixel batch.
        adam_lr              : Adam learning rate.
        adam_b1              : Adam first-moment decay.
        adam_b2              : Adam second-moment decay.
    """

    processed_data_path : Path
    output_prefix       : str                           = "params"
    output_suffix       : Optional[str]                 = None
    z_range             : Optional[Tuple[float, float]] = None
    fit_settings        : FitSettings                   = field(default_factory=FitSettings)
    parameter_workers   : int                           = 20

    col_batch_size       : int   = 3500
    gpu_pixel_batch_size : int   = 24576
    adam_steps           : int   = 3000
    adam_lr              : float = 2e-1
    adam_b1              : float = 0.95
    adam_b2              : float = 0.999


    @property
    def data_directory(self) -> Path:
        """Directory holding the processed cube arrays."""
        return Path(self.processed_data_path) / "data"

    @property
    def metadata_directory(self) -> Path:
        """Directory holding the dataset metadata JSON files."""
        return Path(self.processed_data_path) / "meta"

    @property
    def output_suffix_value(self) -> str:
        """Suffix identifying the fit, explicit or derived from the fit settings."""
        if self.output_suffix:
            return self.output_suffix

        cfg     = self.fit_settings.fit_config
        lam_tag = f"{cfg.lambda_k:g}"
        div_tag = f"{cfg.sigma_init_divisor:g}"
        free    = "_".join(self.fit_settings.free_parameters)

        return f"k{cfg.k_max}_lam{lam_tag}_sig{div_tag}_{free}"

    @property
    def output_subdir_name(self) -> str:
        """Name of the parameter run directory under params/."""
        return f"{self.output_prefix}_{self.output_suffix_value}"

    @property
    def output_directory(self) -> Path:
        """Directory this extraction writes its parameter run into."""
        return Path(self.processed_data_path) / "params" / self.output_subdir_name

    @property
    def parameters_npy_path(self) -> Path:
        """Path of the fitted parameter cube written by this extraction."""
        return self.output_directory / "parameters.npy"

    @property
    def diagnostics_npz_path(self) -> Path:
        """Path of the per-pixel fit diagnostics archive."""
        return self.output_directory / "fit_diagnostics.npz"

    def discover_cube_path(self) -> Path:
        """Returns the path of the full cube to be fitted.

        Returns:
            Path of cube_full.npy inside the dataset data directory.

        Raises:
            FileNotFoundError : If the cube file is absent.
        """
        cube_path = self.data_directory / "cube_full.npy"

        if not cube_path.is_file():
            raise FileNotFoundError(f"No cube_full.npy in {self.data_directory}")

        return cube_path

    def discover_z_range(self) -> Tuple[float, float]:
        """Returns the z axis extent of the cube.

        The configured override wins; otherwise the z range is read from the
        dataset config_state.json.

        Returns:
            The (minimum, maximum) z in metres.

        Raises:
            FileNotFoundError : If no override is set and the metadata does not carry a cube z range.
        """
        if self.z_range is not None:
            return tuple(self.z_range)

        meta_dir   = self.metadata_directory
        state_path = meta_dir / "config_state.json"
        if state_path.is_file():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            hr = state["cube_config"]["z_range"]
            if isinstance(hr, (list, tuple)) and len(hr) == 2:
                return (float(hr[0]), float(hr[1]))

        raise FileNotFoundError(f"No z_range override given and no config_state.json with a cube z_range under {meta_dir}")


@dataclass
class ExtractParamsEntryConfig:
    """Sweep settings for extracting parameters across a tree of datasets.

    Each combination of k, lambda and fit mode becomes a separate extraction
    run, fanned out over the configured GPUs.

    Attributes:
        dataset_base_path      : Root directory scanned for processed datasets.
        dataset_filter         : Exact dataset directory names under the base path to select; empty selects every dataset.
        output_prefix          : Leading token of each parameter run directory name.
        output_suffix          : Explicit suffix, or None to derive one per combination.
        z_range                : Z axis extent in metres, or None to read it per dataset.
        overwrite              : Whether an existing parameter run under a permutation's directory name is deleted and refitted; off, the sweep is refused before any fitting.
        fit_k_values           : Maximum component counts swept.
        fit_lambda_values      : Component-count penalty weights swept.
        fit_modes              : Fit modes swept, each naming the free components.
        fit_sigma_init_divisor : Divisor turning a peak half-width into the initial sigma in metres.
        fit_threshold_factor   : Fraction of the profile maximum below which samples are ignored.
        fit_truncation_index   : Z bin above which profiles are truncated.
        fit_prominence_frac    : Peak prominence fraction required to seed a component.
        fit_activity_threshold : Profile peak at or below which a pixel is skipped by the fit, recorded in every trial's metadata as the activity threshold later stages read.
        adam_steps             : Adam iterations per pixel batch.
        adam_lr                : Adam learning rate.
        gpu_device_ids         : GPU device indices available to the extraction workers.
        col_batch_size         : Number of cols loaded per batch.
        gpu_pixel_batch_size   : Number of pixels fitted per GPU batch.
        parameter_workers      : CPU worker processes writing the parameter chunks.
    """

    dataset_base_path : Path = Path("datasets")
    dataset_filter    : list = field(default_factory=list)

    output_prefix : str          = "params"
    output_suffix : str | None   = None
    z_range       : tuple | None = None
    overwrite     : bool         = False

    fit_k_values           : list  = field(default_factory=lambda: [5])
    fit_lambda_values      : list  = field(default_factory=lambda: [1e-2])
    fit_modes              : list  = field(default_factory=lambda: ["sigma", "sigma_amp", "sigma_amp_mu"])
    fit_sigma_init_divisor : float = 4.0
    fit_threshold_factor   : float = 0.25
    fit_truncation_index   : int   = 170
    fit_prominence_frac    : float = 0.05
    fit_activity_threshold : float = 1e-3

    adam_steps : int   = 3000
    adam_lr    : float = 2e-1

    gpu_device_ids       : list = field(default_factory=lambda: [0])
    col_batch_size       : int  = 3500
    gpu_pixel_batch_size : int  = 24576
    parameter_workers    : int  = field(default_factory=lambda: min(64, os.cpu_count() or 16))


@dataclass
class ParamExtractionInferenceConfig:
    """Settings for analysing already-extracted parameter runs.

    Attributes:
        params_dir : Root directory holding the datasets whose parameter runs are analysed.
        run_tags   : Parameter run directory names to analyse; empty selects all.
        make_plots : Whether the analysis figures are rendered.
    """

    params_dir : Path = Path("datasets")
    run_tags   : list = field(default_factory=list)

    make_plots : bool = True
