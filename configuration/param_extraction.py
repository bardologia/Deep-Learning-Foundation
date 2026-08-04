from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional, Tuple


@dataclass
class FitConfig:
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
    COMPONENTS = ("sigma", "amp", "mu")

    @classmethod
    def free_flags(cls, mode: str) -> Tuple[bool, bool, bool]:
        parts = mode.split("_") if mode else []

        if not parts or len(set(parts)) != len(parts) or any(part not in cls.COMPONENTS for part in parts):
            raise ValueError(f"Unknown fit mode '{mode}'. A mode joins distinct components from {', '.join(cls.COMPONENTS)} with '_', e.g. 'sigma', 'amp_mu', 'sigma_amp_mu'")

        return tuple(component in parts for component in cls.COMPONENTS)


class ChannelOrder:
    COMPONENTS = ("amp", "mu", "sigma")

    @classmethod
    def positions(cls, order: str) -> Tuple[int, int, int]:
        parts = order.split("_") if order else []

        if sorted(parts) != sorted(cls.COMPONENTS):
            raise ValueError(f"Unknown channel order '{order}'. An order is a permutation of {', '.join(cls.COMPONENTS)} joined with '_', e.g. 'mu_sigma_amp'")

        return tuple(parts.index(component) for component in cls.COMPONENTS)


@dataclass
class FitSettings:
    fit_config : FitConfig = field(default_factory=FitConfig)

    @property
    def free_parameters(self) -> Tuple[str, ...]:
        cfg   = self.fit_config
        flags = {"sigma": cfg.fit_sigma, "amp": cfg.fit_amplitude, "mu": cfg.fit_mean}
        names = tuple(name for name in FitMode.COMPONENTS if flags[name])

        if not names:
            raise ValueError("No free parameters: enable at least one of fit_sigma, fit_amplitude, fit_mean")

        return names

    @property
    def fitting_method(self) -> str:
        return "_".join(self.free_parameters) + "_adam"


@dataclass
class ExtractionConfig:
    processed_data_path : Path
    pyrat_directory     : Path                          = field(default_factory=lambda: Path("/ste/rnd/User/vice_vi/pyrat"))
    output_prefix       : str                           = "params"
    output_suffix       : Optional[str]                 = None
    height_range        : Optional[Tuple[float, float]] = None
    fit_settings        : FitSettings                   = field(default_factory=FitSettings)
    parameter_workers   : int                           = 20

    range_batch_size     : int   = 3500
    gpu_pixel_batch_size : int   = 24576
    adam_steps           : int   = 3000
    adam_lr              : float = 2e-1
    adam_b1              : float = 0.95
    adam_b2              : float = 0.999


    @property
    def data_directory(self) -> Path:
        return Path(self.processed_data_path) / "data"

    @property
    def metadata_directory(self) -> Path:
        return Path(self.processed_data_path) / "meta"

    @property
    def output_suffix_value(self) -> str:
        if self.output_suffix:
            return self.output_suffix

        cfg     = self.fit_settings.fit_config
        lam_tag = f"{cfg.lambda_k:g}"
        div_tag = f"{cfg.sigma_init_divisor:g}"
        free    = "_".join(self.fit_settings.free_parameters)

        return f"k{cfg.k_max}_lam{lam_tag}_sig{div_tag}_{free}"

    @property
    def output_subdir_name(self) -> str:
        return f"{self.output_prefix}_{self.output_suffix_value}"

    @property
    def output_directory(self) -> Path:
        return Path(self.processed_data_path) / "params" / self.output_subdir_name

    @property
    def parameters_npy_path(self) -> Path:
        return self.output_directory / "parameters.npy"

    @property
    def diagnostics_npz_path(self) -> Path:
        return self.output_directory / "fit_diagnostics.npz"

    def discover_tomogram_path(self) -> Path:
        tomogram_path = self.data_directory / "tomogram_full.npy"

        if not tomogram_path.is_file():
            raise FileNotFoundError(f"No tomogram_full.npy in {self.data_directory}")

        return tomogram_path

    def discover_height_range(self) -> Tuple[float, float]:
        if self.height_range is not None:
            return tuple(self.height_range)

        meta_dir   = self.metadata_directory
        state_path = meta_dir / "config_state.json"
        if state_path.is_file():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            hr = state["tomogram_config"]["height_range"]
            if isinstance(hr, (list, tuple)) and len(hr) == 2:
                return (float(hr[0]), float(hr[1]))

        raise FileNotFoundError(f"No height_range override given and no config_state.json with a tomogram height_range under {meta_dir}")


@dataclass
class ExtractParamsEntryConfig:
    dataset_base_path : Path = Path("/ste/rnd/User/vice_vi/Dataset")
    dataset_filter    : list = field(default_factory=list)
    pyrat_directory   : Path = Path("/ste/rnd/User/vice_vi/pyrat")

    output_prefix : str          = "params"
    output_suffix : str | None   = None
    height_range  : tuple | None = None

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

    gpu_device_ids       : list = field(default_factory=lambda: [0, 1, 2, 3])
    range_batch_size     : int  = 3500
    gpu_pixel_batch_size : int  = 24576
    parameter_workers    : int  = field(default_factory=lambda: min(64, os.cpu_count() or 16))


@dataclass
class ParamExtractionInferenceConfig:
    params_dir : Path = Path("/ste/rnd/User/vice_vi/Dataset")
    run_tags   : list = field(default_factory=list)

    make_plots : bool = True


@dataclass
class InjectExternalParamsEntryConfig:
    dataset_base_path : Path = Path("/ste/rnd/User/vice_vi/Dataset")
    dataset_filter    : list = field(default_factory=list)

    source_files   : list = field(default_factory=list)
    source_windows : list = field(default_factory=list)
    source_order   : str  = "mu_sigma_amp"

    output_prefix : str = "params"
    output_suffix : str = "external"
    origin_note   : str = ""

    k_slots            : int   = 0
    activity_threshold : float = 1e-3
    overwrite          : bool  = False
