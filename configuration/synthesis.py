from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Tuple


@dataclass
class SynthesisConfig:
    output_directory : Path = Path("datasets/synthetic_small")

    global_crop    : Tuple[int, int, int, int] = (1000, 1512, 500, 820)
    profile_length : int                       = 64
    z_range        : Tuple[float, float]       = (-20.0, 80.0)

    k_max           : int = 2
    params_run_name : str = "params_k2"

    primary_label    : str   = "CH00"
    secondary_labels : tuple = ("CH01", "CH02", "CH03", "CH04")

    field_smoothing_pixels : float = 6.0
    phase_smoothing_pixels : float = 14.0

    amplitude_range  : Tuple[float, float] = (0.6, 6.0)
    sigma_range      : Tuple[float, float] = (2.5, 9.0)
    slot_band_margin : float               = 0.12

    empty_pixel_fraction  : float = 0.05
    dropped_slot_fraction : float = 0.35

    sensitivity_width_fraction : float = 0.22
    channel_phase_rates        : tuple = (0.012, 0.024, 0.036, 0.048)

    cube_noise_fraction    : float = 0.03
    channel_noise_fraction : float = 0.01

    aux_range : Tuple[float, float] = (0.0, 30.0)

    lambda_k           : float = 0.01
    activity_threshold : float = 1e-3
    threshold_factor   : float = 0.0
    truncation_index   : int   = 170

    dataset_type       : str   = "synthetic"
    cube_tag           : str   = "synthetic"
    parameter_tag      : str   = "params"
    max_amplitude_clip : float = 4.0

    seed : int = 0

    @property
    def row_size(self) -> int:
        return int(self.global_crop[1] - self.global_crop[0])

    @property
    def col_size(self) -> int:
        return int(self.global_crop[3] - self.global_crop[2])

    @property
    def n_secondaries(self) -> int:
        return len(self.secondary_labels)

    @property
    def channel_labels(self) -> list:
        return [str(self.primary_label)] + [str(label) for label in self.secondary_labels]

    @property
    def data_directory(self) -> Path:
        return Path(self.output_directory) / "data"

    @property
    def metadata_directory(self) -> Path:
        return Path(self.output_directory) / "meta"

    @property
    def parameters_directory(self) -> Path:
        return Path(self.output_directory) / "params" / self.params_run_name

    def validate(self) -> None:
        if self.row_size <= 0 or self.col_size <= 0:
            raise ValueError(f"global_crop {self.global_crop} must describe a positive (row, col) extent; got row_size={self.row_size} and col_size={self.col_size}.")

        if self.profile_length < 4:
            raise ValueError(f"profile_length must sample the height axis with at least 4 bins; got {self.profile_length}.")

        if self.z_range[1] <= self.z_range[0]:
            raise ValueError(f"z_range must be increasing; got {self.z_range}.")

        if self.k_max < 1:
            raise ValueError(f"k_max must place at least one Gaussian per pixel; got {self.k_max}.")

        if len(self.channel_phase_rates) != self.n_secondaries:
            raise ValueError(f"channel_phase_rates lists {len(self.channel_phase_rates)} entries but secondary_labels lists {self.n_secondaries}; every secondary needs exactly one wavenumber.")

        if self.primary_label in self.secondary_labels:
            raise ValueError(f"primary_label {self.primary_label} also appears in secondary_labels {self.secondary_labels}; the reference pass is never a secondary.")

        if len(set(self.secondary_labels)) != self.n_secondaries:
            raise ValueError(f"secondary_labels contains duplicates: {self.secondary_labels}")

        span      = self.z_range[1] - self.z_range[0]
        bin_width = span / (self.profile_length - 1)

        if self.sigma_range[0] < bin_width:
            raise ValueError(f"sigma_range lower bound {self.sigma_range[0]} is finer than the height bin width {bin_width:.4f}; such a component cannot be resolved on the sampled axis.")

        if self.sigma_range[1] > span / 2.0:
            raise ValueError(f"sigma_range upper bound {self.sigma_range[1]} exceeds half the height span {span / 2.0}; such a component is wider than the axis it lives on.")


@dataclass
class GenerateDatasetEntryConfig:
    synthesis : SynthesisConfig = field(default_factory=SynthesisConfig)
    log_dir   : Path            = Path("logs")
