from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.param_extraction import ExtractionConfig, FitConfig, FitMode, FitSettings


@dataclass
class ExtractionGroup:
    processed_data_path : Path
    k_max               : int
    modes               : list
    lambda_values       : list
    configs             : dict

    @property
    def shared(self) -> ExtractionConfig:
        return next(iter(self.configs.values()))


class ExtractionPlanResolver:
    def __init__(self, entry_config, dataset_dirs: list[Path]) -> None:
        self.entry_config = entry_config
        self.dataset_dirs = dataset_dirs

    def _validate(self) -> None:
        for name in ("fit_k_values", "fit_lambda_values", "fit_modes"):
            value = getattr(self.entry_config, name)
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{name} must be a non-empty list, got {value!r}")

        permutations = len(self.dataset_dirs) * len(self.entry_config.fit_k_values) * len(self.entry_config.fit_lambda_values) * len(self.entry_config.fit_modes)
        if self.entry_config.output_suffix and permutations > 1:
            raise ValueError(f"output_suffix is a fixed name but the sweep expands to {permutations} permutations that would all collide on it; leave output_suffix unset so each permutation gets its auto-encoded name")

    def _build_plan(self, processed_data_path: Path, k_max, lambda_k, mode: str) -> ExtractionConfig:
        fit_sigma, fit_amplitude, fit_mean = FitMode.free_flags(mode)

        fit_config = FitConfig(
            threshold_factor   = self.entry_config.fit_threshold_factor,
            truncation_index   = self.entry_config.fit_truncation_index,
            k_max              = int(k_max),
            lambda_k           = float(lambda_k),
            prominence_frac    = self.entry_config.fit_prominence_frac,
            sigma_init_divisor = self.entry_config.fit_sigma_init_divisor,
            activity_threshold = self.entry_config.fit_activity_threshold,
            fit_sigma          = fit_sigma,
            fit_amplitude      = fit_amplitude,
            fit_mean           = fit_mean,
        )

        return ExtractionConfig(
            processed_data_path = processed_data_path,
            pyrat_directory     = self.entry_config.pyrat_directory,

            output_prefix = self.entry_config.output_prefix,
            output_suffix = self.entry_config.output_suffix,

            height_range = self.entry_config.height_range,

            fit_settings = FitSettings(fit_config=fit_config),

            range_batch_size     = self.entry_config.range_batch_size,
            gpu_pixel_batch_size = self.entry_config.gpu_pixel_batch_size,
            adam_steps           = self.entry_config.adam_steps,
            adam_lr              = self.entry_config.adam_lr,
            parameter_workers    = self.entry_config.parameter_workers,
        )

    def _build_group(self, processed_data_path: Path, k_max) -> ExtractionGroup:
        configs = {}
        for lambda_k in self.entry_config.fit_lambda_values:
            for mode in self.entry_config.fit_modes:
                configs[(mode, float(lambda_k))] = self._build_plan(processed_data_path, k_max, lambda_k, mode)

        return ExtractionGroup(
            processed_data_path = Path(processed_data_path),
            k_max               = int(k_max),
            modes               = list(self.entry_config.fit_modes),
            lambda_values       = [float(lambda_k) for lambda_k in self.entry_config.fit_lambda_values],
            configs             = configs,
        )

    def resolve(self) -> list[ExtractionGroup]:
        self._validate()

        groups = []
        for processed_data_path in self.dataset_dirs:
            for k_max in self.entry_config.fit_k_values:
                groups.append(self._build_group(processed_data_path, k_max))

        return groups
