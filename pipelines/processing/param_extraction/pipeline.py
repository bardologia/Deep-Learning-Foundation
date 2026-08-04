from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, Tuple

import numpy as np

from configuration.param_extraction              import ExtractionConfig
from pipelines.processing.param_extraction.io    import ExtractionMetadataManager, ParameterIO
from pipelines.processing.param_extraction.queue import ExtractionGroup
from tools.data.gaussians                        import GaussianSlotSorter
from tools.monitoring.logger                     import Logger


class ParameterExtractor:
    def __init__(
        self,
        logger               : Logger,
        modes                : list,
        lambda_values        : list,
        k_max                : int         = 5,
        threshold_factor     : float       = 0.25,
        truncation_index     : int         = 170,
        prominence_frac      : float       = 0.05,
        sigma_init_divisor   : float       = 1.0,
        activity_threshold   : float       = 1e-3,
        range_batch_size     : int         = 256,
        adam_steps           : int         = 800,
        adam_lr              : float       = 1e-2,
        adam_b1              : float       = 0.9,
        adam_b2              : float       = 0.999,
        gpu_pixel_batch_size : int         = 8192,
        init_workers         : int | None  = None,
        peak_initialiser     : object | None = None,
        kernel_backend       : tuple | None  = None,
    ) -> None:
        from pipelines.processing.param_extraction.sigma import SigmaFittingExtractor

        self.logger             = logger
        self.k_max              = k_max
        self.activity_threshold = activity_threshold

        self._gpu_extractor = SigmaFittingExtractor(
            logger               = logger,
            modes                = modes,
            lambda_values        = lambda_values,
            k_max                = k_max,
            threshold_factor     = threshold_factor,
            truncation_index     = truncation_index,
            prominence_frac      = prominence_frac,
            sigma_init_divisor   = sigma_init_divisor,
            activity_threshold   = activity_threshold,
            range_batch_size     = range_batch_size,
            adam_steps           = adam_steps,
            adam_lr              = adam_lr,
            adam_b1              = adam_b1,
            adam_b2              = adam_b2,
            gpu_pixel_batch_size = gpu_pixel_batch_size,
            init_workers         = init_workers,
            peak_initialiser     = peak_initialiser,
            kernel_backend       = kernel_backend,
        )

        self.logger.section("[Parameter Extractor Initialized]")
        self.logger.subsection(f"Backend : JAX GPU (modes: {', '.join(modes)})")
        self.logger.subsection(f"Lambdas : {list(lambda_values)}")

    def run(self, tomogram_path: Path, height_range: Tuple[float, float]) -> Dict[tuple, Tuple[np.ndarray, dict]]:
        self.logger.section(f"[Extraction Start] Source: {tomogram_path.name}")

        results        = self._gpu_extractor.run(tomogram_path, height_range)
        sorted_results = {key: (GaussianSlotSorter.by_mean(params, self.k_max, self.activity_threshold), diagnostics) for key, (params, diagnostics) in results.items()}

        self.logger.subsection("[Extraction Complete]")
        return sorted_results


class ParamExtractionPipeline:
    def __init__(self, group: ExtractionGroup, logger: Logger | None = None, peak_initialiser: object | None = None, kernel_backend: tuple | None = None) -> None:
        self.group         = group
        self.shared        = group.shared
        self.tomogram_path = self.shared.discover_tomogram_path()
        self.height_range  = self.shared.discover_height_range()

        log_dir = Path(self.shared.processed_data_path) / "params" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or Logger(log_dir = str(log_dir), name = "param_extraction", level = "INFO")

        fit_cfg = self.shared.fit_settings.fit_config

        self.parameter_extractor = ParameterExtractor(
            logger               = self.logger,
            modes                = group.modes,
            lambda_values        = group.lambda_values,
            k_max                = group.k_max,
            threshold_factor     = fit_cfg.threshold_factor,
            truncation_index     = fit_cfg.truncation_index,
            prominence_frac      = fit_cfg.prominence_frac,
            sigma_init_divisor   = fit_cfg.sigma_init_divisor,
            activity_threshold   = fit_cfg.activity_threshold,
            range_batch_size     = self.shared.range_batch_size,
            adam_steps           = self.shared.adam_steps,
            adam_lr              = self.shared.adam_lr,
            adam_b1              = self.shared.adam_b1,
            adam_b2              = self.shared.adam_b2,
            gpu_pixel_batch_size = self.shared.gpu_pixel_batch_size,
            init_workers         = self.shared.parameter_workers,
            peak_initialiser     = peak_initialiser,
            kernel_backend       = kernel_backend,
        )
        self.parameter_io = ParameterIO(logger=self.logger)

        self.logger.section("[Param Extraction Pipeline Initialized]")
        self.logger.subsection(f"Source tomogram : {self.tomogram_path}")
        self.logger.subsection(f"Height range    : {self.height_range}")
        self.logger.subsection(f"k_max           : {group.k_max}")
        self.logger.subsection(f"Fit modes       : {group.modes}")
        self.logger.subsection(f"Lambda values   : {group.lambda_values}")
        self.logger.subsection(f"Permutations    : {len(group.configs)}")

    def _stage_extract(self) -> Dict[tuple, Tuple[np.ndarray, dict]]:
        self.logger.subsection("Extracting multi-Gaussian parameters (shared load + init across modes and lambdas)")
        return self.parameter_extractor.run(tomogram_path = self.tomogram_path, height_range = self.height_range)

    def _stage_save(self, config: ExtractionConfig, parameters_array: np.ndarray, diagnostics: dict) -> dict[str, Path]:
        npy_path  = self.parameter_io.save_params(parameters_array, config.parameters_npy_path)
        diag_path = self.parameter_io.save_diagnostics(diagnostics, config.diagnostics_npz_path)
        meta_path = ExtractionMetadataManager(config, logger=self.logger).save_run_metadata(npy_path, diag_path, self.tomogram_path, self.height_range)

        return {
            "parameters_npy"   : npy_path,
            "diagnostics_npz"  : diag_path,
            "metadata"         : meta_path,
            "output_directory" : config.output_directory,
            "source_tomogram"  : self.tomogram_path,
        }

    def run(self) -> Dict[tuple, dict]:
        self.logger.section("[Param Extraction Pipeline Execution]")

        results = self._stage_extract()

        saved = {}
        for key, (parameters_array, diagnostics) in results.items():
            config     = self.group.configs[key]
            self.logger.section(f"[Saving] {config.output_subdir_name}")
            saved[key] = self._stage_save(config, parameters_array, diagnostics)

        del results
        gc.collect()

        self.logger.section("[Param Extraction Completed]")

        return saved
