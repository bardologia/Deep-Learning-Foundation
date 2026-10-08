from __future__ import annotations

import numpy as np

from configuration.synthesis              import SynthesisConfig
from pipelines.processing.synthesis.scene import ProfileBuilder
from tools.data.gaussians                 import GaussianActivity, GaussianSlotSorter
from tools.data.preprocessing             import ProfilePreprocessor


class FitDiagnosticsBuilder:
    def __init__(self, config: SynthesisConfig, z_axis: np.ndarray) -> None:
        self.config   = config
        self.z_axis   = z_axis
        self.profiles = ProfileBuilder(z_axis)

    @staticmethod
    def _rank_by_amplitude(amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        order = np.argsort(-amps, axis=0, kind="stable")

        return (
            np.take_along_axis(amps,   order, axis=0),
            np.take_along_axis(mus,    order, axis=0),
            np.take_along_axis(sigmas, order, axis=0),
        )

    def observed_profiles(self, cube: np.ndarray) -> tuple:
        config    = self.config
        magnitude = np.abs(cube).astype(np.float32)
        observed  = ProfilePreprocessor.apply(magnitude, config.threshold_factor, config.truncation_index)

        peak   = observed.max(axis=0)
        active = GaussianActivity.is_active(peak, config.activity_threshold)
        scale  = np.where(active, peak, 1.0).astype(np.float32)

        return (observed / scale[None, :, :]).astype(np.float32), active, scale

    def build(self, cube: np.ndarray, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        config                                  = self.config
        observed_norm, active, scale            = self.observed_profiles(cube)
        ranked_amps, ranked_mus, ranked_sigmas  = self._rank_by_amplitude(amps, mus, sigmas)

        spatial     = active.shape
        mse_per_k   = np.full((config.k_max,) + spatial, np.nan, dtype=np.float32)
        running     = np.zeros_like(observed_norm)
        axis        = self.z_axis[:, None, None]

        for slot in range(config.k_max):
            self.profiles.accumulate(running, ranked_amps[slot], ranked_mus[slot], ranked_sigmas[slot], axis)

            residual         = running / scale[None, :, :] - observed_norm
            mse_per_k[slot]  = np.where(active, (residual ** 2).mean(axis=0), np.nan)

        penalty         = config.lambda_k * np.arange(1, config.k_max + 1, dtype=np.float32)
        penalised_per_k = (mse_per_k + penalty[:, None, None]).astype(np.float32)

        best_index = np.argmin(np.where(active[None, :, :], penalised_per_k, np.inf), axis=0)
        best_k_map = np.where(active, best_index + 1, 0).astype(np.int16)

        parameters = self._assemble(ranked_amps, ranked_mus, ranked_sigmas, best_k_map)

        diagnostics = {
            "mse_per_k"       : mse_per_k,
            "penalised_per_k" : penalised_per_k,
            "best_k_map"      : best_k_map,
            "lambda_k"        : np.float32(config.lambda_k),
        }

        return parameters, diagnostics

    def _assemble(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, best_k_map: np.ndarray) -> np.ndarray:
        config = self.config
        keep   = np.arange(config.k_max)[:, None, None] < best_k_map[None, :, :]
        stack  = np.empty((3 * config.k_max,) + best_k_map.shape, dtype=np.float32)

        stack[0::3] = np.where(keep, amps,   0.0)
        stack[1::3] = np.where(keep, mus,    0.0)
        stack[2::3] = np.where(keep, sigmas, 0.0)

        return GaussianSlotSorter.by_mean(stack, config.k_max, config.activity_threshold).astype(np.float32)
