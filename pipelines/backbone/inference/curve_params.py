
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipelines.processing.param_extraction.sigma.initialiser import PeakInitialiser
from tools.data.gaussians                                    import GaussianActivity, GaussianReconstructor
from tools.metrics.scoring                                   import R2


@dataclass
class CurveExtraction:

    params : np.ndarray
    fit_r2 : np.ndarray


class CurveParamExtractor:

    FWHM_PER_SIGMA = 2.3548200450309493
    RIDGE          = 1e-8

    def __init__(self, z_axis: np.ndarray, n_gaussians: int, activity_threshold: float, prominence_frac: float = 0.05, chunk_pixels: int = 8192, n_workers: int = 1) -> None:
        self.z_axis             = np.asarray(z_axis, dtype=np.float32)
        self.n_gaussians        = int(n_gaussians)
        self.prominence_frac    = float(prominence_frac)
        self.activity_threshold = float(activity_threshold)
        self.chunk_pixels       = int(chunk_pixels)
        self.n_workers          = max(1, int(n_workers))

        if self.z_axis.ndim != 1 or self.z_axis.size < 3:
            raise ValueError(f"CurveParamExtractor needs an z axis of at least 3 samples, got shape {self.z_axis.shape}.")

        self.dx        = float(self.z_axis[1] - self.z_axis[0])
        self.sigma_min = abs(self.dx) * 0.5
        self.sigma_max = abs(float(self.z_axis[-1] - self.z_axis[0]))

    def _flatten(self, curves: np.ndarray) -> np.ndarray:
        n_z = curves.shape[0]
        if n_z != self.z_axis.size:
            raise ValueError(f"Curve cube has {n_z} z bins but the extractor axis holds {self.z_axis.size}; they describe different z grids.")

        return curves.reshape(n_z, -1).T.astype(np.float32, copy=False)

    def _peak_locations(self, profiles: np.ndarray) -> np.ndarray:
        initialiser = PeakInitialiser(n_workers=self.n_workers)

        try:
            _amps, mus, _sigs = initialiser.run(profiles, self.z_axis, self.n_gaussians, prominence_frac=self.prominence_frac)
        finally:
            initialiser.close()

        return mus.astype(np.float32)

    def _peak_indices(self, mus: np.ndarray) -> np.ndarray:
        positions = (mus - self.z_axis[0]) / self.dx
        return np.clip(np.rint(positions), 0, self.z_axis.size - 1).astype(np.int64)

    def _half_widths(self, profiles: np.ndarray, peak_idx: np.ndarray) -> np.ndarray:
        n_pixels, n_z = profiles.shape
        columns          = np.arange(n_z)[None, :]
        sigmas           = np.empty((n_pixels, self.n_gaussians), dtype=np.float32)

        for k in range(self.n_gaussians):
            centre = peak_idx[:, k:k + 1]
            height = np.take_along_axis(profiles, centre, axis=1)
            above  = profiles >= (0.5 * height)

            right = np.where(~above & (columns > centre), columns, n_z).min(axis=1)
            left  = np.where(~above & (columns < centre), columns, -1).max(axis=1)

            fwhm         = (right - left - 1).astype(np.float32) * abs(self.dx)
            sigmas[:, k] = fwhm / self.FWHM_PER_SIGMA

        return np.clip(sigmas, self.sigma_min, self.sigma_max)

    def _design_matrix(self, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        offset = self.z_axis[None, :, None] - mus[:, None, :]
        scale  = np.maximum(sigmas, self.sigma_min)[:, None, :]

        return np.exp(np.clip(-0.5 * (offset / scale) ** 2, -100.0, 0.0)).astype(np.float32)

    def _amplitudes(self, profiles: np.ndarray, design: np.ndarray) -> np.ndarray:
        normal = np.matmul(design.transpose(0, 2, 1), design)
        target = np.matmul(design.transpose(0, 2, 1), profiles[:, :, None].astype(np.float64))

        ridge   = self.RIDGE * np.trace(normal, axis1=1, axis2=2).reshape(-1, 1, 1) + self.RIDGE
        regular = normal + ridge * np.eye(self.n_gaussians, dtype=np.float64)[None, :, :]

        return np.clip(np.linalg.solve(regular, target)[:, :, 0], 0.0, None).astype(np.float32)

    def _order_by_position(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> tuple:
        order = np.argsort(mus, axis=1, kind="stable")
        rows  = np.arange(mus.shape[0])[:, None]

        return amps[rows, order], mus[rows, order], sigmas[rows, order]

    def _interleave(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        stacked = np.stack([amps, mus, sigmas], axis=2)
        return stacked.reshape(amps.shape[0], self.n_gaussians * 3)

    def _chunk(self, profiles: np.ndarray, mus: np.ndarray) -> tuple:
        peak_idx = self._peak_indices(mus)
        sigmas   = self._half_widths(profiles, peak_idx)
        design   = self._design_matrix(mus, sigmas).astype(np.float64)
        amps     = self._amplitudes(profiles, design)

        amps     = np.where(GaussianActivity.is_active(amps, self.activity_threshold), amps, 0.0).astype(np.float32)
        rendered = np.matmul(design, amps[:, :, None].astype(np.float64))[:, :, 0].astype(np.float32)
        parts    = self._order_by_position(amps, mus, sigmas)

        return self._interleave(*parts), R2.pixel_map(rendered.T, profiles.T, axis=0)

    def _reshape_cube(self, flat_params: np.ndarray, spatial: tuple) -> np.ndarray:
        return flat_params.T.reshape(self.n_gaussians * 3, *spatial).astype(np.float32)

    def render(self, params: np.ndarray) -> np.ndarray:
        height, width = params.shape[1:]
        gauss         = params.reshape(1, self.n_gaussians, 3, height, width).astype(np.float32)

        return GaussianReconstructor.reconstruct_batch(gauss, self.z_axis.reshape(1, 1, -1, 1, 1))[0]

    def run(self, curves: np.ndarray) -> CurveExtraction:
        spatial  = curves.shape[1:]
        profiles = self._flatten(curves)
        mus      = self._peak_locations(profiles)

        params_chunks : list = []
        r2_chunks     : list = []

        for start in range(0, profiles.shape[0], self.chunk_pixels):
            stop            = start + self.chunk_pixels
            chunk_params, chunk_r2 = self._chunk(profiles[start:stop], mus[start:stop])

            params_chunks.append(chunk_params)
            r2_chunks.append(chunk_r2)

        params = self._reshape_cube(np.concatenate(params_chunks, axis=0), spatial)
        fit_r2 = np.concatenate(r2_chunks, axis=0).reshape(spatial).astype(np.float32)

        return CurveExtraction(params=params, fit_r2=fit_r2)
