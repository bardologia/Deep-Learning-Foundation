from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import jax

from tools.monitoring.logger import Logger

from .kernels import PmapSigmaAdamKernel, SigmaAdamKernel


class BestKSelector:
    def __init__(self, k_max : int, logger : Logger) -> None:
        self.k_max  = k_max
        self.logger = logger

    def score(self, gpu_results : Dict[int, tuple], batch_tag : str = "") -> np.ndarray:
        tag = f"{batch_tag} | " if batch_tag else ""

        self.logger.section(f"[{tag}Phase 3 — Best-K Selection]")

        mse_all = np.stack([gpu_results[K][3] for K in range(1, self.k_max + 1)], axis=1).astype(np.float64)

        return mse_all

    def select(
        self,
        gpu_results   : Dict[int, tuple],
        mse_all       : np.ndarray,
        scale_all     : np.ndarray,
        lambda_k      : float,
        n_params_out  : int,
        batch_tag     : str = "",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        N_act = mse_all.shape[0]
        tag   = f"{batch_tag} | " if batch_tag else ""

        penalised_all = mse_all + lambda_k * np.arange(1, self.k_max + 1, dtype=np.float64)[None, :]
        best_K_idx    = penalised_all.argmin(axis=1)

        best_params = np.zeros((N_act, n_params_out), dtype=np.float32)
        for K in range(1, self.k_max + 1):
            mask = best_K_idx == (K - 1)

            if not mask.any():
                continue

            amps_norm, mus, final_sigs, _ = gpu_results[K]
            idx      = np.where(mask)[0]
            amps_out = amps_norm[idx] * scale_all[idx, None]

            best_params[np.ix_(idx, list(range(0, K * 3, 3)))] = amps_out
            best_params[np.ix_(idx, list(range(1, K * 3, 3)))] = mus       [idx]
            best_params[np.ix_(idx, list(range(2, K * 3, 3)))] = final_sigs[idx]

        k_dist = {K: int((best_K_idx == K - 1).sum()) for K in range(1, self.k_max + 1)}
        self.logger.subsection(f"{tag}Best-K at lambda_k={lambda_k:g}")
        self.logger.subsection(f"Best-K dist     : {k_dist}")
        self.logger.subsection(f"Mean MSE (best) : {float(mse_all[np.arange(N_act), best_K_idx].mean()):.5f}")

        return best_params, mse_all.astype(np.float32), penalised_all.astype(np.float32), best_K_idx.astype(np.int16)


class KernelBackendSelector:
    def select(self) -> Tuple[object, int, str, list]:
        gpu_devices    = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
        active_devices = gpu_devices if gpu_devices else jax.devices()

        if len(active_devices) > 1:
            kernel    = PmapSigmaAdamKernel(active_devices)
            n_devices = len(active_devices)
            backend   = f"pmap  ({n_devices} GPUs)"
        else:
            kernel    = SigmaAdamKernel()
            n_devices = 1
            backend   = "jit  (1 GPU)"

        return kernel, n_devices, backend, active_devices
