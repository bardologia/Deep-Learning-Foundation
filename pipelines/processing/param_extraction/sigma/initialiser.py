from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools          import partial
from typing             import Tuple

import numpy as np
from scipy.signal import find_peaks


class PeakInitialiser:
    def __init__(self, n_workers : int = 1) -> None:
        self.n_workers = n_workers
        self._pool     = ProcessPoolExecutor(max_workers=n_workers)
        list(self._pool.map(abs, range(n_workers)))

    @staticmethod
    def _prominence_worker(raw_chunk : np.ndarray, height_axis : np.ndarray, K : int, sigma_guess : float, min_dist : int, prominence_frac : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk_N, H = raw_chunk.shape
        amps = np.zeros((chunk_N, K), dtype=np.float32)
        mus  = np.zeros((chunk_N, K), dtype=np.float32)
        sigs = np.full ((chunk_N, K), sigma_guess, dtype=np.float32)

        for n in range(chunk_N):
            raw  = raw_chunk[n]
            pmax = raw.max()
            if pmax < 1e-10:
                idxs = np.linspace(0, H - 1, K, dtype=int)
            else:
                peaks, props = find_peaks(raw, prominence=pmax * prominence_frac, distance=min_dist)

                if len(peaks) > 0:
                    peaks = peaks[np.argsort(props["prominences"])[::-1]]

                if len(peaks) >= K:
                    idxs = peaks[:K]

                elif len(peaks) > 0:
                    residual = raw.copy()
                    extra    = []

                    for p in peaks:
                        lo = max(0, p - min_dist)
                        hi = min(H, p + min_dist + 1)
                        residual[lo:hi] = 0.0

                    for _ in range(K - len(peaks)):
                        ei = int(np.argmax(residual))
                        extra.append(ei)
                        lo = max(0, ei - min_dist)
                        hi = min(H, ei + min_dist + 1)
                        residual[lo:hi] = 0.0

                    idxs = np.concatenate([peaks, np.array(extra, dtype=int)])

                else:
                    idxs = np.linspace(0, H - 1, K, dtype=int)

            for g, idx in enumerate(idxs[:K]):
                amps[n, g] = max(float(raw[idx]), 1e-10)
                mus [n, g] = float(height_axis[idx])

        return amps, mus, sigs

    def run(self, prof_raw : np.ndarray, height_axis : np.ndarray, K : int, prominence_frac : float = 0.05, sigma_divisor : float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, H        = prof_raw.shape
        h_span      = float(height_axis[-1] - height_axis[0])
        dh          = float(height_axis[1] - height_axis[0])
        sigma_base  = max(2.0 * dh, h_span / (8.0 * K))
        sigma_guess = sigma_base / max(float(sigma_divisor), 1e-6)
        min_dist    = max(1, int(sigma_base / dh))
        raw         = prof_raw.astype(np.float32, copy=False)

        chunk_size = max(1, -(-N // (self.n_workers * 2)))
        chunks     = [raw[i:i + chunk_size] for i in range(0, N, chunk_size)]

        worker_fn  = partial(
            self._prominence_worker,
            height_axis     = height_axis,
            K               = K,
            sigma_guess     = sigma_guess,
            min_dist        = min_dist,
            prominence_frac = prominence_frac,
        )

        chunk_results = list(self._pool.map(worker_fn, chunks))

        amps = np.concatenate([r[0] for r in chunk_results], axis=0)
        mus  = np.concatenate([r[1] for r in chunk_results], axis=0)
        sigs = np.concatenate([r[2] for r in chunk_results], axis=0)

        return amps, mus, sigs

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
