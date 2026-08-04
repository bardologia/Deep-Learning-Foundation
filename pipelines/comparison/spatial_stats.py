from __future__ import annotations

import numpy as np


class SpatialDispersion:

    EPSILON = 1e-9

    @staticmethod
    def _blocks(field: np.ndarray, block: int) -> np.ndarray:
        rows = field.shape[0] // block
        cols = field.shape[1] // block

        cropped = field[:rows * block, :cols * block]
        tiled   = cropped.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)

        return tiled.reshape(rows * cols, block * block)

    @staticmethod
    def block_cv(field: np.ndarray, block: int) -> float:
        tiles = SpatialDispersion._blocks(field, block)

        mean = np.nanmean(tiles, axis=1)
        std  = np.nanstd(tiles, axis=1)

        valid = mean > SpatialDispersion.EPSILON
        cv    = np.where(valid, std / np.where(valid, mean, 1.0), np.nan)

        return float(np.nanmedian(cv))

    @staticmethod
    def block_std(field: np.ndarray, block: int) -> float:
        tiles = SpatialDispersion._blocks(field, block)
        return float(np.nanmedian(np.nanstd(tiles, axis=1)))

    @staticmethod
    def autocorr_length(field: np.ndarray, axis: int = 0, max_lines: int = 256) -> float:
        moved  = np.moveaxis(field, axis, 0)
        length = moved.shape[0]

        columns = moved.reshape(length, -1)
        count   = columns.shape[1]

        if count > max_lines:
            picks   = np.linspace(0, count - 1, max_lines).astype(np.int64)
            columns = columns[:, picks]

        centred = columns.astype(np.float64) - np.nanmean(columns, axis=0, keepdims=True)
        centred = np.nan_to_num(centred, nan=0.0)

        spectrum = np.fft.rfft(centred, n=2 * length, axis=0)
        acf      = np.fft.irfft(np.abs(spectrum) ** 2, axis=0)[:length]

        zero_lag = acf[0]
        valid    = zero_lag > SpatialDispersion.EPSILON
        acf      = acf[:, valid] / zero_lag[valid]

        if acf.shape[1] == 0:
            return float("nan")

        mean_acf = acf.mean(axis=1)
        below    = np.where(mean_acf < np.exp(-1.0))[0]

        return float(below[0]) if below.size > 0 else float(length)
