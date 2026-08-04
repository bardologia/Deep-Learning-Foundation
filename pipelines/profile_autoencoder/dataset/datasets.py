from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from pipelines.profile_autoencoder.dataset.augmentation  import ProfileAugmenter
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer
from tools.data.gaussians                                import GaussianMixture
from tools.monitoring.logger                             import Logger


class ProfileDataset(Dataset):
    def __init__(
        self,
        param_arrays    : list[np.ndarray],
        x_axis          : np.ndarray,
        n_gaussians     : int,
        split_name      : str,
        amp_zero_thr    : float,
        pixel_subsample : float = 1.0,
        keep_empty_frac : float = 0.05,
        seed            : int   = 0,
        normalizer      : Optional[ProfileNormalizer] = None,
        augmenter       : Optional[ProfileAugmenter]  = None,
        logger          : Optional[Logger] = None,
    ) -> None:
        self.x_axis      = np.asarray(x_axis, dtype=np.float32)
        self.n_gaussians = int(n_gaussians)
        self.split_name  = split_name
        self.normalizer  = normalizer
        self.augmenter   = augmenter

        self._stack_parameters(param_arrays)
        self._select_pixels(amp_zero_thr, pixel_subsample, keep_empty_frac, seed)

        if logger is not None:
            self._log(logger)

    def _stack_parameters(self, param_arrays: list[np.ndarray]) -> None:
        amps_list, mus_list, sigs_list = [], [], []

        for params in param_arrays:
            params = np.asarray(params, dtype=np.float32)
            C      = params.shape[0]
            flat   = params.reshape(C, -1)

            amps_list.append(flat[0::3].T)
            mus_list.append(flat[1::3].T)
            sigs_list.append(flat[2::3].T)

        self.amps = np.concatenate(amps_list, axis=0)
        self.mus  = np.concatenate(mus_list,  axis=0)
        self.sigs = np.concatenate(sigs_list, axis=0)

    def _select_pixels(self, amp_zero_thr: float, pixel_subsample: float, keep_empty_frac: float, seed: int) -> None:
        active = (self.amps > amp_zero_thr).any(axis=1)
        rng    = np.random.default_rng(seed)

        active_idx = np.flatnonzero(active)
        empty_idx  = np.flatnonzero(~active)

        if len(active_idx) == 0:
            raise ValueError(f"Split '{self.split_name}' holds no pixel with an amplitude above amp_zero_thr={amp_zero_thr}; lower the threshold or pick a split region that carries scatterers.")

        if 0.0 < pixel_subsample < 1.0:
            keep       = max(1, int(round(len(active_idx) * pixel_subsample)))
            active_idx = rng.choice(active_idx, size=keep, replace=False)

        if keep_empty_frac > 0.0 and len(empty_idx) > 0:
            keep      = max(1, int(round(len(empty_idx) * keep_empty_frac)))
            empty_idx = rng.choice(empty_idx, size=min(keep, len(empty_idx)), replace=False)
        else:
            empty_idx = np.empty(0, dtype=np.int64)

        self.n_active = int(active.sum())
        self.index    = np.concatenate([active_idx, empty_idx]).astype(np.int64)
        rng.shuffle(self.index)

    def _log(self, logger: Logger) -> None:
        logger.section(f"[ProfileDataset: {self.split_name}]")
        logger.kv_table({
            "Total pixels"   : int(self.amps.shape[0]),
            "Active pixels"  : self.n_active,
            "Kept pixels"    : int(self.index.shape[0]),
            "Profile length" : int(self.x_axis.shape[0]),
            "Gaussians"      : self.n_gaussians,
        })

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        idx   = self.index[i]
        curve = GaussianMixture.evaluate_batch(
            self.x_axis,
            self.amps[idx:idx + 1],
            self.mus[idx:idx + 1],
            self.sigs[idx:idx + 1],
        )[0].astype(np.float32)

        if self.augmenter is not None and self.split_name == "train":
            curve = self.augmenter(curve)

        if self.normalizer is not None:
            curve = self.normalizer.normalize(curve)

        if self.augmenter is not None and self.split_name == "train":
            curve = self.augmenter.add_noise(curve)

        return curve
