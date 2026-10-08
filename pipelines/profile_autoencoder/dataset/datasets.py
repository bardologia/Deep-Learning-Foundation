"""Torch dataset that synthesises z profiles from Gaussian parameters.

Each pixel of the extracted parameter maps becomes one curve, evaluated on the
fly from its amplitude, mean and sigma triplets.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from pipelines.profile_autoencoder.dataset.augmentation  import ProfileAugmenter
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer
from tools.data.gaussians                                import GaussianActivity, GaussianMixture
from tools.monitoring.logger                             import Logger


class ProfileDataset(Dataset):
    """Serves normalised z profiles rebuilt from Gaussian parameters.

    Attributes:
        x_axis      : Z axis of shape (H,), in metres.
        n_gaussians : Number of Gaussians per pixel.
        split_name  : Split this dataset covers ("train", "val" or "test").
        amps        : Amplitudes of shape (n_pixels, n_gaussians).
        mus         : Means of shape (n_pixels, n_gaussians), in metres.
        sigs        : Standard deviations of shape (n_pixels, n_gaussians), in metres.
        n_active    : Count of pixels carrying an amplitude above the threshold.
        index       : Flat pixel indices kept by the dataset, shuffled.
    """

    def __init__(
        self,
        param_arrays       : list[np.ndarray],
        z_axis             : np.ndarray,
        n_gaussians        : int,
        split_name         : str,
        activity_threshold : float,
        pixel_subsample    : float = 1.0,
        keep_empty_frac    : float = 0.05,
        seed               : int   = 0,
        normalizer         : Optional[ProfileNormalizer] = None,
        augmenter          : Optional[ProfileAugmenter]  = None,
        logger             : Optional[Logger] = None,
    ) -> None:
        """Stacks the parameter maps and selects the pixels to serve.

        Args:
            param_arrays       : Per-region parameter maps of shape (3 * n_gaussians, row, col), amplitude, mean and sigma interleaved.
            z_axis             : Z axis of shape (H,), in metres.
            n_gaussians        : Number of Gaussians per pixel.
            split_name         : Split name; augmentation only runs for "train".
            activity_threshold : Amplitude above which a pixel counts as active, as recorded by the parameter run.
            pixel_subsample    : Fraction of active pixels kept.
            keep_empty_frac    : Fraction of inactive pixels kept.
            seed               : Seed for subsampling and shuffling.
            normalizer         : Applied to each curve when given.
            augmenter          : Applied to training curves when given.
            logger             : Logger for the dataset summary table.
        """
        self.z_axis      = np.asarray(z_axis, dtype=np.float32)
        self.n_gaussians = int(n_gaussians)
        self.split_name  = split_name
        self.normalizer  = normalizer
        self.augmenter   = augmenter

        self._stack_parameters(param_arrays)
        self._select_pixels(activity_threshold, pixel_subsample, keep_empty_frac, seed)

        if logger is not None:
            self._log(logger)

    def _stack_parameters(self, param_arrays: list[np.ndarray]) -> None:
        """Flattens and concatenates the region parameter maps into pixel rows.

        Args:
            param_arrays : Per-region parameter maps of shape (3 * n_gaussians, rows, cols).
        """
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

    def _select_pixels(self, activity_threshold: float, pixel_subsample: float, keep_empty_frac: float, seed: int) -> None:
        """Builds the shuffled index of active and retained empty pixels.

        Args:
            activity_threshold : Amplitude above which a pixel counts as active, as recorded by the parameter run.
            pixel_subsample    : Fraction of active pixels kept, when below 1.
            keep_empty_frac    : Fraction of inactive pixels kept.
            seed               : Seed for the subsampling and shuffling generator.

        Raises:
            ValueError : If the split holds no pixel above the activity threshold.
        """
        active = GaussianActivity.is_active(self.amps, activity_threshold).any(axis=1)
        rng    = np.random.default_rng(seed)

        active_idx = np.flatnonzero(active)
        empty_idx  = np.flatnonzero(~active)

        if len(active_idx) == 0:
            raise ValueError(f"Split '{self.split_name}' holds no pixel with an amplitude above the activity threshold {activity_threshold:g} recorded by the parameter run; pick a split region that carries components or regenerate the labels at a lower threshold.")

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
        """Writes the pixel counts and profile geometry to the log."""
        logger.section(f"[ProfileDataset: {self.split_name}]")
        logger.kv_table({
            "Total pixels"   : int(self.amps.shape[0]),
            "Active pixels"  : self.n_active,
            "Kept pixels"    : int(self.index.shape[0]),
            "Profile length" : int(self.z_axis.shape[0]),
            "Gaussians"      : self.n_gaussians,
        })

    def __len__(self) -> int:
        """Returns the number of pixels served by this split."""
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        """Evaluates, augments and normalises the profile at position `i`.

        Args:
            i : Position within the kept pixel index.

        Returns:
            Tuple of the noisy input curve and the clean target curve, both of
            shape (H,) in normalised units. Outside the training split the two
            are the same array, since augmentation and noise are skipped.
        """
        idx   = self.index[i]
        curve = GaussianMixture.evaluate_batch(
            self.z_axis,
            self.amps[idx:idx + 1],
            self.mus[idx:idx + 1],
            self.sigs[idx:idx + 1],
        )[0].astype(np.float32)

        if self.augmenter is not None and self.split_name == "train":
            curve = self.augmenter(curve)

        if self.normalizer is not None:
            curve = self.normalizer.normalize(curve)

        target = curve

        if self.augmenter is not None and self.split_name == "train":
            curve = self.augmenter.add_noise(curve)

        return curve, target
