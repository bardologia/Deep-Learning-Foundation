"""Dataset configuration for the 1D profile autoencoder pipeline.

Profile datasets flatten the preprocessed scene into individual reflectivity
profiles along the z axis, subsampling pixels and controlling how many
empty profiles are retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional, Tuple

from tools.data.regions import SplitRegions


@dataclass
class ProfileAugmentationConfig:
    """Probabilities and strengths of the profile-level augmentations.

    Attributes:
        p_amp_scale     : Probability of rescaling the profile amplitude.
        amp_scale_range : Lower and upper bound of the amplitude scale factor.
        p_shift         : Probability of shifting the profile along z.
        max_shift       : Largest shift in z samples.
        p_flip          : Probability of reversing the profile along z.
        p_noise         : Probability of applying additive Gaussian noise.
        noise_std       : Standard deviation of the additive noise in normalised units.
    """

    p_amp_scale     : float               = 0.0
    amp_scale_range : Tuple[float, float] = (0.9, 1.1)
    p_shift         : float               = 0.0
    max_shift       : int                 = 4
    p_flip          : float               = 0.5
    p_noise         : float               = 0.0
    noise_std       : float               = 0.01


@dataclass
class ProfileDatasetConfig:
    """Full specification of a profile dataset and its loaders.

    Attributes:
        preprocessing_run_directory : Directory holding the preprocessed scene.
        split_regions               : Row and column regions defining train, val and test.
        parameters_path             : Directory holding the extracted Gaussian parameters.
        n_gaussians                 : Number of Gaussian components describing one profile.
        x_min                       : Lower bound of the z axis in metres.
        x_max                       : Upper bound of the z axis in metres.
        pixel_subsample             : Fraction of scene pixels kept as profiles.
        keep_empty_frac             : Fraction of empty profiles retained after subsampling.
        amp_zero_thr                : Amplitude below which a Gaussian counts as absent.
        batch_size                  : Profiles per batch.
        num_workers                 : Data loader worker processes.
        prefetch_factor             : Batches prefetched per worker.
        pin_memory                  : Whether loader batches are pinned for faster host-to-device copies.
        shuffle_train               : Whether the training split is shuffled.
        stats_max_samples           : Profiles sampled when computing normalisation statistics.
        augmentation                : Profile augmentation settings.
    """

    preprocessing_run_directory : Path
    split_regions               : SplitRegions
    parameters_path             : Optional[Path] = None

    n_gaussians : int   = 5
    x_min       : float = 0.0
    x_max       : float = 1.0

    pixel_subsample : float = 1.0
    keep_empty_frac : float = 0.05
    amp_zero_thr    : float = 1e-4

    batch_size      : int  = 1024
    num_workers     : int  = 32
    prefetch_factor : int  = 2
    pin_memory      : bool = True
    shuffle_train   : bool = True

    stats_max_samples: int = 100_000

    augmentation: ProfileAugmentationConfig = field(default_factory=ProfileAugmentationConfig)
