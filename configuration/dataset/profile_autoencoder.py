from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional, Tuple

from tools.data.regions import SplitRegions


@dataclass
class ProfileAugmentationConfig:
    p_amp_scale     : float               = 0.0
    amp_scale_range : Tuple[float, float] = (0.9, 1.1)
    p_shift         : float               = 0.0
    max_shift       : int                 = 4
    p_flip          : float               = 0.5
    p_noise         : float               = 0.0
    noise_std       : float               = 0.01


@dataclass
class ProfileDatasetConfig:
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
