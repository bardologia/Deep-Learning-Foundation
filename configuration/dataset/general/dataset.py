from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib     import Path
from typing      import Tuple

from configuration.normalization.general import NormalizationConfig
from tools.data.regions                  import SplitRegions


@dataclass
class PatchConfig:

    size                  : Tuple[int, int] = (32, 32)
    stride                : Tuple[int, int] = (16, 16)
    use_symmetric_padding : bool            = True


@dataclass
class AugmentationConfig:

    p_flip_h  : float = 0.5
    p_flip_v  : float = 0.5
    p_rot90   : float = 0.0

    noise_std : float = 0.01
    p_noise   : float = 0.0


@dataclass
class DatasetConfig:

    dataset_directory : Path
    split_regions     : SplitRegions
    patch             : PatchConfig         = field(default_factory=PatchConfig)
    normalization     : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation      : AugmentationConfig  = field(default_factory=AugmentationConfig)

    batch_size          : int  = 8
    num_workers         : int  = 4
    prefetch_factor     : int  = 2
    fingerprint_dataset : bool = True

    def as_dict(self) -> dict:
        return {
            "dataset_directory"   : str(self.dataset_directory),
            "split_regions"       : self.split_regions.bounds(),
            "patch"               : asdict(self.patch),
            "normalization"       : self.normalization.as_dict(),
            "augmentation"        : asdict(self.augmentation),
            "batch_size"          : self.batch_size,
            "num_workers"         : self.num_workers,
            "prefetch_factor"     : self.prefetch_factor,
            "fingerprint_dataset" : self.fingerprint_dataset,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetConfig":
        patch = payload["patch"]

        return cls(
            dataset_directory   = Path(payload["dataset_directory"]),
            split_regions       = SplitRegions.from_bounds(payload["split_regions"]),
            patch               = PatchConfig(size=tuple(patch["size"]), stride=tuple(patch["stride"]), use_symmetric_padding=bool(patch["use_symmetric_padding"])),
            normalization       = NormalizationConfig.from_dict(payload["normalization"]),
            augmentation        = AugmentationConfig(**payload["augmentation"]),
            batch_size          = int(payload["batch_size"]),
            num_workers         = int(payload["num_workers"]),
            prefetch_factor     = int(payload["prefetch_factor"]),
            fingerprint_dataset = bool(payload["fingerprint_dataset"]),
        )
