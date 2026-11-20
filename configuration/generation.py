from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path


@dataclass
class GenerationConfig:
    output_directory : Path  = Path("datasets/synthetic")
    rows             : int   = 256
    cols             : int   = 256
    input_channels   : int   = 4
    smoothing_pixels : float = 3.0
    neighbourhood    : int   = 9
    input_noise      : float = 0.05
    target_noise     : float = 0.02
    val_fraction     : float = 0.15
    test_fraction    : float = 0.15
    seed             : int   = 0


@dataclass
class GenerateDatasetEntryConfig:
    generation : GenerationConfig = field(default_factory=GenerationConfig)
    log_dir    : Path             = Path("logs")
