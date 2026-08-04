from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.benchmark.general    import ComparisonReportConfig, SizeMatchConfig
from configuration.training.general.run import RunPathsConfig


@dataclass
class TrialComparisonConfig:
    runs_dir : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_tags : List[str] = field(default_factory=list)

    compare_images : bool           = True
    compare_gifs   : bool           = True
    embed_images   : bool           = False
    output_dir     : Optional[Path] = None


@dataclass
class PreprocessingComparisonConfig:
    runs_dir : Path      = Path("/ste/rnd/User/vice_vi/Dataset")
    run_tags : List[str] = field(default_factory=list)

    pixel_sample : int = 200000
    block_size   : int = 8
    range_chunk  : int = 512
    workers      : int = 4

    make_plots : bool           = True
    output_dir : Optional[Path] = None


@dataclass
class SeedComparisonConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs")
    group_tags : List[str] = field(default_factory=list)

    inference_subdir : str = ""
    output_subdir    : str = ""

    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"

    compute_disagreement : bool = True
    cubes_subdir         : str  = "cubes"


@dataclass
class ComparisonPathsConfig:
    log_base_dir : Path = RunPathsConfig.log_base_dir


@dataclass
class ComparisonEntryConfig:
    run_tag         : Optional[str] = None
    reference_model : str           = SizeMatchConfig.reference_model
    embed_images    : bool          = ComparisonReportConfig.embed_images

    paths : ComparisonPathsConfig = field(default_factory=ComparisonPathsConfig)


@dataclass
class ParamExtractionComparisonConfig:
    params_dir : Path      = Path("/ste/rnd/User/vice_vi/Dataset")
    run_tags   : List[str] = field(default_factory=list)

    pixel_sample : int = 200000
    block_size   : int = 8
    range_chunk  : int = 512

    make_plots : bool           = True
    output_dir : Optional[Path] = None
