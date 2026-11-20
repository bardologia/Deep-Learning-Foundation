
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.benchmark.general    import ComparisonReportConfig
from configuration.training.general.run import RunPathsConfig


@dataclass
class TrialComparisonConfig:

    runs_dir : Path      = Path("runs/backbone")
    run_tags : List[str] = field(default_factory=list)

    inference_split : str = "test"

    compare_images : bool           = True
    embed_images   : bool           = False
    output_dir     : Optional[Path] = None


@dataclass
class SeedComparisonConfig:

    runs_dir   : Path      = Path("runs")
    group_tags : List[str] = field(default_factory=list)

    split            : str = "test"
    inference_subdir : str = ""
    output_subdir    : str = ""

    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"

    compute_disagreement : bool = False
    compute_ensemble     : bool = True
    maps_subdir          : str  = "maps"


@dataclass
class ComparisonPathsConfig:

    log_base_dir : Path = RunPathsConfig.log_base_dir


@dataclass
class ComparisonEntryConfig:

    run_tag         : Optional[str] = None
    embed_images    : bool          = ComparisonReportConfig.embed_images
    inference_split : str           = "test"

    paths : ComparisonPathsConfig = field(default_factory=ComparisonPathsConfig)
