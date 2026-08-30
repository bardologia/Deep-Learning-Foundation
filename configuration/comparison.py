"""Configuration dataclasses for the cross-run comparison entry points.

Each dataclass drives one comparison report: trained backbone trials, seed
groups of the same configuration, or parameter-extraction variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.benchmark.general    import ComparisonReportConfig, SizeMatchConfig
from configuration.training.general.run import RunPathsConfig


@dataclass
class TrialComparisonConfig:
    """Settings of the report comparing several trained runs.

    Attributes:
        runs_dir        : Directory holding the run folders to compare.
        run_tags        : Run folder names taking part in the comparison.
        inference_split : Dataset split whose inference results are read.
        compare_images  : Whether figures are placed side by side in the report.
        compare_gifs    : Whether animations are placed side by side in the report.
        embed_images    : Whether figures are embedded instead of linked.
        output_dir      : Destination of the report; ``None`` derives it from the runs.
    """

    runs_dir : Path      = Path("runs/backbone")
    run_tags : List[str] = field(default_factory=list)

    inference_split : str = "test"

    compare_images : bool           = True
    compare_gifs   : bool           = True
    embed_images   : bool           = False
    output_dir     : Optional[Path] = None


@dataclass
class SeedComparisonConfig:
    """Settings of the report aggregating runs that differ only by random seed.

    Attributes:
        runs_dir             : Root directory holding the seed run folders.
        group_tags           : Run group names whose seeds are aggregated.
        inference_subdir     : Subdirectory of a run holding its inference results.
        output_subdir        : Subdirectory receiving the aggregated report.
        metrics_filename     : File holding the per-run metrics.
        report_filename      : File name of the generated markdown report.
        compute_disagreement : Whether inter-seed disagreement maps are computed.
        compute_ensemble     : Whether the across-seed mean prediction is scored as a deep ensemble.
        cubes_subdir         : Subdirectory holding the saved cubes used for disagreement.
        active_amp_thr       : Physical amplitude at or above which a Gaussian slot counts as active in the GT-matched slot alignment; must match the threshold the seeds' inference metrics used.
    """

    runs_dir   : Path      = Path("runs")
    group_tags : List[str] = field(default_factory=list)

    inference_subdir : str = ""
    output_subdir    : str = ""

    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"

    compute_disagreement : bool  = True
    compute_ensemble     : bool  = True
    cubes_subdir         : str   = "cubes"
    active_amp_thr       : float = 1e-4


@dataclass
class ComparisonPathsConfig:
    """Paths used by the comparison entry point.

    Attributes:
        log_base_dir : Root directory receiving comparison logs.
    """

    log_base_dir : Path = RunPathsConfig.log_base_dir


@dataclass
class ComparisonEntryConfig:
    """Entry configuration of the benchmark comparison report.

    Attributes:
        run_tag         : Benchmark run whose models are compared.
        reference_model : Model the others are reported relative to.
        embed_images    : Whether figures are embedded instead of linked.
        paths           : Log path configuration.
    """

    run_tag         : Optional[str] = None
    reference_model : str           = SizeMatchConfig.reference_model
    embed_images    : bool          = ComparisonReportConfig.embed_images

    paths : ComparisonPathsConfig = field(default_factory=ComparisonPathsConfig)


@dataclass
class ParamExtractionComparisonConfig:
    """Settings of the report comparing Gaussian parameter-extraction variants.

    Attributes:
        params_dir   : Directory holding the extracted parameter variants.
        run_tags     : Parameter folder names taking part in the comparison.
        pixel_sample : Number of pixels sampled for the statistics.
        block_size   : Side length in pixels of the blocks used for spatial statistics.
        col_chunk  : Number of col lines read per chunk.
        make_plots   : Whether comparison figures are produced.
        output_dir   : Destination of the report; ``None`` derives it from the inputs.
    """

    params_dir : Path      = Path("datasets")
    run_tags   : List[str] = field(default_factory=list)

    pixel_sample : int = 200000
    block_size   : int = 8
    col_chunk    : int = 512

    make_plots : bool           = True
    output_dir : Optional[Path] = None
