from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional


@dataclass
class WeightXrayThresholds:
    dead_abs_threshold     : float = 1e-6
    dead_fraction_warn     : float = 0.50
    dead_fraction_critical : float = 0.95

    dead_unit_norm_frac     : float = 1e-3
    dead_unit_fraction_warn : float = 0.10

    uniform_cv_threshold   : float = 0.05
    constant_std_threshold : float = 1e-9

    rank_ratio_warn        : float = 0.50
    rank_ratio_critical    : float = 0.10

    explode_abs_threshold  : float = 1e3
    spectral_norm_warn     : float = 1e2
    bias_abs_threshold     : float = 1e2

    norm_scale_dead_value  : float = 1e-3
    norm_scale_dead_frac   : float = 0.10
    running_var_dead_value : float = 1e-6
    running_var_dead_frac  : float = 0.10

    init_ratio_low         : float = 0.10
    init_ratio_high        : float = 3.00

    duplicate_cosine        : float = 0.999
    duplicate_fraction_warn : float = 0.05


@dataclass
class WeightXrayConfig:
    checkpoint_path : Path = Path("logs")
    output_dir      : Path = Path("results/weight_xray")

    checkpoint_filename  : str  = "best_model.pt"
    state_dict_keys      : list = field(default_factory=lambda: ["params"])

    thresholds : WeightXrayThresholds = field(default_factory=WeightXrayThresholds)

    svd_max_dim          : int = 4096
    duplicate_max_units  : int = 512
    histogram_sample_max : int = 2_000_000
    max_layer_histograms : int = 24

    make_plots   : bool = True
    embed_images : bool = False

    @property
    def resolved_checkpoint_path(self) -> Path:
        path = Path(self.checkpoint_path)
        if path.is_dir():
            return path / self.checkpoint_filename
        return path

    @property
    def report_directory(self) -> Path:
        return Path(self.output_dir)

    @property
    def plots_directory(self) -> Path:
        return self.report_directory / "plots"

    @property
    def report_markdown_path(self) -> Path:
        return self.report_directory / "weight_xray.md"

    @property
    def report_json_path(self) -> Path:
        return self.report_directory / "weight_xray.json"


@dataclass
class WeightXrayEntryConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    checkpoint_filename : str = "best_model.pt"
    output_subdir       : str = "weight_xray"

    make_plots   : bool = True
    embed_images : bool = False

    dead_abs_threshold      : float = 1e-6
    dead_fraction_warn      : float = 0.50
    dead_unit_fraction_warn : float = 0.10
    uniform_cv_threshold    : float = 0.05
    rank_ratio_warn         : float = 0.50
    explode_abs_threshold   : float = 1e3
    duplicate_cosine        : float = 0.999

    svd_max_dim          : int = 4096
    duplicate_max_units  : int = 512
    max_layer_histograms : int = 24

    def to_config(self, run_dir: Path) -> WeightXrayConfig:
        thresholds = WeightXrayThresholds(
            dead_abs_threshold      = self.dead_abs_threshold,
            dead_fraction_warn      = self.dead_fraction_warn,
            dead_unit_fraction_warn = self.dead_unit_fraction_warn,
            uniform_cv_threshold    = self.uniform_cv_threshold,
            rank_ratio_warn         = self.rank_ratio_warn,
            explode_abs_threshold   = self.explode_abs_threshold,
            duplicate_cosine        = self.duplicate_cosine,
        )

        return WeightXrayConfig(
            checkpoint_path      = run_dir,
            output_dir           = Path(run_dir) / self.output_subdir,
            checkpoint_filename  = self.checkpoint_filename,
            thresholds           = thresholds,
            svd_max_dim          = self.svd_max_dim,
            duplicate_max_units  = self.duplicate_max_units,
            max_layer_histograms = self.max_layer_histograms,
            make_plots           = self.make_plots,
            embed_images         = self.embed_images,
        )


@dataclass
class TensorboardExportConfig:
    run_directory       : Path = Path("logs")
    tensorboard_dirname : str  = "tensorboard"
    output_subdir       : str  = "tensorboard_plots"

    @property
    def tensorboard_directory(self) -> Path:
        return Path(self.run_directory) / self.tensorboard_dirname

    @property
    def plots_directory(self) -> Path:
        return Path(self.run_directory) / self.output_subdir


@dataclass
class TensorboardExportEntryConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    tensorboard_dirname : str = "tensorboard"
    output_subdir       : str = "tensorboard_plots"

    def to_config(self, run_dir: Path) -> TensorboardExportConfig:
        return TensorboardExportConfig(
            run_directory       = Path(run_dir),
            tensorboard_dirname = self.tensorboard_dirname,
            output_subdir       = self.output_subdir,
        )


@dataclass
class ReportCollectionConfig:
    run_directory : Path = Path("logs")
    collector_dir : Path = Path("results/collected_reports")

    inference_dirname : str  = "inference"
    report_filename   : str  = "report.md"
    latest_only       : bool = True
    embed_images      : bool = False

    @property
    def inference_directory(self) -> Path:
        return Path(self.run_directory) / self.inference_dirname


@dataclass
class ReportCollectionEntryConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    collector_dir : Path = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/results/collected_reports")

    inference_dirname : str  = "inference"
    report_filename   : str  = "report.md"
    latest_only       : bool = True
    embed_images      : bool = False

    def to_config(self, run_dir: Path) -> ReportCollectionConfig:
        return ReportCollectionConfig(
            run_directory     = Path(run_dir),
            collector_dir     = Path(self.collector_dir),
            inference_dirname = self.inference_dirname,
            report_filename   = self.report_filename,
            latest_only       = self.latest_only,
            embed_images      = self.embed_images,
        )


@dataclass
class ReceptiveFieldConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "receptive_field"

    window           : int       = 160
    n_azimuth_probes : int       = 4
    n_range_probes   : int       = 3
    mass_windows     : List[int] = field(default_factory=lambda: [8, 16, 24, 32, 40, 48, 56, 64, 96, 128])

    figure_style : str = "report"


@dataclass
class ActivationXrayConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "activation_xray"

    batch_size  : int = 8
    max_batches : int = 8

    dead_zero_frac_warn     : float = 0.90
    dead_zero_frac_critical : float = 0.99
    dead_channel_frac_warn  : float = 0.25
    explode_abs_threshold   : float = 1e4
    constant_std_threshold  : float = 1e-7

    make_plots           : bool = True
    max_layer_histograms : int  = 12
    figure_style         : str  = "report"


@dataclass
class InputAttributionConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "input_attribution"

    window           : int = 64
    n_azimuth_probes : int = 4
    n_range_probes   : int = 3

    occlude          : bool  = True
    batch_size       : int   = 8
    max_batches      : int   = 2
    render_amp_floor : float = 0.0

    figure_style : str = "report"


@dataclass
class AttentionCaptureConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "attention_capture"

    batch_size       : int = 4
    max_gate_figures : int = 8

    figure_style : str = "report"


@dataclass
class LayerProbeConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "layer_probes"

    batch_size        : int   = 4
    max_batches       : int   = 4
    max_layers        : int   = 24
    samples_per_batch : int   = 512
    ridge_lambda      : float = 1.0
    test_fraction     : float = 0.5
    seed              : int   = 0

    figure_style : str = "report"


@dataclass
class CkaConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str  = "test"
    device          : str  = "cpu"
    checkpoint_name : str  = "best_model.pt"
    output_dir      : Path = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/results/cka")

    batch_size        : int = 4
    max_batches       : int = 4
    max_layers        : int = 16
    samples_per_batch : int = 512
    sample_seed       : int = 0

    figure_style : str = "report"


@dataclass
class LossLandscapeConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "loss_landscape"

    batch_size     : int   = 4
    max_batches    : int   = 2
    span           : float = 0.5
    n_points_1d    : int   = 21
    n_points_2d    : int   = 11
    direction_seed : int   = 0

    figure_style : str = "report"


@dataclass
class RobustnessConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "robustness"

    batch_size       : int         = 4
    max_batches      : int         = 2
    noise_sigmas     : List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0])
    draws_per_count  : int         = 3
    render_amp_floor : float       = 0.0
    seed             : int         = 0

    figure_style : str = "report"


@dataclass
class TrainingEvolutionConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "training_evolution"

    window           : int = 64
    n_azimuth_probes : int = 2
    n_range_probes   : int = 2
    fps              : int = 4

    figure_style : str = "report"


@dataclass
class PaperFigurePackConfig:
    runs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    inference_subdir : str = ""
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"
    figures_subdir   : str = "figures"

    patterns : List[str] = field(default_factory=lambda: ["pixel_maps/*", "param_scatter/*", "stratified/*", "profiles/*", "data_consistency/*"])

    output_dir : Path = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/publications/neurips2027/figures")
