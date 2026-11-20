
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List

from configuration.training.general.run import RunPathsConfig


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

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    checkpoint_filename : str = "best_model.pt"
    output_subdir       : str = "weight_xray"

    make_plots   : bool = True
    embed_images : bool = False

    thresholds : WeightXrayThresholds = field(default_factory=WeightXrayThresholds)

    svd_max_dim          : int = 4096
    duplicate_max_units  : int = 512
    max_layer_histograms : int = 24

    def to_config(self, run_dir: Path) -> WeightXrayConfig:
        return WeightXrayConfig(
            checkpoint_path      = run_dir,
            output_dir           = Path(run_dir) / self.output_subdir,
            checkpoint_filename  = self.checkpoint_filename,
            thresholds           = self.thresholds,
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

    runs_dir   : Path      = Path("runs/backbone")
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

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    collector_dir : Path = Path("results/collected_reports")

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
class ExperimentReportMailConfig:

    staging_dir : Path = Path("results/mailed_reports")

    comparison_dirname : str       = "_comparison"
    comparison_reports : List[str] = field(default_factory=lambda: ["overview.md", "metrics_comparison.md"])

    inference_dirname : str = "inference"
    report_filename   : str = "report.md"

    regenerate_reports : bool = False
    embed_images       : bool = False


@dataclass
class ArtifactMailConfig:

    source_dir  : Path = Path("datasets/synthetic_small")
    staging_dir : Path = Path("results/mailed_reports")

    label : str = ""

    required_paths : List[str] = field(default_factory=list)
    optional_paths : List[str] = field(default_factory=list)
    analysis_paths : List[str] = field(default_factory=list)
    analysis_entry : str       = ""

    excluded_paths    : List[str] = field(default_factory=list)
    excluded_suffixes : List[str] = field(default_factory=list)

    max_file_mb       : float = 0.0
    max_attachment_mb : float = 15.0

    @property
    def archive_stem(self) -> str:
        return self.label.replace("/", "__")

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)

    @property
    def max_attachment_bytes(self) -> int:
        return int(self.max_attachment_mb * 1024 * 1024)


@dataclass
class ReportMailEntryConfig:

    TARGETS = ("experiments", "runs", "datasets", "param_extraction")

    target : str = "experiments"

    experiments_dir   : Path      = Path("runs/sequences")
    experiment_filter : List[str] = field(default_factory=list)

    comparison_dirname : str       = "_comparison"
    comparison_reports : List[str] = field(default_factory=lambda: ["overview.md", "metrics_comparison.md"])

    inference_dirname : str = "inference"
    report_filename   : str = "report.md"

    regenerate_reports : bool = False
    embed_images       : bool = False

    runs_dir   : Path      = Path("runs")
    run_filter : List[str] = field(default_factory=list)

    run_excluded_paths    : List[str] = field(default_factory=lambda: ["checkpoints", "cubes"])
    run_excluded_suffixes : List[str] = field(default_factory=lambda: [".pt", ".pth", ".ckpt", ".npy", ".npz", ".h5", ".pkl"])
    run_max_file_mb       : float     = 5.0

    datasets_dir   : Path      = Path("datasets")
    dataset_filter : List[str] = field(default_factory=list)

    dataset_required_paths : List[str] = field(default_factory=lambda: ["data/dataset.json", "meta"])
    dataset_optional_paths : List[str] = field(default_factory=lambda: ["logs", "docs"])

    params_dir       : Path      = Path("datasets")
    param_run_filter : List[str] = field(default_factory=list)

    param_required_paths : List[str] = field(default_factory=lambda: ["param_extraction_meta.json"])
    param_optional_paths : List[str] = field(default_factory=lambda: ["logs", "docs"])
    param_analysis_paths : List[str] = field(default_factory=lambda: ["fit_metrics_summary.json", "images"])

    staging_dir : Path = Path("results/mailed_reports")

    max_attachment_mb : float = 15.0

    recipient      : str   = "victorvicentin42@gmail.com"
    sender         : str   = "victorvicentin42@gmail.com"
    smtp_host      : str   = "smtp.gmail.com"
    smtp_port      : int   = 465
    smtp_timeout_s : float = 300.0
    password_env   : str   = "REPORT_MAIL_PASSWORD"
    subject_prefix : str   = "ML-Template results"
    dry_run        : bool  = False

    def to_config(self, label: str) -> ExperimentReportMailConfig:
        return ExperimentReportMailConfig(
            staging_dir        = Path(self.staging_dir) / label,
            comparison_dirname = self.comparison_dirname,
            comparison_reports = list(self.comparison_reports),
            inference_dirname  = self.inference_dirname,
            report_filename    = self.report_filename,
            regenerate_reports = self.regenerate_reports,
            embed_images       = self.embed_images,
        )

    def run_label(self, run_dir: Path) -> str:
        path = Path(run_dir)
        root = Path(self.runs_dir)

        if path == root:
            return f"{path.parent.name}/{path.name}"

        return path.relative_to(root).as_posix()

    def to_run_config(self, run_dir: Path) -> ArtifactMailConfig:
        label = self.run_label(run_dir)

        return ArtifactMailConfig(
            source_dir        = Path(run_dir),
            staging_dir       = Path(self.staging_dir) / label,
            label             = label,
            excluded_paths    = list(self.run_excluded_paths),
            excluded_suffixes = list(self.run_excluded_suffixes),
            max_file_mb       = self.run_max_file_mb,
            max_attachment_mb = self.max_attachment_mb,
        )

    def dataset_label(self, dataset_dir: Path) -> str:
        path = Path(dataset_dir)
        return f"{path.parent.name}/{path.name}"

    def to_dataset_config(self, dataset_dir: Path) -> ArtifactMailConfig:
        label = self.dataset_label(dataset_dir)

        return ArtifactMailConfig(
            source_dir        = Path(dataset_dir),
            staging_dir       = Path(self.staging_dir) / label,
            label             = label,
            required_paths    = list(self.dataset_required_paths),
            optional_paths    = list(self.dataset_optional_paths),
            max_attachment_mb = self.max_attachment_mb,
        )

    def param_run_label(self, run_dir: Path) -> str:
        path = Path(run_dir)
        return f"{path.parent.parent.name}/{path.name}"

    def to_param_config(self, run_dir: Path) -> ArtifactMailConfig:
        label = self.param_run_label(run_dir)

        return ArtifactMailConfig(
            source_dir        = Path(run_dir),
            staging_dir       = Path(self.staging_dir) / label,
            label             = label,
            required_paths    = list(self.param_required_paths),
            optional_paths    = list(self.param_optional_paths),
            analysis_paths    = list(self.param_analysis_paths),
            analysis_entry    = "analyze_param_extraction",
            max_attachment_mb = self.max_attachment_mb,
        )


@dataclass
class ReceptiveFieldConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "receptive_field"

    window       : int       = 160
    n_row_probes : int       = 4
    n_col_probes : int       = 3
    mass_windows : List[int] = field(default_factory=lambda: [8, 16, 24, 32, 40, 48, 56, 64, 96, 128])

    figure_style : str = "report"


@dataclass
class ActivationXrayConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "activation_xray"

    batch_size  : int = 8
    max_batches : int = 8

    dead_zero_frac_warn        : float = 0.90
    dead_zero_frac_critical    : float = 0.99
    dead_channel_frac_warn     : float = 0.25
    explode_abs_threshold      : float = 1e4
    constant_std_threshold     : float = 1e-7
    channel_collapse_frac_warn : float = 0.10

    make_plots           : bool = True
    max_layer_histograms : int  = 12
    figure_style         : str  = "report"


@dataclass
class InputAttributionConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "input_attribution"

    window       : int = 64
    n_row_probes : int = 4
    n_col_probes : int = 3

    occlude          : bool  = True
    batch_size       : int   = 8
    max_batches      : int   = 2

    figure_style : str = "report"


@dataclass
class LayerProbeConfig:

    runs_dir   : Path      = Path("runs/backbone")
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
    n_folds           : int   = 4
    seed              : int   = 0

    figure_style : str = "report"


@dataclass
class LossLandscapeConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "loss_landscape"

    batch_size     : int   = 4
    max_batches    : int   = 2
    span           : float = 0.5
    n_directions   : int   = 4
    n_points_1d    : int   = 21
    n_points_2d    : int   = 11
    direction_seed : int   = 0

    figure_style : str = "report"


@dataclass
class RobustnessConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "robustness"

    batch_size       : int         = 4
    max_batches      : int         = 2
    noise_sigmas     : List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0])
    noise_draws      : int         = 3
    gain_factors     : List[float] = field(default_factory=lambda: [0.5, 0.8, 1.25, 2.0])
    shift_pixels     : List[int]   = field(default_factory=lambda: [1, 2, 4, 8])
    draws_per_count  : int         = 3
    seed             : int         = 0

    figure_style : str = "report"


@dataclass
class TrainingEvolutionConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "training_evolution"

    window       : int = 64
    n_row_probes : int = 2
    n_col_probes : int = 2
    fps          : int = 4

    figure_style : str = "report"


@dataclass
class PaperFigurePackConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split            : str = "test"
    inference_subdir : str = ""
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"
    figures_subdir   : str = "figures"

    patterns : List[str] = field(default_factory=lambda: ["pixel_maps/*", "param_scatter/*", "stratified/*", "profiles/*", "data_consistency/*"])

    output_dir : Path = Path("results/paper_figures")


@dataclass
class ResultTableEntryConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split            : str = "test"
    inference_subdir : str = ""
    metrics_filename : str = "metrics.json"

    metrics   : tuple = ()
    sort_by   : str   = ""
    ascending : bool  = True
    precision : int   = 4

    output_dir : Path = Path("results/tables")


@dataclass
class OnnxExportEntryConfig:

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    opset         : int  = 18
    dynamic_batch : bool = True


@dataclass
class TrainingProfileEntryConfig:

    model_name : str  = ""
    gpu        : int  = 0
    seed       : int  = 0
    use_amp    : bool = False

    batch_size      : int = 64
    num_workers     : int = 4
    prefetch_factor : int = 4

    warmup_steps   : int  = 8
    active_steps   : int  = 24
    record_shapes  : bool = True
    profile_memory : bool = True
    with_stack     : bool = False

    paths      : RunPathsConfig = field(default_factory=RunPathsConfig)
    output_dir : Path           = Path("runs/profiles")
