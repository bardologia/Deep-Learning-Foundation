"""Configuration dataclasses for the post-training diagnostic analyses.

Each dataclass drives one analysis entry point over trained runs: weight and
activation x-rays, receptive-field measurement, input attribution, attention
capture, linear layer probes, representation similarity, loss landscapes,
robustness stress tests, training evolution animations, TensorBoard exports,
report collection, report mailing and the paper figure pack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List


@dataclass
class WeightXrayThresholds:
    """Decision thresholds that turn weight statistics into pathology flags.

    Attributes:
        dead_abs_threshold      : Magnitude below which a weight counts as dead.
        dead_fraction_warn      : Dead-weight fraction raising a warning.
        dead_fraction_critical  : Dead-weight fraction raising a critical flag.
        dead_unit_norm_frac     : Norm fraction below which a unit counts as dead.
        dead_unit_fraction_warn : Dead-unit fraction raising a warning.
        uniform_cv_threshold    : Coefficient of variation below which a tensor is flagged as suspiciously uniform.
        constant_std_threshold  : Standard deviation below which a tensor is constant.
        rank_ratio_warn         : Effective-rank ratio raising a warning.
        rank_ratio_critical     : Effective-rank ratio raising a critical flag.
        explode_abs_threshold   : Weight magnitude flagged as exploded.
        spectral_norm_warn      : Spectral norm raising a warning.
        bias_abs_threshold      : Bias magnitude flagged as exploded.
        norm_scale_dead_value   : Normalisation scale below which a channel is dead.
        norm_scale_dead_frac    : Dead normalisation-scale fraction raising a warning.
        running_var_dead_value  : Running variance below which a channel is dead.
        running_var_dead_frac   : Dead running-variance fraction raising a warning.
        init_ratio_low          : Lower bound of the accepted weight-to-initialisation scale ratio.
        init_ratio_high         : Upper bound of the accepted weight-to-initialisation scale ratio.
        duplicate_cosine        : Cosine similarity above which two units count as duplicates.
        duplicate_fraction_warn : Duplicate-unit fraction raising a warning.
    """

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
    """Settings of the weight x-ray analysis of a single checkpoint.

    Attributes:
        checkpoint_path      : Checkpoint file, or a run directory containing one.
        output_dir           : Directory receiving the report and figures.
        checkpoint_filename  : Checkpoint name looked up inside a run directory.
        state_dict_keys      : Keys of the checkpoint payload holding the weights.
        thresholds           : Pathology decision thresholds.
        svd_max_dim          : Largest matrix dimension still submitted to an SVD.
        duplicate_max_units  : Largest number of units compared pairwise for duplicates.
        histogram_sample_max : Weights sampled per histogram.
        max_layer_histograms : Number of per-layer histograms plotted.
        make_plots           : Whether figures are produced.
        embed_images         : Whether figures are embedded in the report instead of linked.
    """

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
        """Checkpoint file itself, resolving a run directory to its checkpoint name."""
        path = Path(self.checkpoint_path)
        if path.is_dir():
            return path / self.checkpoint_filename
        return path

    @property
    def report_directory(self) -> Path:
        """Directory receiving the weight x-ray outputs."""
        return Path(self.output_dir)

    @property
    def plots_directory(self) -> Path:
        """Directory receiving the weight x-ray figures."""
        return self.report_directory / "plots"

    @property
    def report_markdown_path(self) -> Path:
        """Path of the markdown weight x-ray report."""
        return self.report_directory / "weight_xray.md"

    @property
    def report_json_path(self) -> Path:
        """Path of the machine-readable weight x-ray report."""
        return self.report_directory / "weight_xray.json"


@dataclass
class WeightXrayEntryConfig:
    """Entry configuration running the weight x-ray over many run directories.

    Attributes:
        runs_dir             : Root directory holding the run folders.
        run_filter           : Run names to restrict the analysis to; empty means all.
        checkpoint_filename  : Checkpoint name loaded from each run.
        output_subdir        : Subdirectory of a run receiving its report.
        make_plots           : Whether figures are produced.
        embed_images         : Whether figures are embedded in the report instead of linked.
        thresholds           : Pathology decision thresholds.
        svd_max_dim          : Largest matrix dimension still submitted to an SVD.
        duplicate_max_units  : Largest number of units compared pairwise for duplicates.
        max_layer_histograms : Number of per-layer histograms plotted.
    """

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
        """Returns the single-run weight x-ray configuration for one run directory.

        Args:
            run_dir : Run folder to analyse.

        Returns:
            Configuration pointing at that run's checkpoint and output subdirectory.
        """
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
    """Settings of the TensorBoard scalar export for a single run.

    Attributes:
        run_directory       : Run folder holding the event files.
        tensorboard_dirname : Subdirectory of the run holding the event files.
        output_subdir       : Subdirectory of the run receiving the exported plots.
    """

    run_directory       : Path = Path("logs")
    tensorboard_dirname : str  = "tensorboard"
    output_subdir       : str  = "tensorboard_plots"

    @property
    def tensorboard_directory(self) -> Path:
        """Directory holding the run's TensorBoard event files."""
        return Path(self.run_directory) / self.tensorboard_dirname

    @property
    def plots_directory(self) -> Path:
        """Directory receiving the exported TensorBoard plots."""
        return Path(self.run_directory) / self.output_subdir


@dataclass
class TensorboardExportEntryConfig:
    """Entry configuration running the TensorBoard export over many runs.

    Attributes:
        runs_dir            : Root directory holding the run folders.
        run_filter          : Run names to restrict the export to; empty means all.
        tensorboard_dirname : Subdirectory of a run holding the event files.
        output_subdir       : Subdirectory of a run receiving the exported plots.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    tensorboard_dirname : str = "tensorboard"
    output_subdir       : str = "tensorboard_plots"

    def to_config(self, run_dir: Path) -> TensorboardExportConfig:
        """Returns the single-run TensorBoard export configuration for one run directory.

        Args:
            run_dir : Run folder whose event files are exported.

        Returns:
            Configuration pointing at that run's event and output directories.
        """
        return TensorboardExportConfig(
            run_directory       = Path(run_dir),
            tensorboard_dirname = self.tensorboard_dirname,
            output_subdir       = self.output_subdir,
        )


@dataclass
class ReportCollectionConfig:
    """Settings of the report collection for a single run.

    Attributes:
        run_directory     : Run folder whose inference reports are collected.
        collector_dir     : Destination directory gathering the copied reports.
        inference_dirname : Subdirectory of a run holding inference results.
        report_filename   : Report file name inside an inference directory.
        latest_only       : Whether only the most recent inference stamp is collected.
        embed_images      : Whether figures are embedded in the copied report.
    """

    run_directory : Path = Path("logs")
    collector_dir : Path = Path("results/collected_reports")

    inference_dirname : str  = "inference"
    report_filename   : str  = "report.md"
    latest_only       : bool = True
    embed_images      : bool = False

    @property
    def inference_directory(self) -> Path:
        """Directory holding the run's inference results."""
        return Path(self.run_directory) / self.inference_dirname


@dataclass
class ReportCollectionEntryConfig:
    """Entry configuration collecting inference reports across many runs.

    Attributes:
        runs_dir          : Root directory holding the run folders.
        run_filter        : Run names to restrict the collection to; empty means all.
        collector_dir     : Destination directory gathering the copied reports.
        inference_dirname : Subdirectory of a run holding inference results.
        report_filename   : Report file name inside an inference directory.
        latest_only       : Whether only the most recent inference stamp is collected.
        embed_images      : Whether figures are embedded in the copied report.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    collector_dir : Path = Path("results/collected_reports")

    inference_dirname : str  = "inference"
    report_filename   : str  = "report.md"
    latest_only       : bool = True
    embed_images      : bool = False

    def to_config(self, run_dir: Path) -> ReportCollectionConfig:
        """Returns the single-run report collection configuration for one run directory.

        Args:
            run_dir : Run folder whose reports are collected.

        Returns:
            Configuration pointing at that run's inference directory.
        """
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
    """Settings of the report gathering for a single experiment directory.

    Attributes:
        experiment_dir             : Experiment folder holding the trial runs and the comparison outputs.
        staging_dir                : Directory receiving this experiment's gathered files.
        comparison_dirname         : Subdirectory of the experiment holding the trial-comparison outputs.
        comparison_reports         : Report filenames gathered from the newest comparison stamp.
        inference_dirname          : Subdirectory of a trial holding its inference results.
        report_filename            : Report file name inside an inference directory.
        scheduler_results_filename : Experiment-level scheduler summary attached when present.
        resolved_config_relpath    : Path of the resolved entry configuration inside a run directory.
        embed_images               : Whether figures are embedded in the gathered reports.
    """

    experiment_dir : Path = Path("runs/sequences")
    staging_dir    : Path = Path("results/mailed_reports")

    comparison_dirname : str       = "_comparison"
    comparison_reports : List[str] = field(default_factory=lambda: ["overview.md", "metrics_comparison.md"])

    inference_dirname : str = "inference"
    report_filename   : str = "report.md"

    scheduler_results_filename : str = "train_scheduler_results.json"
    resolved_config_relpath    : str = "docs/resolved_entry_config.json"

    embed_images : bool = False


@dataclass
class ReportMailEntryConfig:
    """Entry configuration mailing the reports of every experiment in a folder.

    Attributes:
        experiments_dir            : Root directory holding one folder per experiment, as written by a sequence run.
        experiment_filter          : Experiment names to restrict the mailing to; empty means all.
        staging_dir                : Directory receiving the gathered files, one ``<parent>/<experiment>`` subtree per experiment.
        comparison_dirname         : Subdirectory of an experiment holding the trial-comparison outputs.
        comparison_reports         : Report filenames gathered from the newest comparison stamp.
        inference_dirname          : Subdirectory of a trial holding its inference results.
        report_filename            : Report file name inside an inference directory.
        scheduler_results_filename : Experiment-level scheduler summary attached when present.
        resolved_config_relpath    : Path of the resolved entry configuration inside a run directory.
        embed_images               : Whether figures are embedded in the gathered reports.
        recipient                  : Address every experiment message is sent to.
        sender                     : Address the messages are sent from, also the SMTP login user.
        smtp_host                  : SMTP server the messages are submitted to.
        smtp_port                  : SMTP SSL port of the server.
        password_env               : Environment variable holding the SMTP password, never stored in the config.
        subject_prefix             : Prefix of every message subject, followed by the experiment name.
        dry_run                    : Whether messages are only staged and logged instead of sent.
    """

    experiments_dir   : Path      = Path("runs/sequences")
    experiment_filter : List[str] = field(default_factory=list)

    staging_dir : Path = Path("results/mailed_reports")

    comparison_dirname : str       = "_comparison"
    comparison_reports : List[str] = field(default_factory=lambda: ["overview.md", "metrics_comparison.md"])

    inference_dirname : str = "inference"
    report_filename   : str = "report.md"

    scheduler_results_filename : str = "train_scheduler_results.json"
    resolved_config_relpath    : str = "docs/resolved_entry_config.json"

    embed_images : bool = False

    recipient      : str  = "victorvicentin42@gmail.com"
    sender         : str  = "victorvicentin42@gmail.com"
    smtp_host      : str  = "smtp.gmail.com"
    smtp_port      : int  = 465
    password_env   : str  = "REPORT_MAIL_PASSWORD"
    subject_prefix : str  = "ML-Template experiment reports"
    dry_run        : bool = False

    def experiment_label(self, experiment_dir: Path) -> str:
        """Returns the mailed identity of one experiment, prefixed by its parent directory.

        The parent prefix keeps same-named sweep directories apart: every
        cross-validation block writes its folds into ``<run_tag>/folds``, so
        the run tag rather than the literal ``folds`` tells the bundles apart
        in the subject line and on the staging tree.

        Args:
            experiment_dir : Experiment folder whose reports are mailed.
        """
        path = Path(experiment_dir)
        return f"{path.parent.name}/{path.name}"

    def to_config(self, experiment_dir: Path) -> ExperimentReportMailConfig:
        """Returns the single-experiment gathering configuration for one experiment directory.

        Args:
            experiment_dir : Experiment folder whose reports are gathered.

        Returns:
            Configuration staging that experiment's files under its labelled subtree of the staging directory.
        """
        return ExperimentReportMailConfig(
            experiment_dir             = Path(experiment_dir),
            staging_dir                = Path(self.staging_dir) / self.experiment_label(experiment_dir),
            comparison_dirname         = self.comparison_dirname,
            comparison_reports         = list(self.comparison_reports),
            inference_dirname          = self.inference_dirname,
            report_filename            = self.report_filename,
            scheduler_results_filename = self.scheduler_results_filename,
            resolved_config_relpath    = self.resolved_config_relpath,
            embed_images               = self.embed_images,
        )


@dataclass
class ReceptiveFieldConfig:
    """Settings of the empirical receptive-field measurement.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Run names to restrict the analysis to; empty means all.
        split            : Dataset split the probe patches are drawn from.
        device           : Torch device the gradients are computed on.
        checkpoint_name  : Checkpoint loaded from each run.
        output_subdir    : Subdirectory of a run receiving the results.
        window           : Side length in pixels of the analysed neighbourhood.
        n_row_probes : Probe pixels along row.
        n_col_probes   : Probe pixels along col.
        mass_windows     : Window sizes in pixels at which gradient mass is accumulated.
        figure_style     : Named matplotlib style applied to the figures.
    """

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
    """Settings of the activation x-ray recorded over forward channels.

    Attributes:
        runs_dir                   : Root directory holding the run folders.
        run_filter                 : Run names to restrict the analysis to; empty means all.
        split                      : Dataset split the batches are drawn from.
        device                     : Torch device the forward channels run on.
        checkpoint_name            : Checkpoint loaded from each run.
        output_subdir              : Subdirectory of a run receiving the results.
        batch_size                 : Patches per recorded batch.
        max_batches                : Number of batches recorded.
        dead_zero_frac_warn        : Zero fraction of an activation raising a warning.
        dead_zero_frac_critical    : Zero fraction raising a critical flag.
        dead_channel_frac_warn     : Dead-channel fraction raising a warning.
        explode_abs_threshold      : Activation magnitude flagged as exploded.
        constant_std_threshold     : Standard deviation below which an activation is constant.
        channel_collapse_frac_warn : Collapsed-channel fraction raising a warning.
        make_plots                 : Whether figures are produced.
        max_layer_histograms       : Number of per-layer histograms plotted.
        figure_style               : Named matplotlib style applied to the figures.
    """

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
    """Settings of the per-channel input attribution analysis.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Run names to restrict the analysis to; empty means all.
        split            : Dataset split the probe patches are drawn from.
        device           : Torch device the attributions are computed on.
        checkpoint_name  : Checkpoint loaded from each run.
        output_subdir    : Subdirectory of a run receiving the results.
        window           : Side length in pixels of the analysed neighbourhood.
        n_row_probes : Probe pixels along row.
        n_col_probes   : Probe pixels along col.
        occlude          : Whether occlusion attribution runs alongside the gradient one.
        batch_size       : Patches per batch.
        max_batches      : Number of batches analysed.
        render_amp_floor : Amplitude below which a component is omitted from the renders.
        figure_style     : Named matplotlib style applied to the figures.
    """

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
    render_amp_floor : float = 0.0

    figure_style : str = "report"


@dataclass
class AttentionCaptureConfig:
    """Settings of the attention and gate map capture for attention-based models.

    Attributes:
        runs_dir              : Root directory holding the run folders.
        run_filter            : Run names to restrict the analysis to; empty means all.
        split                 : Dataset split the batches are drawn from.
        device                : Torch device the forward channels run on.
        checkpoint_name       : Checkpoint loaded from each run.
        output_subdir         : Subdirectory of a run receiving the results.
        batch_size            : Patches per batch.
        max_batches           : Number of batches captured.
        max_gate_figures      : Attention-gate maps plotted.
        max_attention_figures : Attention weight maps plotted.
        figure_style          : Named matplotlib style applied to the figures.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str = "test"
    device          : str = "cpu"
    checkpoint_name : str = "best_model.pt"
    output_subdir   : str = "attention_capture"

    batch_size            : int = 4
    max_batches           : int = 2
    max_gate_figures      : int = 8
    max_attention_figures : int = 8

    figure_style : str = "report"


@dataclass
class LayerProbeConfig:
    """Settings of the linear probes fitted on intermediate activations.

    Attributes:
        runs_dir          : Root directory holding the run folders.
        run_filter        : Run names to restrict the analysis to; empty means all.
        split             : Dataset split the probe data is drawn from.
        device            : Torch device the forward channels run on.
        checkpoint_name   : Checkpoint loaded from each run.
        output_subdir     : Subdirectory of a run receiving the results.
        batch_size        : Patches per batch.
        max_batches       : Number of batches used to build the probe dataset.
        max_layers        : Number of layers probed.
        samples_per_batch : Activation vectors sampled per batch.
        ridge_lambda      : Ridge regularisation strength of the probe.
        n_folds           : Cross-validation folds used to score the probe.
        seed              : Random seed for sampling and fold assignment.
        figure_style      : Named matplotlib style applied to the figures.
    """

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
class CkaConfig:
    """Settings of the centred kernel alignment comparison between runs.

    Attributes:
        runs_dir          : Root directory holding the run folders.
        run_filter        : Run names to restrict the comparison to; empty means all.
        split             : Dataset split the activations are drawn from.
        device            : Torch device the forward channels run on.
        checkpoint_name   : Checkpoint loaded from each run.
        output_dir        : Directory receiving the similarity matrices and figures.
        batch_size        : Patches per batch.
        max_batches       : Number of batches used to collect activations.
        max_layers        : Number of layers compared per model.
        samples_per_batch : Activation vectors sampled per batch.
        sample_seed       : Random seed for activation sampling.
        figure_style      : Named matplotlib style applied to the figures.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    split           : str  = "test"
    device          : str  = "cpu"
    checkpoint_name : str  = "best_model.pt"
    output_dir      : Path = Path("results/cka")

    batch_size        : int = 4
    max_batches       : int = 4
    max_layers        : int = 16
    samples_per_batch : int = 512
    sample_seed       : int = 0

    figure_style : str = "report"


@dataclass
class LossLandscapeConfig:
    """Settings of the loss landscape mapped around a trained checkpoint.

    Attributes:
        runs_dir        : Root directory holding the run folders.
        run_filter      : Run names to restrict the analysis to; empty means all.
        split           : Dataset split the loss is evaluated on.
        device          : Torch device the evaluations run on.
        checkpoint_name : Checkpoint at the centre of the landscape.
        output_subdir   : Subdirectory of a run receiving the results.
        batch_size      : Patches per evaluation batch.
        max_batches     : Batches averaged per landscape point.
        span            : Half-width of the perturbation col in filter-normalised units.
        n_directions    : Random directions sampled for the 1D profiles.
        n_points_1d     : Points evaluated along each 1D direction.
        n_points_2d     : Points per axis of the 2D landscape grid.
        direction_seed  : Random seed for the perturbation directions.
        figure_style    : Named matplotlib style applied to the figures.
    """

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
    """Settings of the input stress tests probing prediction robustness.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Run names to restrict the analysis to; empty means all.
        split            : Dataset split the batches are drawn from.
        device           : Torch device the forward channels run on.
        checkpoint_name  : Checkpoint loaded from each run.
        output_subdir    : Subdirectory of a run receiving the results.
        batch_size       : Patches per batch.
        max_batches      : Number of batches stressed.
        noise_sigmas     : Additive noise standard deviations in normalised input units.
        noise_draws      : Noise realisations averaged per sigma.
        gain_factors     : Multiplicative amplitude gains applied to the input.
        shift_pixels     : Translations in pixels applied to test shift consistency.
        draws_per_count  : Random channel subsets drawn per retained-channel count.
        render_amp_floor : Amplitude below which a component is omitted from the renders.
        seed             : Random seed for the perturbations.
        figure_style     : Named matplotlib style applied to the figures.
    """

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
    render_amp_floor : float       = 0.0
    seed             : int         = 0

    figure_style : str = "report"


@dataclass
class TrainingEvolutionConfig:
    """Settings of the animation showing predictions evolving across checkpoints.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Run names to restrict the analysis to; empty means all.
        split            : Dataset split the probe patches are drawn from.
        device           : Torch device the forward channels run on.
        checkpoint_name  : Checkpoint used as the final animation frame.
        output_subdir    : Subdirectory of a run receiving the animation.
        window           : Side length in pixels of the animated neighbourhood.
        n_row_probes : Probe pixels along row.
        n_col_probes   : Probe pixels along col.
        fps              : Frames per second of the animation.
        figure_style     : Named matplotlib style applied to the figures.
    """

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
    """Settings of the export gathering publication figures from inference runs.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Run names to restrict the export to; empty means all.
        inference_subdir : Subdirectory of a run holding inference results.
        metrics_filename : File holding the per-run metrics.
        report_filename  : Report file name inside an inference directory.
        figures_subdir   : Subdirectory holding the figures of an inference run.
        patterns         : Glob patterns selecting the figures to export.
        output_dir       : Destination directory of the exported figure pack.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    inference_subdir : str = ""
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"
    figures_subdir   : str = "figures"

    patterns : List[str] = field(default_factory=lambda: ["pixel_maps/*", "param_scatter/*", "stratified/*", "profiles/*", "data_consistency/*"])

    output_dir : Path = Path("results/paper_figures")


@dataclass
class ResultTableEntryConfig:
    """Entry configuration exporting run metrics as CSV, LaTeX and markdown tables.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Substrings selecting the runs to export; empty means all.
        inference_subdir : Explicit inference stamp to read; empty picks each run's latest.
        metrics_filename : File holding the per-run metrics inside an inference stamp.
        metrics          : Metric keys forming the LaTeX and markdown columns; empty exports every scalar key.
        sort_by          : Metric key ordering the rows; empty keeps discovery order.
        ascending        : Whether the sort orders ascending.
        precision        : Significant digits used in the LaTeX and markdown cells.
        output_dir       : Destination directory of the written tables.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    inference_subdir : str = ""
    metrics_filename : str = "metrics.json"

    metrics   : tuple = ()
    sort_by   : str   = ""
    ascending : bool  = True
    precision : int   = 4

    output_dir : Path = Path("results/tables")
