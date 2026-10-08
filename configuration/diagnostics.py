"""Configuration dataclasses for the post-training diagnostic analyses.

Each dataclass drives one analysis entry point over trained runs: weight and
activation x-rays, receptive-field measurement, input attribution, attention
capture, linear layer probes, representation similarity, loss landscapes,
robustness stress tests, training evolution animations, TensorBoard exports,
report collection, result mailing of experiments, training runs, datasets and
parameter runs, and the paper figure pack.
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
    """Settings of the report gathering for a single experiment.

    Attributes:
        staging_dir        : Directory receiving this experiment's gathered files.
        comparison_dirname : Subdirectory of a sweep's trials directory holding the trial-comparison stamps.
        comparison_reports : Report filenames gathered from the newest trial-comparison stamp of a sweep or cross-validation run.
        inference_dirname  : Subdirectory of a run holding its inference results.
        report_filename    : Report file name inside an inference stamp.
        regenerate_reports : Whether the seed aggregates, the trial comparison and a cross-validation run's cross-fold report are rewritten even when present.
        embed_images       : Whether figures are embedded in the gathered reports.
    """

    staging_dir : Path = Path("results/mailed_reports")

    comparison_dirname : str       = "_comparison"
    comparison_reports : List[str] = field(default_factory=lambda: ["overview.md", "metrics_comparison.md"])

    inference_dirname : str = "inference"
    report_filename   : str = "report.md"

    regenerate_reports : bool = False
    embed_images       : bool = False


@dataclass
class ArtifactMailConfig:
    """Settings of the artifact gathering for a single dataset, parameter run or training run.

    The three path lists are read relative to the source directory and each may
    name a file or a directory, whose whole subtree is then taken. They split by
    what their absence means: a missing required path aborts the gathering, a
    missing optional path only warns, and analysis paths are absent whenever the
    analysis entry has not been run on that source directory yet. The two
    exclusion lists and the per-file cap then drop from what those paths hold
    whatever is too heavy to mail.

    Attributes:
        source_dir        : Dataset, parameter-run or training-run directory whose artifacts are packed.
        staging_dir       : Directory receiving this source's archives.
        label             : Identity of the source in the subject line and as the root folder inside the archives.
        required_paths    : Paths that must exist under the source directory; a missing one aborts.
        optional_paths    : Paths attached when present under the source directory; a missing one warns.
        analysis_paths    : Paths written by the analysis entry, attached when present.
        analysis_entry    : Name of the analysis entry that writes the analysis paths, quoted in the warning when none of them exist.
        excluded_paths    : Globs dropped from what the gathered paths hold, matched against a file's path relative to the source directory, against its parent directories and against the bare name of each of them.
        excluded_suffixes : File extensions dropped from what the gathered paths hold, matched case-insensitively.
        max_file_mb       : Upper bound of one gathered file in megabytes, above which it is dropped; zero turns the size rule off.
        max_attachment_mb : Upper bound of one archive in megabytes, measured on the uncompressed sources.
    """

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
        """Filename stem shared by the written archives, the label flattened to one path segment."""
        return self.label.replace("/", "__")

    @property
    def max_file_bytes(self) -> int:
        """Upper bound of one gathered file in bytes, zero when the size rule is off."""
        return int(self.max_file_mb * 1024 * 1024)

    @property
    def max_attachment_bytes(self) -> int:
        """Upper bound of one archive in bytes."""
        return int(self.max_attachment_mb * 1024 * 1024)


@dataclass
class ReportMailEntryConfig:
    """Entry configuration mailing one bundle per experiment, training run, dataset or parameter run.

    ``target`` picks which of the four sources the run mails, and only that
    target's root, filter and gathered paths are read; the mail settings and the
    attachment cap are shared by all of them.

    Attributes:
        target                 : Source the run mails, one of ``experiments``, ``runs``, ``datasets`` or ``param_extraction``.
        experiments_dir        : Path searched for experiments at or below it: a sweep directory of trial runs, a cross-validation run, a benchmark run, a sequence workspace holding several of them, or a whole runs tree.
        experiment_filter      : Experiment paths relative to that directory, or experiment names, to restrict the mailing to; empty means all.
        comparison_dirname     : Subdirectory of a sweep's trials directory holding the trial-comparison stamps.
        comparison_reports     : Report filenames gathered from the newest trial-comparison stamp of a sweep or cross-validation run.
        inference_dirname      : Subdirectory of a run holding its inference results.
        report_filename        : Report file name inside an inference stamp.
        regenerate_reports     : Whether every selected experiment gets its seed aggregates, trial comparison and cross-fold report rewritten, even when present.
        embed_images           : Whether figures are embedded in the gathered reports.
        runs_dir               : Path searched for training runs at or below it, or one run directory itself.
        run_filter             : Run paths relative to that root, or run names, to restrict the mailing to; empty means all.
        run_excluded_paths     : Globs of a run's content never mailed, matched against a file's path relative to the run, against its parent directories and against the bare name of each of them.
        run_excluded_suffixes  : File extensions of a run's content never mailed, the arrays and checkpoints by default.
        run_max_file_mb        : Upper bound of one mailed file of a run in megabytes, above which it is left out; zero mails every file the other two rules keep.
        datasets_dir           : Root directory holding one folder per generated dataset.
        dataset_filter         : Dataset folder names to restrict the mailing to; empty means all.
        dataset_required_paths : Dataset generation outputs that must exist in a dataset; a missing one aborts.
        dataset_optional_paths : Dataset generation outputs attached when present; a missing one warns.
        params_dir             : Root directory searched for parameter-extraction runs, the same dataset root.
        param_run_filter       : Parameter-run paths relative to that root; empty means all.
        param_required_paths   : Extraction outputs that must exist in a run; a missing one aborts.
        param_optional_paths   : Extraction outputs attached when present; a missing one warns.
        param_analysis_paths   : Outputs written by analyze_param_extraction, attached when present.
        staging_dir            : Directory receiving the gathered files, one ``<parent>/<source>`` subtree per source.
        max_attachment_mb      : Upper bound of one archive in megabytes, before the roughly one third that base64 adds on the wire.
        recipient              : Address every message is sent to.
        sender                 : Address the messages are sent from, also the SMTP login user.
        smtp_host              : SMTP server the messages are submitted to.
        smtp_port              : SMTP SSL port of the server.
        password_env           : Environment variable holding the SMTP password, never stored in the config.
        subject_prefix         : Prefix of every message subject, followed by the source label.
        dry_run                : Whether messages are only staged and logged instead of sent.
        TARGETS                : The accepted ``target`` values.
    """

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

    recipient      : str  = "victorvicentin42@gmail.com"
    sender         : str  = "victorvicentin42@gmail.com"
    smtp_host      : str  = "smtp.gmail.com"
    smtp_port      : int  = 465
    password_env   : str  = "REPORT_MAIL_PASSWORD"
    subject_prefix : str  = "ML-Template results"
    dry_run        : bool = False

    def to_config(self, label: str) -> ExperimentReportMailConfig:
        """Returns the gathering configuration of one experiment.

        Args:
            label : Mailed identity of the experiment, naming its subtree of the staging directory.

        Returns:
            Configuration staging that experiment's files under its labelled subtree of the staging directory.
        """
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
        """Returns the mailed identity of one training run, its path below the runs root.

        A run nested in a sweep or a fold tree is only told apart from its
        siblings by the directories above it, so the whole relative path is the
        label; a root that is the run itself has no relative path and is named
        by its parent and its own name instead.

        Args:
            run_dir : Training run folder whose content is mailed.
        """
        path = Path(run_dir)
        root = Path(self.runs_dir)

        if path == root:
            return f"{path.parent.name}/{path.name}"

        return path.relative_to(root).as_posix()

    def to_run_config(self, run_dir: Path) -> ArtifactMailConfig:
        """Returns the artifact gathering configuration for one training run directory.

        The whole run tree is gathered, so the run's content is described by
        what is left out rather than by what is named: the excluded paths, the
        excluded suffixes and the per-file cap drop the checkpoints, the cubes
        and any other file too heavy to travel by mail.

        Args:
            run_dir : Training run folder whose content is gathered.

        Returns:
            Configuration staging that run's archives under its labelled subtree of the staging directory.
        """
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
        """Returns the mailed identity of one dataset, prefixed by its root directory.

        Args:
            dataset_dir : Generated dataset folder whose artifacts are mailed.
        """
        path = Path(dataset_dir)
        return f"{path.parent.name}/{path.name}"

    def to_dataset_config(self, dataset_dir: Path) -> ArtifactMailConfig:
        """Returns the artifact gathering configuration for one dataset directory.

        Args:
            dataset_dir : Generated dataset folder whose artifacts are gathered.

        Returns:
            Configuration staging that dataset's archives under its labelled subtree of the staging directory.
        """
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
        """Returns the mailed identity of one parameter run, prefixed by its dataset.

        Parameter runs live at ``<dataset>/params/<run>``, so the grandparent
        rather than the literal ``params`` directory is what tells two runs of
        different datasets apart in the subject line and on the staging tree.

        Args:
            run_dir : Parameter-extraction run folder whose artifacts are mailed.
        """
        path = Path(run_dir)
        return f"{path.parent.parent.name}/{path.name}"

    def to_param_config(self, run_dir: Path) -> ArtifactMailConfig:
        """Returns the artifact gathering configuration for one parameter run directory.

        Args:
            run_dir : Parameter-extraction run folder whose artifacts are gathered.

        Returns:
            Configuration staging that run's archives under its labelled subtree of the staging directory.
        """
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
        split            : Data split the packed inference stamp of every run must have been scored on.
        inference_subdir : Subdirectory of a run holding inference results.
        metrics_filename : File holding the per-run metrics.
        report_filename  : Report file name inside an inference directory.
        figures_subdir   : Subdirectory holding the figures of an inference run.
        patterns         : Glob patterns selecting the figures to export.
        output_dir       : Destination directory of the exported figure pack.
    """

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
    """Entry configuration exporting run metrics as CSV, LaTeX and markdown tables.

    Attributes:
        runs_dir         : Root directory holding the run folders.
        run_filter       : Substrings selecting the runs to export; empty means all.
        split            : Data split the inference stamp of every run must have been scored on.
        inference_subdir : Explicit inference stamp to read; empty picks each run's newest completed stamp of the split.
        metrics_filename : File holding the per-run metrics inside an inference stamp.
        metrics          : Metric keys forming the LaTeX and markdown columns; empty exports every scalar key.
        sort_by          : Metric key ordering the rows; empty keeps discovery order.
        ascending        : Whether the sort orders ascending.
        precision        : Significant digits used in the LaTeX and markdown cells.
        output_dir       : Destination directory of the written tables.
    """

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
    """Entry configuration exporting trained backbone checkpoints as ONNX graphs.

    Attributes:
        runs_dir      : Root directory holding the run folders.
        run_filter    : Substrings selecting the runs to export; empty means all.
        opset         : ONNX opset version the graph targets.
        dynamic_batch : Whether the exported input and output keep a dynamic batch axis.
    """

    runs_dir   : Path      = Path("runs/backbone")
    run_filter : List[str] = field(default_factory=list)

    opset         : int  = 18
    dynamic_batch : bool = True
