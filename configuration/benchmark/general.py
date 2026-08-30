"""Configuration of the model benchmark experiment and its per-stage settings.

A benchmark trains every model of a family under matched conditions: an
optional parameter-count matching stage, a VRAM-bounded batch size probe, the
training queue itself, inference over the trained runs, and a comparison
report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib     import Path

from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import LossScaleProbeConfig, default_curriculum
from configuration.training.general.loss        import LossConfig
from configuration.training.general.pretraining import PretrainConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.training.general.runtime     import OverfitCheckConfig
from configuration.training.jepa                import JepaCouplingConfig
from configuration.training.profile_autoencoder import ProfileAeLossConfig


@dataclass
class MaxBatchConfig:
    """Settings of the probe that finds the largest batch fitting the VRAM budget.

    Attributes:
        vram_budget_gb : Memory budget per GPU in gigabytes.
        max_batch      : Upper bound on the batch size the probe may report.
        measure_steps  : Training steps timed per candidate batch size.
        seed           : Random seed for the probe workload.
    """

    vram_budget_gb : float = PretrainConfig.vram_budget_gb
    max_batch      : int   = PretrainConfig.max_batch
    measure_steps  : int   = PretrainConfig.measure_steps
    seed           : int   = PretrainConfig.seed


@dataclass
class FindLrConfig:
    """Settings of the LR range test run inside each model's max-batch probe job.

    The test rewinds the probe trainer to one captured state before every rung,
    trains it there at the probed batch size and records the minimum-loss
    multiplier in the probe result, which the training worker folds into
    ``training.lr_multiplier`` per model.

    Attributes:
        enabled           : Whether the range test runs after each passing batch probe.
        scale_min         : Smallest learning-rate multiplier the range test starts from.
        scale_max         : Largest learning-rate multiplier the range test ramps to.
        rungs             : Multiplier rungs of the ramp, each measured from the same rewound trainer state.
        rung_steps        : Optimizer steps taken at each rung, mirroring the real step including accumulation and clipping.
        probe_batches     : Held-out batches measured at every rung, the same batches throughout the ramp.
        divergence_factor : Multiple of the best rung loss that ends the ramp.
        safety_factor     : Divisor applied to the minimum-loss multiplier.
        min_improvement   : Smallest relative drop below the first rung that counts as a response.
        noise_sigmas      : Standard errors of the rung measurement the drop must also clear.
        seed              : Seed of the shuffle generator drawing the range-test batches.
    """

    enabled           : bool  = False
    scale_min         : float = PretrainConfig.lr_scale_min
    scale_max         : float = PretrainConfig.lr_scale_max
    rungs             : int   = PretrainConfig.lr_rungs
    rung_steps        : int   = PretrainConfig.lr_rung_steps
    probe_batches     : int   = PretrainConfig.lr_probe_batches
    divergence_factor : float = PretrainConfig.lr_divergence_factor
    safety_factor     : float = PretrainConfig.lr_safety_factor
    min_improvement   : float = PretrainConfig.lr_min_improvement
    noise_sigmas      : float = PretrainConfig.lr_noise_sigmas
    seed              : int   = PretrainConfig.seed


@dataclass
class SizeMatchConfig:
    """Settings of the stage that rescales every model to a matched parameter count.

    Attributes:
        reference_model : Model whose parameter count all others are matched to.
        tolerance       : Accepted relative deviation from the reference count.
        max_iterations  : Maximum bisection steps per model.
        scale_low       : Lower bound of the initial width-scale bracket.
        scale_high      : Upper bound of the initial width-scale bracket.
        scale_ceiling   : Hard cap on the width scale during bracket expansion.
        in_channels     : Input channel count used when instantiating candidates.
        locked_params   : Config fields excluded from width scaling.
    """

    reference_model : str             = "unet"
    tolerance       : float           = 0.05
    max_iterations  : int             = 100
    scale_low       : float           = 0.05
    scale_high      : float           = 8.0
    scale_ceiling   : float           = 64.0
    in_channels     : int             = 9
    locked_params   : tuple[str, ...] = ("embedding_dim", "embedding_dims")


@dataclass
class InferenceQueueConfig:
    """Settings of the inference pass run over every trained benchmark model.

    Attributes:
        split                    : Dataset split inference runs on.
        checkpoint_name          : Checkpoint file loaded from each run directory.
        batch_size               : Inference batch size; ``None`` reuses the training value.
        num_workers              : Data loader workers used during inference.
        cpu_workers              : Worker processes used for CPU-side post-processing.
        gif_workers              : Worker processes rendering animation frames.
        seed                     : Seed used for the random profile and slice selections.
        save_plots               : Whether per-run diagnostic plots are written.
        save_animations          : Whether GIF animations are written.
        save_cubes               : Whether reconstructed tomographic cubes are saved.
        stitch_window            : Window applied when stitching overlapping patches.
        cube_dtype               : NumPy dtype name used when writing cubes.
        render_amp_floor         : Amplitude below which components are hidden in renders.
        active_amp_thr           : Physical amplitude at or above which a Gaussian slot counts as active in the Hungarian slot matching, metrics, failure modes, stratified errors, normalization audit and report figures.
        compute_flip_consistency : Whether flip-equivariance metrics run.
        compute_stratified       : Whether metrics are stratified by scene strata.
        extract_curve_params     : Whether Gaussians are refitted to predicted curves.
        extract_prominence_frac  : Peak prominence fraction used in curve fitting.
        n_best_profiles          : Best-reconstructed profiles plotted per run.
        n_worst_profiles         : Worst-reconstructed profiles plotted per run.
        n_random_profiles        : Randomly drawn profiles plotted per run.
        profile_seed             : Seed for the random profile draw.
        n_col_slices             : Cube slices plotted along the columns.
        n_row_slices             : Cube slices plotted along the rows.
        n_z_slices               : Cube slices plotted along z.
        gif_axes                 : Cube axes animated in the GIF exports.
        gif_fps                  : Frames per second of the GIF exports.
        gif_max_frames           : Maximum frames per GIF export.
        gif_dpi                  : Rendering resolution of animation frames.
        cmap_intensity           : Colormap used for intensity imagery.
        cmap_error               : Colormap used for error imagery.
        normalize_intensity      : Whether intensity maps are normalized before display.
        fig_dpi                  : Figure resolution used on screen.
        save_dpi                 : Figure resolution used when saving.
        figure_style             : Named matplotlib style applied to the figures.
    """

    split           : str        = "test"
    checkpoint_name : str        = "best_model.pt"
    batch_size      : int | None = None
    num_workers     : int        = 4
    cpu_workers     : int        = 16
    gif_workers     : int        = 40
    seed            : int        = 0

    save_plots       : bool  = False
    save_animations  : bool  = False
    save_cubes       : bool  = False
    stitch_window    : str   = "hann"
    cube_dtype       : str   = "float32"
    render_amp_floor : float = 0.0
    active_amp_thr   : float = 1e-4

    compute_flip_consistency : bool = False
    compute_stratified       : bool = True

    extract_curve_params    : bool  = True
    extract_prominence_frac : float = 0.05

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12
    profile_seed      : int = 0

    n_col_slices : int = 5
    n_row_slices : int = 5
    n_z_slices   : int = 5

    gif_axes       : list[str] = field(default_factory=lambda: ["z"])
    gif_fps        : int       = 12
    gif_max_frames : int       = 150
    gif_dpi        : int       = 110

    cmap_intensity      : str  = "jet"
    cmap_error          : str  = "magma"
    normalize_intensity : bool = True
    fig_dpi             : int  = 150
    save_dpi            : int  = 300
    figure_style        : str  = "report"


@dataclass
class ComparisonReportConfig:
    """Settings of the markdown report comparing the benchmarked runs.

    Attributes:
        embed_images : Whether figures are embedded in the report instead of linked.
    """

    embed_images: bool = False


def _default_ae_loss():
    """Returns the default profile autoencoder loss configuration."""
    return ProfileAeLossConfig()


def _default_base_loss():
    """Returns the complete-stage loss configuration of the standard curriculum."""
    return default_curriculum().complete


@dataclass
class BenchmarkConfig:
    """Top-level configuration of a model benchmark run.

    Attributes:
        training_type         : Pipeline family being benchmarked.
        stage                 : Which staged part of the benchmark to execute.
        paths                 : Dataset and run path configuration.
        max_batch             : Batch-size probe settings.
        find_lr               : LR range test settings, run inside the batch-probe jobs.
        size_match            : Parameter-count matching settings.
        training              : Training queue settings shared by all benchmarked runs.
        inference             : Inference queue settings.
        comparison            : Comparison report settings.
        input                 : Input representation and channel selection.
        geometry              : SAR acquisition geometry.
        normalization         : Input and target normalisation settings.
        augmentation          : Training-time augmentation settings.
        loss                  : Loss configuration applied to every benchmarked run.
        probe                 : Loss-scale probe settings.
        overfit_check         : Pre-training overfit sanity gate settings.
        ae_loss               : Loss configuration used when benchmarking autoencoders.
        jepa                  : JEPA coupling settings used when benchmarking JEPA runs.
        pixel_subsample       : Fraction of pixels retained from the dataset.
        keep_empty_frac       : Fraction of empty-profile pixels retained.
        gpus                  : GPU indices available to the run queue.
        gpus_file             : Path to a file that can resize the GPU pool while running.
        heads                 : Output heads benchmarked for each backbone.
        skip_models           : Model names excluded from the benchmark.
        run_tag               : Optional tag appended to generated run names.
        resume                : Whether completed units are skipped on a repeated launch.
        infer_after           : Whether inference runs once training finishes.
        seed                  : Base random seed.
        seeds                 : Seeds trained per benchmarked configuration.
        poll_interval_s       : Scheduler polling interval in seconds.
        sweep_loss_components : Loss terms swept across benchmark units.
    """

    training_type: str = "backbone"
    stage        : str = "full"

    paths      : RunPathsConfig         = field(default_factory=RunPathsConfig)
    max_batch  : MaxBatchConfig         = field(default_factory=MaxBatchConfig)
    find_lr    : FindLrConfig           = field(default_factory=FindLrConfig)
    size_match : SizeMatchConfig        = field(default_factory=SizeMatchConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)
    loss          : LossConfig           = field(default_factory=_default_base_loss)
    probe         : LossScaleProbeConfig = field(default_factory=LossScaleProbeConfig)
    overfit_check : OverfitCheckConfig   = field(default_factory=OverfitCheckConfig)

    ae_loss         : object             = field(default_factory=_default_ae_loss)
    jepa            : JepaCouplingConfig = field(default_factory=JepaCouplingConfig)
    pixel_subsample : float              = 1.0
    keep_empty_frac : float              = 0.05

    gpus            : list[int]  = field(default_factory=lambda: [0])
    gpus_file       : str        = ""
    heads           : list[str]  = field(default_factory=lambda: ["conv"])
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    infer_after     : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    sweep_loss_components: list[str] = field(default_factory=lambda: ["param_l1"])

    def runs_size_match(self) -> bool:
        """Returns whether the parameter-count matching stage applies to this run."""
        return self.training_type == "backbone"

    def runs_inference(self) -> bool:
        """Returns whether inference follows training for this benchmark family."""
        return self.infer_after and self.training_type in ("backbone", "jepa")

    def size_match_settings(self) -> dict:
        """Returns the size-matching stage arguments as a serialisable mapping.

        Returns:
            Reference model, input channels, locked fields and patch size.
        """
        return {
            "reference_model" : self.size_match.reference_model,
            "in_channels"     : self.size_match.in_channels,
            "locked_params"   : list(self.size_match.locked_params),
            "patch_size"      : list(self.training.patch_size),
        }

    def max_batch_settings(self) -> dict:
        """Returns the batch-size probe arguments as a serialisable mapping.

        The key covers every input that defines the base learning rates the
        probe measured its multiplier against, not just the range-test settings:
        a recorded multiplier is meaningful only relative to
        ``training.lr_multiplier``, ``scale_lr_with_batch`` and
        ``lr_reference_batch_size``, so a cached record must not survive a
        change to any of them.

        Returns:
            VRAM budget, batch ceiling, measurement steps, probe seed, patch
            size, input channel count, the LR range-test settings and the
            training fields the probed multiplier is relative to.
        """
        return {
            "budget_gb"               : self.max_batch.vram_budget_gb,
            "ceiling"                 : self.max_batch.max_batch,
            "measure_steps"           : self.max_batch.measure_steps,
            "probe_seed"              : self.max_batch.seed,
            "patch_size"              : list(self.training.patch_size),
            "in_channels"             : self.size_match.in_channels,
            "find_lr"                 : asdict(self.find_lr),
            "lr_multiplier"           : self.training.lr_multiplier,
            "scale_lr_with_batch"     : self.training.scale_lr_with_batch,
            "lr_reference_batch_size" : self.training.lr_reference_batch_size,
        }
