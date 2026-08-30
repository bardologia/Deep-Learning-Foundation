"""Derivation of dataset, trainer and inference configs from a benchmark config.

Turns the flat benchmark entry configuration into the structured dataset, trainer
and inference configurations that the individual pipelines consume.
"""

from __future__ import annotations

from pathlib import Path

from configuration.benchmark            import BenchmarkConfig
from configuration.dataset             import DatasetConfig, InputConfig, OutputConfig, PatchConfig, SplitRegions
from configuration.inference           import InferenceConfig
from configuration.training            import default_curriculum, EarlyStoppingConfig, GradientClipperConfig, MemoryConfig, OptimizerConfig, SchedulerConfig, WarmupConfig, IOConfig, TrainingLoopConfig, BackboneTrainerConfig
from configuration.dataset.gaussian import GaussianConfig
from tools.data.io                     import FileIO
from tools.data.regions                import CropRegion


class ConfigFactory:
    """Builds the pipeline configurations implied by one benchmark configuration.

    Attributes:
        config : Benchmark entry configuration all derived configs come from.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Stores the benchmark configuration used as the single source of truth."""
        self.config = config

    def _secondary_labels(self) -> tuple | None:
        """Returns the configured secondary channel labels as a tuple, or None when unset."""
        labels = self.config.paths.secondary_labels
        return tuple(labels) if labels else None

    def global_crop(self) -> CropRegion:
        """Returns the global row/col crop recorded by the preprocessing run."""
        layout_path = Path(self.config.paths.dataset_path) / "data" / "dataset.json"
        layout      = FileIO.load_json(layout_path)

        return CropRegion(*layout["global_crop"])

    def benchmark_input_config(self) -> InputConfig:
        """Returns the full-stack input configuration used by every benchmark model."""
        return InputConfig.full_stack()

    def _output_config(self) -> OutputConfig:
        """Returns the output configuration with per-target normalization strategies."""
        norm = self.config.normalization
        return OutputConfig(output_strategies={key: norm.strategy("output", key) for key in ("out/amp", "out/mu", "out/sigma")})

    def split_regions(self) -> SplitRegions:
        """Returns train/val/test crops: configured row spans over the full range extent."""
        crop     = self.global_crop()
        training = self.config.training

        return SplitRegions(
            train = CropRegion(training.train_rows[0], training.train_rows[1], crop.col_start, crop.col_end),
            val   = CropRegion(training.val_rows[0],   training.val_rows[1],   crop.col_start, crop.col_end),
            test  = CropRegion(training.test_rows[0],  training.test_rows[1],  crop.col_start, crop.col_end),
        )

    def training_dataset_config(self) -> DatasetConfig:
        """Returns the patch dataset configuration used for benchmark training."""
        training = self.config.training

        return DatasetConfig(
            preprocessing_run_directory = self.config.paths.dataset_path,
            parameters_path             = self.config.paths.parameters_path,
            split_regions               = self.split_regions(),
            secondary_labels            = self._secondary_labels(),
            patch                       = PatchConfig(size=training.patch_size, stride=training.patch_stride, use_symmetric_padding=training.use_symmetric_padding),
            input_config                = self.benchmark_input_config(),
            output_config               = self._output_config(),
            normalization               = self.config.normalization,
            augmentation                = self.config.augmentation,
            batch_size                  = training.batch_size,
            num_workers                 = training.num_workers,
            prefetch_factor             = training.prefetch_factor,
            shuffle_train               = True,
            pin_memory                  = True,
            fingerprint_dataset         = self.config.paths.fingerprint_dataset,
        )

    def gaussian_config(self) -> GaussianConfig:
        """Returns the Gaussian mixture configuration read from the dataset parameters."""
        return GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path)

    def _base_trainer_kwargs(self, logdir: Path) -> dict:
        """Returns the trainer arguments shared by the backbone and unrolled families.

        Args:
            logdir : Run directory the trainer writes its outputs to.
        """
        training = self.config.training
        return {
            "gaussian"         : self.gaussian_config(),
            "gradient_clipper" : GradientClipperConfig(clip_mode=training.clip_mode.value, max_grad_norm=training.max_grad_norm, adaptive_window=training.clip_adaptive_window, adaptive_percentile=training.clip_adaptive_percentile, adaptive_mean_std_k=training.clip_adaptive_mean_std_k),
            "io"               : IOConfig(logdir=str(logdir)),
        }

    def training_trainer_config(self, logdir: Path) -> BackboneTrainerConfig:
        """Returns the backbone trainer configuration for one benchmark run.

        Args:
            logdir : Run directory the trainer writes its outputs to.

        Returns:
            Trainer configuration whose learning rate is linearly rescaled by the
            ratio of the batch size to the reference batch size when enabled, and
            whose scheduler spans the configured epoch budget.
        """
        training         = self.config.training
        scheduler_epochs = training.scheduler_epochs if training.scheduler_epochs is not None else training.epochs
        lr_scale         = (training.batch_size / training.lr_reference_batch_size if training.scale_lr_with_batch else 1.0) * training.lr_multiplier

        return BackboneTrainerConfig(
            **self._base_trainer_kwargs(logdir),

            early_stopping = EarlyStoppingConfig(patience=training.early_stop_patience, min_delta=training.early_stop_min_delta, restore_best=True),
            warmup         = WarmupConfig(warmup_fraction=training.warmup_fraction, warmup_start_factor=0.1, warmup_enabled=training.warmup_enabled, warmup_mode=training.warmup_mode.value, warmup_poly_power=training.warmup_poly_power),
            scheduler      = SchedulerConfig(type=training.scheduler_type.value, epochs=scheduler_epochs, eta_min=training.eta_min, step_size=training.scheduler_step_size, gamma=training.scheduler_gamma, power=training.scheduler_power),
            optimizer      = OptimizerConfig(type=training.optimizer_type.value, betas=(0.9, 0.999), eps=1e-8, momentum=training.momentum, nesterov=training.nesterov, lr_scale=lr_scale),
            memory         = MemoryConfig(reserve_vram=training.reserve_vram, vram_keep_free_gb=training.vram_keep_free_gb),

            training = TrainingLoopConfig(
                epochs                      = training.epochs,
                validation_frequency        = training.validation_frequency,
                use_amp                     = training.use_amp,
                compile_model               = training.compile_model,
                compile_mode                = training.compile_mode,
                channels_last               = training.channels_last,
                activation_checkpointing    = tuple(training.activation_checkpointing),
                gradient_accumulation_steps = training.gradient_accumulation_steps,
                log_all_losses              = training.log_all_losses,
                log_debug                   = training.log_debug,
                abort_on_nonfinite_loss     = training.abort_on_nonfinite_loss,
                use_ema                     = training.use_ema,
                ema_decay                   = training.ema_decay,
                ema_warmup                  = training.ema_warmup,
                use_swa                     = training.use_swa,
                swa_start_fraction          = training.swa_start_fraction,
                resume                      = training.resume,
                snapshot_every_n_epochs     = training.snapshot_every_n_epochs,
            ),

            curriculum = default_curriculum(),
        )

    def inference_config(self, run_directory: Path) -> InferenceConfig:
        """Returns the inference configuration for scoring one trained run.

        Args:
            run_directory : Training run whose checkpoint is scored.
        """
        inference = self.config.inference

        return InferenceConfig(
            run_directory            = run_directory,
            output_subdir            = None,
            device                   = "cuda",
            seed                     = inference.seed,
            checkpoint_name          = inference.checkpoint_name,
            split                    = inference.split,
            batch_size               = inference.batch_size,
            num_workers              = inference.num_workers,
            stitch_window            = inference.stitch_window,
            cube_dtype               = inference.cube_dtype,
            render_amp_floor         = inference.render_amp_floor,
            active_amp_thr           = inference.active_amp_thr,
            save_cubes               = inference.save_cubes,
            save_plots               = inference.save_plots,
            save_animations          = inference.save_animations,
            compute_flip_consistency = inference.compute_flip_consistency,
            compute_stratified       = inference.compute_stratified,
            extract_curve_params     = inference.extract_curve_params,
            extract_prominence_frac  = inference.extract_prominence_frac,
            n_best_profiles          = inference.n_best_profiles,
            n_worst_profiles         = inference.n_worst_profiles,
            n_random_profiles        = inference.n_random_profiles,
            profile_seed             = inference.profile_seed,
            n_col_slices           = inference.n_col_slices,
            n_row_slices         = inference.n_row_slices,
            n_z_slices       = inference.n_z_slices,
            gif_axes                 = list(inference.gif_axes),
            gif_fps                  = inference.gif_fps,
            gif_max_frames           = inference.gif_max_frames,
            gif_dpi                  = inference.gif_dpi,
            cmap_intensity           = inference.cmap_intensity,
            cmap_error               = inference.cmap_error,
            normalize_intensity      = inference.normalize_intensity,
            fig_dpi                  = inference.fig_dpi,
            save_dpi                 = inference.save_dpi,
            figure_style             = inference.figure_style,
            cpu_workers              = inference.cpu_workers,
            gif_workers              = inference.gif_workers,
        )
