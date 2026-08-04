from __future__ import annotations

from pathlib import Path

from configuration.benchmark            import BenchmarkConfig
from configuration.dataset             import DatasetConfig, InputConfig, OutputConfig, PatchConfig, SplitRegions
from configuration.inference           import InferenceConfig
from configuration.training            import default_curriculum, EarlyStoppingConfig, GradientClipperConfig, MemoryConfig, OptimizerConfig, SchedulerConfig, WarmupConfig, IOConfig, TrainingLoopConfig, BackboneTrainerConfig
from configuration.sar.gaussian_config import GaussianConfig
from configuration.sar.geometry_config import GeometryConfig
from tools.data.io                     import FileIO
from tools.data.regions                import CropRegion


class ConfigFactory:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def _secondary_labels(self) -> tuple | None:
        labels = self.config.paths.secondary_labels
        return tuple(labels) if labels else None

    def global_crop(self) -> CropRegion:
        layout_path = Path(self.config.paths.dataset_path) / "data" / "dataset.json"
        layout      = FileIO.load_json(layout_path)

        return CropRegion(*layout["global_crop"])

    def benchmark_input_config(self) -> InputConfig:
        return InputConfig.full_stack()

    def _output_config(self) -> OutputConfig:
        norm = self.config.normalization
        return OutputConfig(output_strategies={key: norm.strategy("output", key) for key in ("out/amp", "out/mu", "out/sigma")})

    def split_regions(self) -> SplitRegions:
        crop     = self.global_crop()
        training = self.config.training

        return SplitRegions(
            train = CropRegion(training.train_azimuth[0], training.train_azimuth[1], crop.range_start, crop.range_end),
            val   = CropRegion(training.val_azimuth[0],   training.val_azimuth[1],   crop.range_start, crop.range_end),
            test  = CropRegion(training.test_azimuth[0],  training.test_azimuth[1],  crop.range_start, crop.range_end),
        )

    def training_dataset_config(self) -> DatasetConfig:
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
        )

    def gaussian_config(self) -> GaussianConfig:
        return GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path)

    def _base_trainer_kwargs(self, logdir: Path) -> dict:
        training = self.config.training
        return {
            "gaussian"         : self.gaussian_config(),
            "geometry"         : GeometryConfig().resolved(self.config.paths.dataset_path, secondary_labels=self._secondary_labels()),
            "gradient_clipper" : GradientClipperConfig(clip_mode=training.clip_mode.value, max_grad_norm=training.max_grad_norm, adaptive_window=training.clip_adaptive_window, adaptive_percentile=training.clip_adaptive_percentile, adaptive_mean_std_k=training.clip_adaptive_mean_std_k),
            "io"               : IOConfig(logdir=str(logdir)),
        }

    def training_trainer_config(self, logdir: Path) -> BackboneTrainerConfig:
        training         = self.config.training
        scheduler_epochs = training.scheduler_epochs if training.scheduler_epochs is not None else training.epochs
        lr_scale         = training.batch_size / training.lr_reference_batch_size if training.scale_lr_with_batch else 1.0

        return BackboneTrainerConfig(
            **self._base_trainer_kwargs(logdir),

            early_stopping = EarlyStoppingConfig(patience=training.early_stop_patience, min_delta=training.early_stop_min_delta, restore_best=True),
            warmup         = WarmupConfig(warmup_steps=training.warmup_steps, warmup_start_factor=0.1, warmup_enabled=training.warmup_enabled, warmup_mode=training.warmup_mode.value, warmup_poly_power=training.warmup_poly_power),
            scheduler      = SchedulerConfig(type=training.scheduler_type.value, epochs=scheduler_epochs, eta_min=training.eta_min, step_size=training.scheduler_step_size, gamma=training.scheduler_gamma, power=training.scheduler_power),
            optimizer      = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8, lr_scale=lr_scale),
            memory         = MemoryConfig(reserve_vram=training.reserve_vram, vram_keep_free_gb=training.vram_keep_free_gb),

            training = TrainingLoopConfig(
                epochs                      = training.epochs,
                validation_frequency        = training.validation_frequency,
                use_amp                     = training.use_amp,
                gradient_accumulation_steps = training.gradient_accumulation_steps,
                log_all_losses              = training.log_all_losses,
                log_debug                   = training.log_debug,
                abort_on_nonfinite_loss     = training.abort_on_nonfinite_loss,
                use_ema                     = training.use_ema,
                ema_decay                   = training.ema_decay,
                resume                      = training.resume,
                snapshot_every_n_epochs     = training.snapshot_every_n_epochs,
            ),

            curriculum = default_curriculum(),
        )

    def inference_config(self, run_directory: Path) -> InferenceConfig:
        inference = self.config.inference

        return InferenceConfig(
            run_directory      = run_directory,
            output_subdir      = None,
            device             = "cuda",
            checkpoint_name    = inference.checkpoint_name,
            split              = inference.split,
            batch_size         = inference.batch_size,
            num_workers        = inference.num_workers,
            stitch_window      = inference.stitch_window,
            save_cubes         = inference.save_cubes,
            save_plots         = inference.save_plots,
            save_animations    = inference.save_animations,
            compute_reduced    = inference.compute_reduced,
            n_best_profiles    = inference.n_best_profiles,
            n_worst_profiles   = inference.n_worst_profiles,
            n_random_profiles  = inference.n_random_profiles,
            n_range_slices     = inference.n_range_slices,
            n_azimuth_slices   = inference.n_azimuth_slices,
            n_elevation_slices = inference.n_elevation_slices,
            gif_axes           = list(inference.gif_axes),
            gif_fps            = inference.gif_fps,
            gif_max_frames     = inference.gif_max_frames,
            cpu_workers        = inference.cpu_workers,
        )
