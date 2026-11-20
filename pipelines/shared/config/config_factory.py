
from __future__ import annotations

from pathlib import Path

from configuration.benchmark                import BenchmarkConfig
from configuration.dataset                  import DatasetConfig, PatchConfig
from configuration.inference                import InferenceConfig
from configuration.training                 import EarlyStoppingConfig, GradientClipperConfig, MemoryConfig, OptimizerConfig, SchedulerConfig, WarmupConfig, IOConfig, TrainingLoopConfig, BackboneTrainerConfig
from pipelines.shared.dataset.dataset_files import DatasetFiles
from tools.data.regions                     import SplitRegions


class ConfigFactory:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def split_regions(self) -> SplitRegions:
        return DatasetFiles.read(self.config.paths.dataset_path).splits

    def training_dataset_config(self) -> DatasetConfig:
        training = self.config.training

        return DatasetConfig(
            dataset_directory   = self.config.paths.dataset_path,
            split_regions       = self.split_regions(),
            patch               = PatchConfig(size=training.patch_size, stride=training.patch_stride, use_symmetric_padding=training.use_symmetric_padding),
            normalization       = self.config.normalization,
            augmentation        = self.config.augmentation,
            batch_size          = training.batch_size,
            num_workers         = training.num_workers,
            prefetch_factor     = training.prefetch_factor,
            shuffle_train       = True,
            pin_memory          = True,
            fingerprint_dataset = self.config.paths.fingerprint_dataset,
        )

    def _base_trainer_kwargs(self, logdir: Path) -> dict:
        training = self.config.training
        return {
            "gradient_clipper" : GradientClipperConfig(clip_mode=training.clip_mode.value, max_grad_norm=training.max_grad_norm, adaptive_window=training.clip_adaptive_window, adaptive_percentile=training.clip_adaptive_percentile, adaptive_mean_std_k=training.clip_adaptive_mean_std_k),
            "io"               : IOConfig(logdir=str(logdir)),
        }

    def training_trainer_config(self, logdir: Path) -> BackboneTrainerConfig:
        training         = self.config.training
        scheduler_epochs = training.scheduler_epochs if training.scheduler_epochs is not None else training.epochs
        lr_scale         = training.lr_scale()

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

            loss = self.config.loss,
        )

    def inference_config(self, run_directory: Path) -> InferenceConfig:
        inference = self.config.inference

        return InferenceConfig(
            run_directory   = run_directory,
            output_subdir   = None,
            device          = "cuda",
            seed            = inference.seed,
            checkpoint_name = inference.checkpoint_name,
            split           = inference.split,
            batch_size      = inference.batch_size,
            num_workers     = inference.num_workers,
            stitch_window   = inference.stitch_window,
            save_maps       = inference.save_maps,
            save_plots      = inference.save_plots,
            cmap            = inference.cmap,
            cmap_error      = inference.cmap_error,
            fig_dpi         = inference.fig_dpi,
            save_dpi        = inference.save_dpi,
            figure_style    = inference.figure_style,
        )
