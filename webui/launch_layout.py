
from __future__ import annotations

import copy
from collections import Counter


class LayoutError(Exception):
    pass


class GpuWidgets:

    ONE  = {"kind": "gpu"}
    MANY = {"kind": "gpu", "multi": True}


class InferenceEssentials:

    @staticmethod
    def for_block(split_path: str) -> list:
        return [
            "runs_dir",
            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "splitFrom": split_path, "pendingButton": True}},
            "skip_completed",
            {"path": "gpus", "widget": GpuWidgets.MANY},
            "poll_interval_s",
        ]


class LaunchLayout:

    GPU_ONE  = GpuWidgets.ONE
    GPU_MANY = GpuWidgets.MANY

    NUM_EPOCHS   = {"kind": "number", "int": True, "min": 1, "max": 1000, "presets": [10, 50, 60, 100, 200, 500]}
    NUM_BATCH    = {"kind": "number", "int": True, "min": 1, "max": 4096, "presets": [32, 64, 128, 256, 512, 1024]}
    NUM_WORKERS  = {"kind": "number", "int": True, "min": 0, "max": 64, "presets": [0, 2, 4, 8, 16, 32]}
    NUM_PREFETCH = {"kind": "number", "int": True, "min": 1, "max": 32, "presets": [2, 4, 8, 16]}
    NUM_ACCUM    = {"kind": "number", "int": True, "min": 1, "max": 64, "presets": [1, 2, 4, 8, 16]}
    NUM_PATIENCE = {"kind": "number", "int": True, "min": 1, "max": 200, "presets": [5, 10, 20, 30, 50, 100]}
    NUM_SEED     = {"kind": "number", "int": True, "min": 0, "max": 9999, "presets": [0, 1, 42, 123, 2024]}
    TUNE_WORKERS = {"kind": "number", "int": True, "min": 0, "max": 32, "presets": [0, 2, 4, 8]}
    NUM_PLATEAU  = {"kind": "number", "int": True, "min": 0, "max": 10000, "presets": [0, 50, 100, 200, 400]}
    NUM_DELTA    = {"kind": "number", "min": 0, "max": 1, "step": 0.001, "presets": [0, 0.001, 0.005, 0.01, 0.02]}
    NUM_HOURS    = {"kind": "number", "min": 0, "max": 240, "step": 0.5, "presets": [0, 12, 24, 48, 72]}
    NUM_FREQ     = {"kind": "number", "int": True, "min": 1, "max": 50, "presets": [1, 2, 5, 10]}
    NUM_ETA_MIN  = {"kind": "number", "log": True, "min": 1e-9, "max": 1e-2, "presets": [0, 1e-7, 1e-6, 1e-5]}
    NUM_FRACTION = {"kind": "number", "min": 0, "max": 1, "step": 0.01, "presets": [0.05, 0.1, 0.25, 0.5, 1.0]}
    NUM_WARMFRAC = {"kind": "number", "min": 0, "max": 1, "step": 0.005, "presets": [0.01, 0.02, 0.03, 0.05, 0.1]}
    NUM_PROB     = {"kind": "number", "min": 0, "max": 1, "step": 0.05, "presets": [0, 0.25, 0.5, 0.75, 1.0]}
    NUM_DPI      = {"kind": "number", "int": True, "min": 72, "max": 600, "presets": [110, 150, 300, 600]}
    NUM_MAIL_MB  = {"kind": "number", "min": 1, "max": 100, "step": 0.5, "presets": [10, 15, 18, 24]}
    NUM_FILE_MB  = {"kind": "number", "min": 0, "max": 100, "step": 0.5, "presets": [0, 2, 5, 10, 15]}

    CH_STAGE    = {"kind": "choice", "options": ["full", "prepare", "compare"]}
    CH_FIGSTYLE = {"kind": "choice", "options": ["report", "paper"]}

    PICK_DATASET = {"kind": "dataset", "mode": "datasets", "baseFromParent": True, "validOnly": True}

    INFERENCE_TARGET_FIELDS = [
        "inference.target.enabled",
        "inference.target.dataset_path",
    ]

    MULTI_INT = {"kind": "multi", "numeric": True, "integer": True, "placeholder": "add value, Enter", "empty": "add at least one value"}

    MULTI_LOSS_KINDS = {"kind": "multi", "empty": "select at least one loss kind to sweep", "choices": [
        {"value": "mse",   "label": "Mean squared error"},
        {"value": "l1",    "label": "Mean absolute error"},
        {"value": "huber", "label": "Huber"},
    ]}

    INFERENCE_REPORT_GROUPS = [
        {"title": "Outputs", "fields": ["save_maps", "save_plots", "stitch_window"]},
        {"title": "Figures", "fields": ["cmap", "cmap_error", {"path": "fig_dpi", "widget": NUM_DPI}, {"path": "save_dpi", "widget": NUM_DPI}, {"path": "figure_style", "widget": CH_FIGSTYLE}]},
    ]

    TEMPLATES = {
        "loss": [
            {"title": "Loss", "fields": ["kind", "huber_delta", {"path": "channel_weights", "widget": {"kind": "multi", "numeric": True, "placeholder": "add weight, Enter", "empty": "every target channel weighs 1"}}]},
        ],
        "training_queue": [
            {"title": "Schedule", "fields": [
                {"path": "epochs", "widget": NUM_EPOCHS},
                "scheduler_epochs",
                {"path": "validation_frequency", "widget": NUM_FREQ},
                "scheduler_type",
                "scheduler_step_size",
                "scheduler_gamma",
                "scheduler_power",
                {"path": "eta_min", "widget": NUM_ETA_MIN},
                {"path": "early_stop_patience", "widget": NUM_PATIENCE},
                "early_stop_min_delta",
                "log_debug",
                "resume",
                "snapshot_every_n_epochs",
            ]},
            {"title": "Weight averaging", "fields": [
                {"gate": "use_ema", "fields": ["ema_decay", "ema_warmup"]},
                {"gate": "use_swa", "fields": [{"path": "swa_start_fraction", "widget": NUM_FRACTION}]},
            ]},
            {"title": "Optimizer", "fields": [
                "optimizer_type",
                {"path": "momentum", "widget": {"kind": "number", "min": 0, "max": 1, "step": 0.05, "presets": [0.0, 0.8, 0.9, 0.99]}},
                "nesterov",
            ]},
            {"title": "LR warmup", "fields": [
                {"gate": "warmup_enabled", "fields": [{"path": "warmup_fraction", "widget": NUM_WARMFRAC}, "warmup_mode", "warmup_poly_power"]},
            ]},
            {"title": "Batch and loader", "fields": [
                {"path": "batch_size", "widget": NUM_BATCH},
                {"path": "num_workers", "widget": NUM_WORKERS},
                {"path": "prefetch_factor", "widget": NUM_PREFETCH},
                "use_amp",
                {"path": "gradient_accumulation_steps", "widget": NUM_ACCUM},
                "scale_lr_with_batch",
                {"path": "lr_reference_batch_size", "widget": NUM_BATCH},
                "lr_multiplier",
                "abort_on_nonfinite_loss",
            ]},
            {"title": "Throughput", "fields": [
                {"gate": "compile_model", "fields": [{"path": "compile_mode", "widget": {"kind": "choice", "options": ["default", "reduce-overhead", "max-autotune"]}}]},
                "channels_last",
                {"path": "activation_checkpointing", "widget": {"kind": "multi", "placeholder": "add module pattern, Enter"}},
            ]},
            {"title": "Gradient clipping", "fields": [
                "clip_mode",
                "max_grad_norm",
                "clip_adaptive_window",
                "clip_adaptive_percentile",
                "clip_adaptive_mean_std_k",
            ]},
            {"title": "VRAM reservation", "fields": [
                {"gate": "reserve_vram", "fields": ["vram_keep_free_gb"]},
            ]},
            {"title": "Patches", "fields": [
                "patch_size",
                "patch_stride",
                "use_symmetric_padding",
            ]},
        ],
        "paths_rest": [
            {"title": "Run paths", "fields": ["log_base_dir", "fingerprint_dataset"]},
        ],
        "paths_train": [
            {"title": "Run paths", "fields": ["fingerprint_dataset"]},
        ],
        "normalization": [
            {"title": "Strategy", "fields": ["inputs", "targets"]},
        ],
        "augmentation": [
            {"title": "Flips", "fields": [
                {"path": "p_flip_h", "widget": NUM_PROB},
                {"path": "p_flip_v", "widget": NUM_PROB},
                {"path": "p_rot90", "widget": NUM_PROB},
            ]},
            {"title": "Noise", "fields": [{"path": "p_noise", "widget": NUM_PROB}, "noise_std"]},
        ],
        "augmentation_flips": [
            {"title": "Flips", "fields": [
                {"path": "p_flip_h", "widget": NUM_PROB},
                {"path": "p_flip_v", "widget": NUM_PROB},
                {"path": "p_rot90", "widget": NUM_PROB},
            ]},
        ],
        "inference_full": [
            {"title": "Run", "fields": ["run_directory", "output_subdir", "device", "log_level", {"path": "seed", "widget": NUM_SEED}]},
            {"title": "Execution", "fields": [
                "split",
                "checkpoint_name",
                {"path": "batch_size", "widget": NUM_BATCH},
                {"path": "num_workers", "widget": NUM_WORKERS},
            ]},
            *INFERENCE_REPORT_GROUPS,
            {"title": "Output layout", "fields": [
                "paths.figures_subdir",
                "paths.logs_subdir",
                "paths.maps_subdir",
                "paths.metrics_filename",
                "paths.report_filename",
            ]},
        ],
        "inference_queue": [
            {"title": "Execution", "fields": [
                "split",
                "checkpoint_name",
                {"path": "seed", "widget": NUM_SEED},
                {"path": "batch_size", "widget": NUM_BATCH},
                {"path": "num_workers", "widget": NUM_WORKERS},
            ]},
            *INFERENCE_REPORT_GROUPS,
        ],
        "inference_target": [
            {"title": "Dataset", "fields": [
                {"gate": "enabled", "fields": [{"path": "dataset_path", "widget": PICK_DATASET}]},
            ]},
        ],
    }

    TRAIN_ESSENTIALS = [
        "run_name",
        "resume",
        {"path": "resume_from", "widget": {"kind": "dataset", "mode": "resume_runs", "baseFrom": "logdir"}},
        {"path": "gpu", "widget": GPU_ONE},
        {"path": "gpus", "widget": GPU_MANY},
        "logdir",
        {"path": "paths.dataset_path", "widget": PICK_DATASET},
        {"path": "seed", "widget": NUM_SEED},
        "seeds",
    ]

    EXPERIMENT_TRIALS = [
        "trials_enabled", "trials_mode", "context_trials",
        "augmentation_trials.flip_probability", "augmentation_trials.noise_probability", "augmentation_trials.noise_stds",
        "stability_trials.warmup_fractions", "stability_trials.warmup_mode", "stability_trials.clip_mode", "stability_trials.max_grad_norms",
        "ema_trials.decays",
        "patch_trials.minimum", "patch_trials.maximum", "patch_trials.step", "patch_trials.stride_ratio", "patch_trials.constant_pixel_budget",
        "reach_trials.rungs", "reach_trials.patch_size", "reach_trials.patch_stride", "reach_trials.in_channels", "reach_trials.match_tolerance",
    ]

    INFERENCE_TARGET_NOTE = "Scores every selected run on the test split of a dataset it was never trained on. The run keeps its checkpoint and training normalization, so the target must carry the same input and target channels. Stamps land under <run>/cross_dataset/<dataset>/, apart from the run's own inference, and split must stay test."

    INFER_BACKBONE_LAYOUT = {
        "essentials": InferenceEssentials.for_block("inference.split"),
        "sections": [
            {"key": "backbone", "title": "Backbone", "panels": [
                {"kind": "fields", "title": "Backbone inference", "template": "inference_full", "at": "inference"},
                {"kind": "fields", "title": "Target dataset", "note": INFERENCE_TARGET_NOTE, "template": "inference_target", "at": "inference.target"},

                {"kind": "hidden", "fields": ["inference.region_override", "gpus_file"]},
            ]},
        ],
    }


    LAYOUTS = {
        "generate_dataset": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Output", "fields": ["generation.output_directory", "log_dir"]},
                        {"title": "Grid", "fields": [
                            "generation.rows",
                            "generation.cols",
                            "generation.input_channels",
                            {"path": "generation.seed", "widget": NUM_SEED},
                        ]},
                        {"title": "Fields", "fields": [
                            "generation.smoothing_pixels",
                            "generation.neighbourhood",
                            "generation.input_noise",
                            "generation.target_noise",
                        ]},
                        {"title": "Splits", "fields": [
                            {"path": "generation.val_fraction", "widget": NUM_FRACTION},
                            {"path": "generation.test_fraction", "widget": NUM_FRACTION},
                        ]},
                    ]},
                ]},
            ],
        },
        "train_backbone": {
            "essentials": TRAIN_ESSENTIALS,
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["backbone_name"]},
                    {"kind": "special", "panel": "arch_overrides", "modelFrom": "backbone_name", "fields": ["model_overrides"]},
                ]},
                {"key": "data", "title": "Data", "panels": [
                    {"kind": "fields", "title": "Paths", "template": "paths_train", "at": "paths"},
                    {"kind": "fields", "title": "Normalization", "template": "normalization", "at": "normalization"},
                    {"kind": "fields", "title": "Augmentation", "template": "augmentation", "at": "augmentation"},
                ]},
                {"key": "training", "title": "Training", "panels": [
                    {"kind": "fields", "title": "Training", "template": "training_queue", "at": "training"},
                ]},
                {"key": "loss", "title": "Loss", "panels": [
                    {"kind": "fields", "title": "Loss", "template": "loss", "at": "loss"},
                ]},
                {"key": "experiments", "title": "Experiments", "panels": [
                    {"kind": "special", "panel": "experiment_builder", "fields": EXPERIMENT_TRIALS},
                    {"kind": "fields", "groups": [
                        {"title": "Fan-out execution", "fields": ["poll_interval_s"]},
                    ]},
                    {"kind": "hidden", "fields": ["gpus_file"]},
                ]},
                {"key": "inference", "title": "Inference", "panels": [
                    {"kind": "fields", "groups": [{"title": "After training", "fields": ["infer_after", "infer_at_end"]}]},
                    {"kind": "fields", "title": "Inference run", "template": "inference_full", "at": "inference"},

                    {"kind": "hidden", "fields": ["inference.region_override", *INFERENCE_TARGET_FIELDS]},
                ]},
            ],
        },
        "infer_backbone":            INFER_BACKBONE_LAYOUT,
        "benchmark": {
            "essentials": [
                "run_tag",
                {"path": "gpus", "widget": GPU_MANY},
                {"path": "paths.dataset_path", "widget": PICK_DATASET},
                {"path": "seed", "widget": NUM_SEED},
                "seeds",
                "resume",
                {"path": "resume_from", "widget": {"kind": "dataset", "mode": "resume_runs", "baseFrom": "paths.log_base_dir"}},
                {"path": "stage", "widget": CH_STAGE},
                "poll_interval_s",
            ],
            "sections": [
                {"key": "sweep", "title": "Sweep", "panels": [
                    {"kind": "hidden", "fields": ["gpus_file"]},
                    {"kind": "special", "panel": "model_toggle", "fields": ["skip_models"]},
                    {"kind": "fields", "groups": [
                        {"title": "Loss kinds", "fields": [{"path": "sweep_loss_kinds", "widget": MULTI_LOSS_KINDS}]},
                    ]},
                ]},
                {"key": "data", "title": "Data", "panels": [
                    {"kind": "fields", "title": "Paths", "template": "paths_rest", "at": "paths"},
                ]},
                {"key": "transforms", "title": "Normalization and augmentation", "panels": [
                    {"kind": "fields", "title": "Normalization", "template": "normalization", "at": "normalization"},
                    {"kind": "fields", "title": "Augmentation", "template": "augmentation", "at": "augmentation"},
                ]},
                {"key": "training", "title": "Training", "panels": [
                    {"kind": "fields", "title": "Training", "template": "training_queue", "at": "training"},
                ]},
                {"key": "loss", "title": "Loss", "panels": [
                    {"kind": "fields", "title": "Base loss for the swept kinds", "template": "loss", "at": "loss"},
                ]},
                {"key": "inference", "title": "Inference", "panels": [
                    {"kind": "fields", "groups": [{"title": "After training", "fields": ["infer_after"]}]},
                    {"kind": "fields", "title": "Inference run", "template": "inference_queue", "at": "inference"},
                    {"kind": "fields", "groups": [{"title": "Comparison report", "fields": ["comparison.embed_images"]}]},
                ]},
            ],
        },
        "cross_validate": {
            "essentials": [
                "run_tag",
                {"path": "gpus", "widget": GPU_MANY},
                {"path": "paths.dataset_path", "widget": PICK_DATASET},
                {"path": "seed", "widget": NUM_SEED},
                "seeds",
                "resume",
                {"path": "resume_from", "widget": {"kind": "dataset", "mode": "resume_runs", "baseFrom": "paths.log_base_dir"}},
                {"path": "stage", "widget": CH_STAGE},
                "poll_interval_s",
            ],
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["backbone_name"]},
                    {"kind": "special", "panel": "arch_overrides", "modelFrom": "backbone_name", "fields": ["model_overrides"]},
                ]},
                {"key": "folds", "title": "Folds", "panels": [
                    {"kind": "hidden", "fields": ["gpus_file"]},
                    {"kind": "fields", "groups": [
                        {"title": "Fold layout", "fields": ["folds.n_folds", "folds.row_start", "folds.row_end", "folds.guard"]},
                    ]},
                ]},
                {"key": "data", "title": "Data", "panels": [
                    {"kind": "fields", "title": "Paths", "template": "paths_rest", "at": "paths"},
                ]},
                {"key": "transforms", "title": "Normalization and augmentation", "panels": [
                    {"kind": "fields", "title": "Normalization", "template": "normalization", "at": "normalization"},
                    {"kind": "fields", "title": "Augmentation", "template": "augmentation", "at": "augmentation"},
                ]},
                {"key": "training", "title": "Training", "panels": [
                    {"kind": "fields", "title": "Training", "template": "training_queue", "at": "training"},
                ]},
                {"key": "loss", "title": "Loss", "panels": [
                    {"kind": "fields", "title": "Loss", "template": "loss", "at": "loss"},
                ]},
                {"key": "inference", "title": "Inference", "panels": [
                    {"kind": "fields", "groups": [{"title": "Splits", "fields": ["inference_splits"]}]},
                    {"kind": "fields", "title": "Inference run", "template": "inference_queue", "at": "inference"},
                    {"kind": "fields", "groups": [{"title": "Comparison report", "fields": ["comparison.embed_images"]}]},
                ]},
            ],
        },
        "tune": {
            "essentials": [
                "run_tag",
                {"path": "resume_from", "widget": {"kind": "dataset", "mode": "resume_runs", "baseFrom": "paths.log_base_dir"}},
                {"path": "stage", "widget": CH_STAGE},
                {"path": "gpus", "widget": GPU_MANY},
                {"path": "n_workers", "widget": TUNE_WORKERS},
                {"path": "paths.dataset_path", "widget": PICK_DATASET},
            ],
            "sections": [
                {"key": "search", "title": "Search", "panels": [
                    {"kind": "hidden", "fields": ["gpus_file"]},
                    {"kind": "special", "panel": "model_toggle", "fields": ["skip_models"]},
                    {"kind": "fields", "title": "Optuna search", "groups": [
                        {"title": "Trials", "fields": ["tuning.n_trials", {"path": "tuning.n_epochs", "widget": NUM_EPOCHS}, {"path": "tuning.base_seed", "widget": NUM_SEED}, {"path": "tuning.early_stop_patience", "widget": NUM_PATIENCE}]},
                        {"title": "Search groups", "fields": ["tuning.search_lr", "tuning.search_arch", "tuning.search_batch", {"path": "tuning.batch_sizes", "widget": MULTI_INT}]},
                        {"title": "Stop rules", "fields": [{"path": "tuning.plateau_patience", "widget": NUM_PLATEAU}, {"path": "tuning.plateau_min_delta", "widget": NUM_DELTA}, {"path": "tuning.time_budget_h", "widget": NUM_HOURS}]},
                        {"title": "Pruner", "fields": ["tuning.pruner_n_startup_trials", "tuning.pruner_n_warmup_steps"]},
                        {"title": "Outputs", "fields": ["tuning.emit_trial_docs", "tuning.emit_study_plots", "tuning.keep_trial_checkpoints"]},
                    ]},
                ]},
                {"key": "data", "title": "Data", "panels": [
                    {"kind": "fields", "title": "Paths", "template": "paths_rest", "at": "paths"},
                ]},
                {"key": "transforms", "title": "Normalization and augmentation", "panels": [
                    {"kind": "fields", "title": "Normalization", "template": "normalization", "at": "normalization"},
                    {"kind": "fields", "title": "Augmentation", "template": "augmentation", "at": "augmentation"},
                ]},
                {"key": "training", "title": "Training", "panels": [
                    {"kind": "fields", "title": "Training", "template": "training_queue", "at": "training"},
                ]},
                {"key": "loss", "title": "Loss", "panels": [
                    {"kind": "fields", "title": "Loss", "template": "loss", "at": "loss"},
                ]},
            ],
        },
        "compare_trials": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_tags", "widget": {"kind": "dataset", "mode": "runs_compare", "multi": True, "baseFrom": "runs_dir", "seedButton": True}},
                            "inference_split",
                        ]},
                        {"title": "Report", "fields": ["compare_images", "compare_gifs", "embed_images", "output_dir"]},
                    ]},
                ]},
            ],
        },
        "compare_runs": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Run", "fields": ["paths.log_base_dir", "run_tag", "inference_split"]},
                        {"title": "Report", "fields": ["embed_images"]},
                    ]},
                ]},
            ],
        },
        "compare_seeds": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Seed groups", "fields": [
                            "runs_dir",
                            {"path": "group_tags", "widget": {"kind": "dataset", "mode": "run_groups", "multi": True, "baseFrom": "runs_dir"}},
                        ]},
                        {"title": "Report", "fields": ["split", "inference_subdir", "output_subdir", "metrics_filename", "report_filename"]},
                        {"title": "Disagreement maps", "fields": ["compute_disagreement", "cubes_subdir"]},
                        {"title": "Seed ensemble", "fields": ["compute_ensemble"]},
                    ]},
                ]},
            ],
        },
        "xray_weights": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_filename",
                            "output_subdir",
                        ]},
                        {"title": "Report", "fields": ["make_plots", "embed_images"]},
                        {"title": "Dead weights", "fields": ["thresholds.dead_abs_threshold", "thresholds.dead_fraction_warn", "thresholds.dead_fraction_critical", "thresholds.dead_unit_norm_frac", "thresholds.dead_unit_fraction_warn"]},
                        {"title": "Degenerate weights", "fields": ["thresholds.uniform_cv_threshold", "thresholds.constant_std_threshold", "thresholds.rank_ratio_warn", "thresholds.rank_ratio_critical", "thresholds.duplicate_cosine", "thresholds.duplicate_fraction_warn"]},
                        {"title": "Exploding weights", "fields": ["thresholds.explode_abs_threshold", "thresholds.spectral_norm_warn", "thresholds.bias_abs_threshold"]},
                        {"title": "Normalization layers", "fields": ["thresholds.norm_scale_dead_value", "thresholds.norm_scale_dead_frac", "thresholds.running_var_dead_value", "thresholds.running_var_dead_frac"]},
                        {"title": "Initialization drift", "fields": ["thresholds.init_ratio_low", "thresholds.init_ratio_high"]},
                        {"title": "Limits", "fields": ["svd_max_dim", "duplicate_max_units", "max_layer_histograms"]},
                    ]},
                ]},
            ],
        },
        "xray_activations": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Capture", "fields": [
                            "split",
                            "device",
                            {"path": "batch_size",  "widget": {"kind": "number", "min": 1, "max": 128, "step": 1}},
                            {"path": "max_batches", "widget": {"kind": "number", "min": 1, "max": 128, "step": 1}},
                        ]},
                        {"title": "Detection thresholds", "fields": ["dead_zero_frac_warn", "dead_zero_frac_critical", "dead_channel_frac_warn", "explode_abs_threshold", "constant_std_threshold", "channel_collapse_frac_warn"]},
                        {"title": "Report", "fields": ["make_plots", "max_layer_histograms", {"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "measure_receptive_field": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Probing", "fields": [
                            "split",
                            "device",
                            {"path": "window",           "widget": {"kind": "number", "min": 32, "max": 512, "step": 16}},
                            {"path": "n_row_probes", "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            {"path": "n_col_probes",   "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            {"path": "mass_windows",     "widget": {"kind": "multi", "numeric": True}},
                        ]},
                        {"title": "Figures", "fields": [{"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "animate_training": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Animation", "fields": [
                            "split",
                            "device",
                            {"path": "window",           "widget": {"kind": "number", "min": 16, "max": 256, "step": 16}},
                            {"path": "n_row_probes", "widget": {"kind": "number", "min": 1, "max": 8, "step": 1}},
                            {"path": "n_col_probes",   "widget": {"kind": "number", "min": 1, "max": 8, "step": 1}},
                            {"path": "fps",              "widget": {"kind": "number", "min": 1, "max": 30, "step": 1}},
                        ]},
                        {"title": "Figures", "fields": [{"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "stress_inputs": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Stress", "fields": [
                            "split",
                            "device",
                            {"path": "batch_size",      "widget": {"kind": "number", "min": 1, "max": 64, "step": 1}},
                            {"path": "max_batches",     "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            {"path": "noise_sigmas",    "widget": {"kind": "multi", "numeric": True}},
                            {"path": "noise_draws",     "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            {"path": "gain_factors",    "widget": {"kind": "multi", "numeric": True}},
                            {"path": "shift_pixels",    "widget": {"kind": "multi", "numeric": True}},
                            {"path": "draws_per_count", "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            {"path": "seed", "widget": NUM_SEED},
                        ]},
                        {"title": "Figures", "fields": [{"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "map_loss_landscape": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Landscape", "fields": [
                            "split",
                            "device",
                            {"path": "batch_size",  "widget": {"kind": "number", "min": 1, "max": 64, "step": 1}},
                            {"path": "max_batches", "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            "span",
                            {"path": "n_directions", "widget": {"kind": "number", "min": 2, "max": 16, "step": 1}},
                            {"path": "n_points_1d", "widget": {"kind": "number", "min": 5, "max": 101, "step": 2}},
                            {"path": "n_points_2d", "widget": {"kind": "number", "min": 5, "max": 41, "step": 2}},
                            {"path": "direction_seed", "widget": NUM_SEED},
                        ]},
                        {"title": "Figures", "fields": [{"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "probe_layers": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Probing", "fields": [
                            "split",
                            "device",
                            {"path": "batch_size",        "widget": {"kind": "number", "min": 1, "max": 64, "step": 1}},
                            {"path": "max_batches",       "widget": {"kind": "number", "min": 1, "max": 64, "step": 1}},
                            {"path": "max_layers",        "widget": {"kind": "number", "min": 2, "max": 128, "step": 1}},
                            {"path": "samples_per_batch", "widget": {"kind": "number", "min": 64, "max": 8192, "step": 64}},
                            "ridge_lambda",
                            {"path": "n_folds", "widget": {"kind": "number", "min": 2, "max": 10, "step": 1}},
                            "seed",
                        ]},
                        {"title": "Figures", "fields": [{"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "attribute_inputs": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "checkpointOnly": True}},
                            "checkpoint_name",
                            "output_subdir",
                        ]},
                        {"title": "Gradient probes", "fields": [
                            "split",
                            "device",
                            {"path": "window",           "widget": {"kind": "number", "min": 16, "max": 256, "step": 16}},
                            {"path": "n_row_probes", "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                            {"path": "n_col_probes",   "widget": {"kind": "number", "min": 1, "max": 16, "step": 1}},
                        ]},
                        {"title": "Occlusion", "fields": [
                            {"gate": "occlude", "fields": [
                                {"path": "batch_size",  "widget": {"kind": "number", "min": 1, "max": 128, "step": 1}},
                                {"path": "max_batches", "widget": {"kind": "number", "min": 1, "max": 64, "step": 1}},
                            ]},
                        ]},
                        {"title": "Figures", "fields": [{"path": "figure_style", "widget": CH_FIGSTYLE}]},
                    ]},
                ]},
            ],
        },
        "export_tensorboard_plots": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir"}},
                        ]},
                        {"title": "Output", "fields": ["tensorboard_dirname", "output_subdir"]},
                    ]},
                ]},
            ],
        },
        "export_paper_figures": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir"}},
                            "split",
                            "inference_subdir",
                        ]},
                        {"title": "Pack", "fields": [
                            {"path": "patterns", "widget": {"kind": "multi"}},
                            "figures_subdir",
                            "metrics_filename",
                            "report_filename",
                            "output_dir",
                        ]},
                    ]},
                ]},
            ],
        },
        "export_onnx": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir"}},
                        ]},
                        {"title": "Graph", "fields": [
                            {"path": "opset", "widget": {"kind": "number", "int": True, "min": 11, "max": 22, "presets": [17, 18, 20]}},
                            "dynamic_batch",
                        ]},
                    ]},
                ]},
            ],
        },
        "profile_training": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Workload", "fields": ["model_name", {"path": "gpu", "widget": GPU_ONE}, {"path": "seed", "widget": NUM_SEED}, "use_amp"]},
                        {"title": "Steps", "fields": [
                            {"path": "batch_size", "widget": NUM_BATCH},
                            {"path": "num_workers", "widget": NUM_WORKERS},
                            {"path": "prefetch_factor", "widget": NUM_PREFETCH},
                            {"path": "warmup_steps", "widget": {"kind": "number", "int": True, "min": 0, "max": 200, "presets": [4, 8, 16]}},
                            {"path": "active_steps", "widget": {"kind": "number", "int": True, "min": 1, "max": 500, "presets": [12, 24, 48]}},
                        ]},
                        {"title": "Trace", "fields": ["record_shapes", "profile_memory", "with_stack"]},
                        {"title": "Paths", "fields": [
                            {"path": "paths.dataset_path", "widget": PICK_DATASET},
                            "paths.log_base_dir",
                            "paths.fingerprint_dataset",
                            "output_dir",
                        ]},
                    ]},
                ]},
            ],
        },
        "export_result_tables": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir"}},
                            "split",
                            "inference_subdir",
                            "metrics_filename",
                        ]},
                        {"title": "Table", "fields": [
                            {"path": "metrics", "widget": {"kind": "multi", "placeholder": "add metric key, Enter"}},
                            "sort_by",
                            "ascending",
                            {"path": "precision", "widget": {"kind": "number", "int": True, "min": 1, "max": 12, "presets": [3, 4, 6]}},
                            "output_dir",
                        ]},
                    ]},
                ]},
            ],
        },
        "collect_reports": {
            "sections": [
                {"key": "config", "title": "Configuration", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "units": True, "seedButton": True}},
                        ]},
                        {"title": "Collection", "fields": ["collector_dir", "inference_dirname", "report_filename", "latest_only", "embed_images"]},
                    ]},
                ]},
            ],
        },
        "mail_experiment_reports": {
            "type_tab": {"field": "target", "options": [["experiments", "Experiments"], ["runs", "Runs"], ["datasets", "Datasets"]]},
            "sections": [
                {"key": "experiments", "title": "Experiments", "when": {"field": "target", "in": ["experiments"]}, "panels": [
                    {"kind": "fields", "note": "Mails one message per experiment found at or below the path: a sweep of trial runs, a cross-validation run, a benchmark run, a sequence workspace holding several of them, or a whole runs tree. Each message carries the seed-aggregate report of every trial, the trial-comparison reports, a cross-validation run's cross-fold report and the configuration the experiment ran with; a missing seed aggregate, comparison or cross-fold report is written first, so nothing has to be run by hand before mailing.", "groups": [
                        {"title": "Experiments", "fields": [
                            "experiments_dir",
                            {"path": "experiment_filter", "widget": {"kind": "dataset", "mode": "experiments", "multi": True, "baseFrom": "experiments_dir", "markerFrom": "comparison_dirname"}},
                        ]},
                        {"title": "Bundle", "fields": [
                            "comparison_dirname",
                            {"path": "comparison_reports", "widget": {"kind": "multi"}},
                            "inference_dirname",
                            "report_filename",
                            "regenerate_reports",
                            "embed_images",
                        ]},
                    ]},
                ]},
                {"key": "runs", "title": "Runs", "when": {"field": "target", "in": ["runs"]}, "panels": [
                    {"kind": "fields", "note": "Mails one message per selected run carrying everything that run holds: the configuration and metadata it recorded, its logs and docs, the TensorBoard events, and every report and figure its inference wrote, packed into zip volumes. What travels is decided by what is left out rather than by a list of gathered paths: the three rules below drop the checkpoints, the reconstructed cubes and any file above the per-file cap, which is what keeps the walk animations behind.", "groups": [
                        {"title": "Runs", "fields": [
                            "runs_dir",
                            {"path": "run_filter", "widget": {"kind": "dataset", "mode": "runs", "multi": True, "baseFrom": "runs_dir", "hint": "nothing selected = every run at or below the path is mailed"}},
                        ]},
                        {"title": "Left out", "fields": [
                            {"path": "run_excluded_paths", "widget": {"kind": "multi"}},
                            {"path": "run_excluded_suffixes", "widget": {"kind": "multi"}},
                            {"path": "run_max_file_mb", "widget": NUM_FILE_MB},
                        ]},
                    ]},
                ]},
                {"key": "datasets", "title": "Datasets", "when": {"field": "target", "in": ["datasets"]}, "panels": [
                    {"kind": "fields", "note": "Mails everything generate_dataset wrote into a dataset, the layout and the metadata, packed into zip volumes. The arrays under data/ are left out unless you name them; the gathered tree is rebuilt by unpacking every volume into one directory.", "groups": [
                        {"title": "Datasets", "fields": [
                            "datasets_dir",
                            {"path": "dataset_filter", "widget": {"kind": "dataset", "mode": "datasets", "multi": True, "baseFrom": "datasets_dir", "validOnly": True}},
                        ]},
                        {"title": "Gathered paths", "fields": [
                            {"path": "dataset_required_paths", "widget": {"kind": "multi"}},
                            {"path": "dataset_optional_paths", "widget": {"kind": "multi"}},
                        ]},
                    ]},
                ]},
                {"key": "mail", "title": "Mail", "panels": [
                    {"kind": "fields", "groups": [
                        {"title": "Staging", "fields": ["staging_dir", {"path": "max_attachment_mb", "widget": NUM_MAIL_MB}]},
                        {"title": "Mail", "fields": ["recipient", "sender", "smtp_host", "smtp_port", "smtp_timeout_s", "password_env", "subject_prefix", "dry_run"]},
                    ]},
                ]},
            ],
        },
    }

    def _field_entry(self, item, prefix, widgets):
        if isinstance(item, str):
            return {"path": self._join(prefix, item)}

        if "gate" in item:
            entry = {"gate": self._join(prefix, item["gate"]), "fields": [self._field_entry(sub, prefix, widgets) for sub in item["fields"]]}
            return entry

        path = self._join(prefix, item["path"])
        if "widget" in item:
            widgets[path] = item["widget"]
        return {"path": path}

    def _join(self, prefix, name):
        return f"{prefix}.{name}" if prefix else name

    def _expand_groups(self, groups, prefix, widgets):
        expanded = []
        for group in groups:
            fields = [self._field_entry(item, prefix, widgets) for item in group["fields"]]
            expanded.append({"title": group.get("title"), "fields": fields})
        return expanded

    def _panel_groups(self, panel, prefix, widgets):
        if "template" in panel:
            return self._expand_groups(self.TEMPLATES[panel["template"]], prefix, widgets)
        return self._expand_groups(panel["groups"], prefix, widgets)

    def _expand_panel(self, panel, widgets):
        if panel["kind"] == "special":
            expanded = {"kind": "special", "panel": panel["panel"], "fields": list(panel["fields"])}
            if "modelFrom" in panel:
                expanded["modelFrom"] = panel["modelFrom"]
            if "title" in panel:
                expanded["title"] = panel["title"]
            return expanded

        if panel["kind"] == "hidden":
            return {"kind": "hidden", "fields": list(panel["fields"])}

        groups = self._panel_groups(panel, panel.get("at", ""), widgets)
        return {"kind": "fields", "title": panel.get("title"), "note": panel.get("note"), "groups": groups}

    def _expand(self, key):
        spec    = self.LAYOUTS[key]
        widgets = {}

        essentials = [self._field_entry(item, "", widgets) for item in spec.get("essentials", [])]

        sections = []
        for section in spec["sections"]:
            panels   = [self._expand_panel(panel, widgets) for panel in section["panels"]]
            expanded = {"key": section["key"], "title": section["title"], "panels": panels}
            if "when" in section:
                expanded["when"] = copy.deepcopy(section["when"])
            sections.append(expanded)

        layout = {"essentials": essentials, "sections": sections, "widgets": widgets}
        if "type_tab" in spec:
            layout["type_tab"] = copy.deepcopy(spec["type_tab"])
        return layout

    def _entry_claims(self, entry, out):
        if "gate" in entry:
            out.append(entry["gate"])
            for sub in entry["fields"]:
                self._entry_claims(sub, out)
            return
        out.append(entry["path"])

    def _when_conditions(self, when):
        if when is None:
            return []
        return list(when) if isinstance(when, list) else [when]

    def _claims(self, layout):
        claimed = []

        for entry in layout["essentials"]:
            self._entry_claims(entry, claimed)

        if "type_tab" in layout:
            claimed.append(layout["type_tab"]["field"])

        for section in layout["sections"]:
            for panel in section["panels"]:
                if panel["kind"] in ("special", "hidden"):
                    claimed.extend(panel["fields"])
                    continue

                rows = []
                for group in panel["groups"]:
                    for entry in group["fields"]:
                        self._entry_claims(entry, rows)

                claimed.extend(rows)

        return claimed

    def _validate(self, key, layout, leaves):
        paths   = [leaf["path"] for leaf in leaves]
        known   = set(paths)
        claimed = self._claims(layout)

        counts    = Counter(claimed)
        claim_set = set(counts)

        duplicates = sorted(path for path, count in counts.items() if count > 1)
        unknown    = sorted(claim_set - known)
        unclaimed  = [path for path in paths if path not in claim_set]

        problems = []
        if unknown:
            problems.append(f"layout for {key} names unknown fields: {', '.join(unknown)}")
        if duplicates:
            problems.append(f"layout for {key} claims fields twice: {', '.join(duplicates)}")
        if unclaimed:
            problems.append(f"layout for {key} leaves fields unclaimed: {', '.join(unclaimed)}")

        for section in layout["sections"]:
            for condition in self._when_conditions(section.get("when")):
                if condition["field"] not in known:
                    problems.append(f"section {section['key']} gates on unknown field {condition['field']}")
                if ("in" in condition) == ("set" in condition):
                    problems.append(f"section {section['key']} has a when condition that needs exactly one of 'in' or 'set'")

        for section in layout["sections"]:
            for panel in section["panels"]:
                if panel["kind"] != "special":
                    continue
                if panel.get("modelFrom") and panel["modelFrom"] not in known:
                    problems.append(f"special panel {panel['panel']} reads unknown model field {panel['modelFrom']}")

        for path, widget in layout["widgets"].items():
            if widget.get("kind") == "number" and not ("min" in widget and "max" in widget):
                problems.append(f"number widget for {path} lacks min/max bounds")

        if problems:
            raise LayoutError("\n".join(problems))

    def build(self, key, leaves):
        if key not in self.LAYOUTS:
            raise LayoutError(f"no launch layout declared for {key}")

        layout = self._expand(key)
        self._validate(key, layout, leaves)

        layout["mode"] = "single" if len(layout["sections"]) == 1 and not layout["essentials"] else "sections"
        return layout
