"""Markdown report layout for profile-autoencoder inference runs."""

from __future__ import annotations

from pipelines.shared.inference.report import InferenceReportBase


class ProfileAeReport(InferenceReportBase):
    """Groups the profile-autoencoder metrics and figures into a report."""

    DOC_TITLE = "Profile Autoencoder Inference Report"

    METRIC_GROUPS = [
        ("Dataset",                     ["n_curves", "n_active_curves", "profile_length", "embedding_dim"]),
        ("Reconstruction (physical)",   ["mse_mean", "mse_median", "mae_mean", "rmse", "max_abs_error_mean", "r2"]),
        ("Reconstruction (normalized)", ["mse_mean_normalized", "mae_mean_normalized"]),
        ("Profile shape",               ["pearson_mean", "pearson_median", "relative_l2_mean", "relative_l2_median"]),
        ("Power and peak",              ["power_rel_error_mean", "power_rel_error_median", "peak_location_mae", "peak_amplitude_rel_err_mean"]),
        ("Latent embedding",            ["embedding_norm_mean", "embedding_dim_std_mean", "embedding_active_dim_fraction"]),
    ]

    FIGURE_SECTIONS = [
        ("Mean profile",           ["mean_profile"]),
        ("Error distribution",     ["error_histogram"]),
        ("Integrated power",       ["power_scatter"]),
        ("Embedding norm",         ["embedding_norm"]),
        ("Latent RGB scatter",     ["latent_rgb_scatter"]),
        ("Latent RGB scene map",   ["latent_rgb_map"]),
        ("Best reconstructions",   ["best"]),
        ("Worst reconstructions",  ["worst"]),
        ("Random reconstructions", ["random"]),
    ]

    def _summary(self) -> dict:
        """Returns the run header fields: model, geometry, split, scored checkpoint, curve sampling and data source."""
        return {
            "model_name"         : self.run.ae_name,
            "embedding_dim"      : self.run.embedding_dim,
            "profile_length"     : int(self.run.z_axis.shape[0]),
            "x_axis_min"         : float(self.run.z_axis.min()),
            "x_axis_max"         : float(self.run.z_axis.max()),
            "split"              : self.run.split_name,
            "split_regions"      : ", ".join(str(region.as_tuple()) for region in self.run.split_regions),
            "checkpoint"         : self.config.checkpoint_name,
            "checkpoint_epoch"   : int(self.run.checkpoint_meta["epoch"]) + 1,
            "best_epoch"         : int(self.run.checkpoint_meta["best_epoch"]) + 1,
            "best_val_loss"      : self.run.checkpoint_meta["best_val_loss"],
            "activity_threshold" : self.run.activity_threshold,
            "pixel_subsample"    : self.config.pixel_subsample,
            "keep_empty_frac"    : self.config.keep_empty_frac,
            "selection_seed"     : self.config.seed,
            "preprocessing_dir"  : str(self.run.preprocessing_run_directory),
        }
