from __future__ import annotations

from pipelines.shared.inference.report import InferenceReportBase


class ImageAeReport(InferenceReportBase):
    DOC_TITLE = "Image Autoencoder Inference Report"

    METRIC_GROUPS = [
        ("Dataset",                     ["n_patches", "n_channels", "patch_height", "patch_width", "embedding_dim"]),
        ("Reconstruction (physical)",   ["mse_mean", "mse_median", "mae_mean", "rmse", "max_abs_error_mean", "r2", "psnr"]),
        ("Reconstruction (normalized)", ["mse_mean_normalized", "mae_mean_normalized"]),
        ("Latent embedding",            ["embedding_norm_mean", "embedding_dim_std_mean", "embedding_active_dim_fraction"]),
    ]

    FIGURE_SECTIONS = [
        ("Error distribution",     ["error_histogram"]),
        ("Per-channel error",      ["channel_mse"]),
        ("Mean patch intensity",   ["intensity_scatter"]),
        ("Embedding norm",         ["embedding_norm"]),
        ("Best reconstructions",   ["best"]),
        ("Worst reconstructions",  ["worst"]),
        ("Random reconstructions", ["random"]),
    ]

    def _summary(self) -> dict:
        return {
            "model_name"        : self.run.ae_name,
            "embedding_dim"     : self.run.embedding_dim,
            "in_channels"       : self.run.in_channels,
            "patch_size"        : self.run.patch_size,
            "split"             : self.run.split_name,
            "split_region"      : str(self.run.split_region.as_tuple()),
            "best_epoch"        : self.run.checkpoint_meta["best_epoch"],
            "best_val_loss"     : self.run.checkpoint_meta["best_val_loss"],
            "preprocessing_dir" : str(self.run.preprocessing_run_directory),
        }
