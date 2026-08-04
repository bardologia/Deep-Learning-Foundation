from __future__ import annotations

from pipelines.shared.inference.report import InferenceReportBase


class UnrolledReport(InferenceReportBase):
    DOC_TITLE = "Unrolled Physics Network Inference Report"

    METRIC_GROUPS = [
        ("Curve reconstruction", ["loss", "curve_l1", "curve_l1_p50", "curve_l1_p90", "curve_mse", "curve_rmse"]),
        ("Peak localisation",    ["peak_mae_m"]),
        ("Coverage",             ["n_pixels", "n_valid_pixels", "valid_fraction"]),
        ("Training reference",   ["best_val_loss", "train_test_loss"]),
    ]

    FIGURE_SECTIONS = [
        ("Error maps",         ["curve_l1_map", "peak_error_map"]),
        ("Error distribution", ["error_histogram"]),
        ("Peak height maps",   ["peak_heights"]),
        ("Best profiles",      ["best"]),
        ("Median profiles",    ["median"]),
        ("Worst profiles",     ["worst"]),
    ]

    def _summary(self) -> dict:
        return {
            "model_name"            : self.run.model_name,
            "n_iterations"          : self.run.model_config.n_iterations,
            "parameters"            : sum(parameter.numel() for parameter in self.run.model.parameters()),
            "split"                 : self.run.split_name,
            "split_region"          : str(self.run.split_region.as_tuple()),
            "tracks"                : self.run.kz_field.shape[0],
            "x_axis_bins"           : self.run.x_axis.size,
            "checkpoint"            : str(self.run.checkpoint_path),
            "curve_loss"            : self.run.curve_loss,
            "measurement_noise_std" : self.run.noise_std,
            "power_floor"           : self.run.power_floor,
        }
