from __future__ import annotations

from pathlib import Path
from typing  import Any, Dict, List, Tuple

import numpy as np

from tools.reporting.reporting import ReportAssets
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter
from tools.runtime.run_tag     import RunTag


class ReportPayloadBuilder:
    @staticmethod
    def run_summary(run, x_axis_np: np.ndarray) -> Dict[str, Any]:
        return {
            "model_name"        : run.backbone_name,
            "in_channels"       : run.in_channels,
            "out_channels"      : run.out_channels,
            "n_gaussians"       : run.n_gaussians,
            "x_axis_length"     : run.x_axis_length,
            "x_axis_min"        : float(x_axis_np.min()),
            "x_axis_max"        : float(x_axis_np.max()),
            "split"             : run.split_name,
            "split_region"      : str(run.split_region.as_tuple()),
            "global_crop"       : str(run.global_crop.as_tuple()),
            "patches"           : run.grid.number_of_patches,
            "patch_size"        : str(run.grid.patch_size),
            "patch_stride"      : run.grid.stride,
            "preprocessing_dir" : str(run.dataset_config.preprocessing_run_directory),
            "input_config"      : run.dataset_config.input_config.as_dict(),
            "secondary_labels"  : ", ".join(run.secondary_labels) if run.secondary_labels else "all passes",
        }

    @staticmethod
    def inference_config(cfg, run) -> Dict[str, Any]:
        return {
            "stitch_window"      : cfg.stitch_window,
            "cube_dtype"         : cfg.cube_dtype,
            "save_cubes"         : cfg.save_cubes,
            "save_plots"         : cfg.save_plots,
            "save_animations"    : cfg.save_animations,
            "n_best_profiles"    : cfg.n_best_profiles,
            "n_worst_profiles"   : cfg.n_worst_profiles,
            "n_random_profiles"  : cfg.n_random_profiles,
            "n_range_slices"     : cfg.n_range_slices,
            "n_azimuth_slices"   : cfg.n_azimuth_slices,
            "n_elevation_slices" : cfg.n_elevation_slices,
            "gif_axes"           : cfg.gif_axes,
            "gif_fps"            : cfg.gif_fps,
            "gif_max_frames"     : cfg.gif_max_frames,
            "device"             : cfg.device,
            "batch_size"         : run.dataset_config.batch_size,
            "num_workers"        : cfg.num_workers,
        }


class Report:

    _DATASET_KEYS = frozenset({
        "gt_mean", "gt_std", "gt_max",
        "pred_mean", "pred_std", "pred_max",
        "n_pixels", "n_elevation",
        "x_axis_min", "x_axis_max", "x_axis_step",
    })

    def __init__(
        self,
        output_dir       : Path,
        run_summary      : Dict,
        inference_config : Dict,
        checkpoint_meta  : Dict,
        global_metrics   : Dict,
        figure_paths     : Dict[str, Path],
        gif_paths        : Dict[str, Path],
        report_path      : Path,
    ) -> None:

        self.report_path      = Path(report_path)
        self.output_dir       = Path(output_dir)
        self.run_summary      = run_summary
        self.inference_config = inference_config
        self.checkpoint_meta  = checkpoint_meta
        self.global_metrics   = global_metrics
        self.figure_paths     = figure_paths
        self.gif_paths        = gif_paths
        self.assets           = ReportAssets(self.output_dir)

    @staticmethod
    def _fmt(v: Any) -> str:
        return ScalarFormatter.format_scalar(v, precision=6, adaptive=True)

    @staticmethod
    def _kv_table(rows: List[Tuple[str, Any]], header: Tuple[str, str] = ("Key", "Value")) -> str:
        table = MarkdownTable(header)
        for k, v in rows:
            table.add_row(f"`{k}`", Report._fmt(v))
        return "\n".join(table.render())

    @staticmethod
    def _dict_table(d: Dict[str, Any]) -> str:
        return Report._kv_table(sorted(d.items()))

    @staticmethod
    def _three_col_table(
        rows   : List[Tuple[str, Any, str]],
        header : Tuple[str, str, str] = ("Metric", "Pred vs GT", "Description"),
    ) -> str:
        table = MarkdownTable(header)
        for label, gt_val, desc in rows:
            table.add_row(label, Report._fmt(gt_val), desc)
        return "\n".join(table.render())

    @staticmethod
    def _gt_pred_table(
        rows   : List[Tuple[str, Any, Any, str]],
        header : Tuple[str, str, str, str] = ("Metric", "GT", "Pred", "Description"),
    ) -> str:
        table = MarkdownTable(header)
        for label, gt_val, pred_val, desc in rows:
            table.add_row(label, Report._fmt(gt_val), Report._fmt(pred_val), desc)
        return "\n".join(table.render())

    @staticmethod
    def _padded_column(payload: Dict[str, Any], key: str, n_tracks: int) -> list:
        values = payload[key]
        if len(values) != n_tracks:
            raise ValueError(f"Track payload column '{key}' has {len(values)} entries but the run lists {n_tracks} tracks; the persisted track metadata is inconsistent.")
        return values

    _PER_SLICE_PREFIXES = (
        "ssim_gt_elev_",   "ssim_gt_range_",   "ssim_gt_azimuth_",
        "ssim_norm_elev_", "ssim_norm_range_", "ssim_norm_azimuth_",
        "ssim_red_elev_",  "ssim_red_range_",  "ssim_red_azimuth_",
        "elev_mae_gt_",  "elev_rmse_gt_",  "elev_r2_gt_",  "elev_ce_gt_",
        "elev_mae_red_", "elev_rmse_red_", "elev_r2_red_", "elev_ce_red_",
    )

    @classmethod
    def _is_per_slice_ssim(cls, k: str) -> bool:
        return any(k.startswith(prefix) and k[len(prefix):].lstrip("-").isdigit() for prefix in cls._PER_SLICE_PREFIXES)

    def _build_run_summary(self) -> List[str]:
        rs  = self.run_summary
        cfg = self.inference_config
        ck  = self.checkpoint_meta
        out = ["\n## 1. Run summary\n"]

        out.append("\n### 1.1 Model\n")
        out.append(self._kv_table([
            ("model_name",     rs["model_name"]),
            ("in_channels",    rs["in_channels"]),
            ("out_channels",   rs["out_channels"]),
            ("n_gaussians",    rs["n_gaussians"]),
        ]))
        out.append("")

        out.append("\n### 1.2 Dataset & patch grid\n")
        out.append(self._kv_table([
            ("split",             rs["split"]),
            ("split_region",      rs["split_region"]),
            ("global_crop",       rs["global_crop"]),
            ("secondary_labels",  rs["secondary_labels"]),
            ("patches",           rs["patches"]),
            ("patch_size",        rs["patch_size"]),
            ("patch_stride",      rs["patch_stride"]),
            ("preprocessing_dir", rs["preprocessing_dir"]),
            ("x_axis_length",     rs["x_axis_length"]),
            ("x_axis_min",        rs["x_axis_min"]),
            ("x_axis_max",        rs["x_axis_max"]),
        ]))
        out.append("")

        input_cfg = rs["input_config"]
        if input_cfg:
            out.append("\n### 1.3 Input configuration\n")
            out.append(self._kv_table(list(input_cfg.items())))
            out.append("")

        out.append("\n### 1.4 Checkpoint\n")
        out.append(self._kv_table([
            ("epoch",         ck["epoch"]),
            ("best_epoch",    ck["best_epoch"]),
            ("best_val_loss", ck["best_val_loss"]),
        ]))
        out.append("")

        out.append("\n### 1.5 Inference configuration\n")
        out.append(self._kv_table([
            ("device",             cfg["device"]),
            ("batch_size",         cfg["batch_size"]),
            ("num_workers",        cfg["num_workers"]),
            ("stitch_window",      cfg["stitch_window"]),
            ("cube_dtype",         cfg["cube_dtype"]),
            ("save_cubes",         cfg["save_cubes"]),
            ("save_plots",         cfg["save_plots"]),
            ("save_animations",    cfg["save_animations"]),
            ("n_best_profiles",    cfg["n_best_profiles"]),
            ("n_worst_profiles",   cfg["n_worst_profiles"]),
            ("n_random_profiles",  cfg["n_random_profiles"]),
            ("n_range_slices",     cfg["n_range_slices"]),
            ("n_azimuth_slices",   cfg["n_azimuth_slices"]),
            ("n_elevation_slices", cfg["n_elevation_slices"]),
            ("gif_axes",           cfg["gif_axes"]),
            ("gif_fps",            cfg["gif_fps"]),
            ("gif_max_frames",     cfg["gif_max_frames"]),
        ]))
        out.append("")

        return out

    def _build_headline_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 2. Headline metrics\n"]

        out.append("**Curve-level** (aggregated over the full stitched test cube)\n")
        out.append(self._three_col_table([
            ("MSE",     gm["curve_mse_gt"],   "Mean squared error over all (elev, az, rg)"),
            ("MAE",     gm["curve_mae_gt"],   "Mean absolute error"),
            ("RMSE",    gm["curve_rmse_gt"],  "Root mean squared error"),
            ("R\u00b2", gm["overall_r2_gt"],  "Global coefficient of determination"),
            ("PSNR dB", gm["psnr_db_gt"],     "Peak signal-to-noise ratio"),
        ]))
        out.append("")

        out.append("**Per-pixel** (computed per (az, rg) pixel over the elevation axis)\n")
        out.append(self._three_col_table([
            ("MSE mean",        gm["pixel_mse_gt_mean"],              "Mean of per-pixel MSE"),
            ("MSE median",      gm["pixel_mse_gt_median"],            "Median of per-pixel MSE"),
            ("MSE p95",         gm["pixel_mse_gt_p95"],               "95th percentile"),
            ("MAE mean",        gm["pixel_mae_gt_mean"],              "Mean of per-pixel MAE"),
            ("R\u00b2 mean",    gm["pixel_r2_gt_mean"],               "Mean of per-pixel R\u00b2"),
            ("R\u00b2 median",  gm["pixel_r2_gt_median"],             "Median of per-pixel R\u00b2"),
            ("R\u00b2 p5",      gm["pixel_r2_gt_p5"],                 "5th percentile (worst tail)"),
            ("Cosine mean",     gm["pixel_cosine_gt_mean"],           "Mean cosine similarity"),
            ("Cosine median",   gm["pixel_cosine_gt_median"],         "Median cosine similarity"),
            ("Peak err mean",   gm["pixel_peak_err_units_mean_gt"],   "|\u0394 peak| in elevation units"),
            ("Peak err median", gm["pixel_peak_err_units_median_gt"], "Median"),
            ("Peak err p95",    gm["pixel_peak_err_units_p95_gt"],    "95th percentile"),
        ]))
        out.append("")

        out.append("**SSIM** (denorm, mean over all slices of that axis)\n")
        out.append(self._three_col_table([
            ("SSIM elev mean",    gm["ssim_gt_elev_mean"],    "H\u00d7W intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_gt_range_mean"],   "n_elev\u00d7H cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_gt_azimuth_mean"], "n_elev\u00d7W cross-sectional planes"),
        ]))
        out.append("")

        out.append("**SSIM** (unit-area normalised pred vs GT, mean over all slices of that axis)\n")
        out.append(self._three_col_table([
            ("SSIM elev mean",    gm["ssim_norm_elev_mean"],    "H\u00d7W intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_norm_range_mean"],   "n_elev\u00d7H cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_norm_azimuth_mean"], "n_elev\u00d7W cross-sectional planes"),
        ]))
        out.append("")

        out.append("**Per-elevation-bin** (mean over all elevation bins)\n")
        out.append(self._three_col_table([
            ("MAE mean",           gm["elev_mae_gt_mean"],   "Mean absolute error averaged over elevation"),
            ("RMSE mean",          gm["elev_rmse_gt_mean"],  "Root mean squared error averaged over elevation"),
            ("R\u00b2 mean",       gm["elev_r2_gt_mean"],    "Coefficient of determination averaged over elevation"),
            ("Cross-entropy mean", gm["elev_ce_gt_mean"],    "Cross-entropy (normalised profiles) averaged over elevation"),
        ]))
        out.append("")

        self._subsection_index = 3

        out += self._build_active_count_headline()
        out += self._build_param_population_headline()
        out += self._build_matched_headline()
        out += self._build_slot_organization_headline()
        out += self._build_tracks_table()
        out += self._build_track_positions_table()
        out += self._build_jepa_headline()
        out += self._build_reduced_headline()
        out += self._build_data_consistency_headline()

        return out

    def _next_subsection(self) -> str:
        self._subsection_index += 1
        return f"2.{self._subsection_index}"

    def _build_active_count_headline(self) -> List[str]:
        gm = self.global_metrics
        if "active_frac_gt" not in gm:
            return []

        n_K = self.run_summary["n_gaussians"]
        out = [f"\n### {self._next_subsection()} Slot occupancy (GT vs Pred)\n"]
        out.append(
            "Per-slot and overall fraction of pixels carrying an active Gaussian (amplitude above the active threshold), "
            "ground truth versus prediction, over the full stitched test cube. Because Hungarian matching "
            "makes the training loss invariant to predicted slot ordering, the raw predicted slot index is "
            "arbitrary and not comparable to the GT slot index. The per-slot predicted fractions are therefore "
            "reported on the Hungarian-aligned prediction (each predicted Gaussian placed in the GT slot it "
            "matches, as in §6), so row k compares like with like. The overall fraction and mean active count "
            "are permutation-invariant aggregates and are unaffected by the alignment.\n"
        )

        rows = [
            ("Active fraction (all slots)", gm["active_frac_gt"],       gm["active_frac_pred"],       "Share of slot-pixels that are active"),
            ("Mean active count / pixel",   gm["active_count_gt_mean"], gm["active_count_pred_mean"], "Average number of active Gaussians per pixel"),
        ]
        for k in range(n_K):
            rows.append((f"Slot {k} active fraction", gm[f"slot_{k}_active_gt_frac"], gm[f"slot_{k}_active_pred_frac"], "Active-pixel fraction for this slot"))

        out.append(self._gt_pred_table(rows))
        out.append("")

        out.append("**Active-count agreement** (predicted vs GT number of active Gaussians per pixel)\n")
        agree_rows = [
            ("Exact count", gm["count_exact_frac"], "Pixels where pred count == GT count"),
            ("Undercount",  gm["count_under_frac"], "Pixels where pred count < GT count"),
            ("Overcount",   gm["count_over_frac"],  "Pixels where pred count > GT count"),
        ]
        for k in range(1, n_K + 1):
            key = f"count_acc_gt{k}"
            if key in gm:
                agree_rows.append((f"Accuracy | GT count = {k}", gm[key], f"Fraction correct among pixels whose GT has {k} active"))

        out.append(self._three_col_table(agree_rows, header=("Metric", "Fraction", "Description")))
        out.append("")

        return out

    def _build_param_population_headline(self) -> List[str]:
        gm = self.global_metrics
        if "slot_0_amp_active_gt_mean" not in gm:
            return []

        n_K = self.run_summary["n_gaussians"]
        out = [f"\n### {self._next_subsection()} Per-slot parameter statistics (GT vs Pred, by activity)\n"]
        out.append(
            "Mean ± std of each Gaussian parameter per GT slot, split by that slot's GT activity "
            "(amplitude above the active threshold): the active population, its inactive complement, and "
            "all pixels. GT amplitudes are averaged over all three populations, so the inactive row exposes "
            "the placeholder fill and the active-vs-global gap shows how much it dilutes unconditioned "
            "averages; GT μ and σ exist only where the slot is active, since inactive GT slots carry no "
            "position or width, and their inactive and global cells are therefore left empty rather than "
            "restated from the active population. Pred columns average the Hungarian-aligned emissions "
            "placed in this slot (as in §6); pixels where the model emitted no active Gaussian for the slot "
            "carry no value and are excluded — §2.4 gives the corresponding emission fractions. The inactive "
            "Pred rows therefore characterise the model's false fires in slots the GT leaves empty.\n"
        )

        for k in range(n_K):
            out.append(f"**Slot {k}** (GT-active fraction {self._fmt(gm.get(f'slot_{k}_active_gt_frac', float('nan')))})\n")

            table = MarkdownTable(("Parameter", "Population", "GT mean", "GT std", "Pred mean", "Pred std"))
            for param, label in (("amp", "a"), ("mu", "μ"), ("sig", "σ")):
                for scope in ("active", "inactive", "global"):
                    gt_measured = param == "amp" or scope == "active"
                    table.add_row(
                        label,
                        scope,
                        self._fmt(gm[f"slot_{k}_{param}_{scope}_gt_mean"]) if gt_measured else "—",
                        self._fmt(gm[f"slot_{k}_{param}_{scope}_gt_std"])  if gt_measured else "—",
                        self._fmt(gm.get(f"slot_{k}_{param}_{scope}_pred_mean", float("nan"))),
                        self._fmt(gm.get(f"slot_{k}_{param}_{scope}_pred_std",  float("nan"))),
                    )

            out += table.render()
            out.append("")

        return out

    def _build_matched_headline(self) -> List[str]:
        gm = self.global_metrics
        if "matched_mu_mae" not in gm:
            return []

        n_K = self.run_summary["n_gaussians"]
        out = [f"\n### {self._next_subsection()} Permutation-invariant matched Gaussian errors\n"]
        out.append(
            "Each pixel's predicted Gaussians are Hungarian-matched to its GT Gaussians on |Δμ| before "
            "scoring, so the errors are independent of slot ordering. A match counts as a detection hit when "
            f"|Δμ| ≤ {self._fmt(gm['matched_tol'])} elevation units.\n"
        )
        out.append(self._three_col_table([
            ("Matched μ MAE",   gm["matched_mu_mae"],    "Mean |Δμ| over matched pairs (all counts)"),
            ("Matched μ RMSE",  gm["matched_mu_rmse"],   "Root mean squared Δμ"),
            ("Matched σ MAE",   gm["matched_sig_mae"],   "Mean |Δσ| over matched pairs"),
            ("Matched a MAE",   gm["matched_amp_mae"],   "Mean |Δa| over matched pairs"),
            ("Detection recall",     gm["matched_recall"],    "Share of GT Gaussians matched within tolerance"),
            ("Detection precision",  gm["matched_precision"], "Share of predicted Gaussians matching a GT within tolerance"),
            ("Detection F1",         gm["matched_f1"],        "Harmonic mean of recall and precision"),
        ], header=("Metric", "Value", "Description")))
        out.append("")

        present = [k for k in range(1, n_K + 1) if f"matched_recall_gt{k}" in gm or f"matched_mu_mae_gt{k}" in gm]

        if present:
            out.append("**By GT count** (recall = recovery; precision = filler-free; μ MAE = placement)\n")
            table = MarkdownTable(("GT count", "Recall (higher better)", "Precision (higher better)", "Matched μ MAE (lower better)"))
            for k in present:
                table.add_row(
                    str(k),
                    self._fmt(gm.get(f"matched_recall_gt{k}",    float("nan"))),
                    self._fmt(gm.get(f"matched_precision_gt{k}", float("nan"))),
                    self._fmt(gm.get(f"matched_mu_mae_gt{k}",    float("nan"))),
                )
            out += table.render()
            out.append("")

        return out

    def _build_slot_organization_headline(self) -> List[str]:
        gm = self.global_metrics
        if "slot_mu_rank_diag" not in gm:
            return []

        n_K = self.run_summary["n_gaussians"]
        out = [f"\n### {self._next_subsection()} Slot internal organization (emergent)\n"]
        out.append(
            "Training Hungarian-matches predicted slots to GT before scoring, so the loss is invariant to "
            "predicted slot ordering: the model gets no gradient fixing a role to an output channel. These "
            "metrics, computed on the raw unsorted predicted slots, measure whether a consistent organization "
            "emerged anyway. μ-rank diagonality is the fraction of active predicted slots whose channel index "
            "equals their μ-rank within the pixel; slot↔GT alignment is the same against the μ-sorted GT index. "
            f"1.0 means the model spontaneously emits slots in height order; chance is about {1.0 / n_K:.3f}.\n"
        )

        rows = [
            ("Usage entropy (norm.)", gm["slot_usage_entropy"], "Evenness of per-slot activation; 1 = all slots used equally, 0 = a single slot"),
            ("μ-rank diagonality",    gm["slot_mu_rank_diag"],  "Fraction of active slots whose channel index equals their μ-rank"),
        ]
        if "slot_gt_alignment" in gm:
            rows.append(("Slot↔GT alignment", gm["slot_gt_alignment"], "Fraction of matched slots whose channel index equals the matched μ-sorted GT index"))

        out.append(self._three_col_table(rows, header=("Metric", "Value", "Description")))
        out.append("")

        return out

    def _build_tracks_table(self) -> List[str]:
        tracks = self.global_metrics.get("tracks")
        if not isinstance(tracks, dict):
            return []

        out = [f"\n### {self._next_subsection()} Tracks used in this run\n"]
        out.append(
            f"Baselines relative to the reference pass `{tracks['reference']}` over azimuth window "
            f"{tracks['azimuth_window']}; absolute values are the resa-frame windowed means.\n"
        )

        labels   = tracks["labels"]
        n_tracks = len(labels)

        table = MarkdownTable(("Pass", "Horizontal [m]", "Vertical [m]", "H std [m]", "V std [m]", "H absolute [m]", "V absolute [m]"))
        for row in zip(
            labels,
            self._padded_column(tracks, "horizontal",          n_tracks),
            self._padded_column(tracks, "vertical",            n_tracks),
            self._padded_column(tracks, "horizontal_std",      n_tracks),
            self._padded_column(tracks, "vertical_std",        n_tracks),
            self._padded_column(tracks, "horizontal_absolute", n_tracks),
            self._padded_column(tracks, "vertical_absolute",   n_tracks),
        ):
            table.add_row(row[0], *(self._fmt(value) for value in row[1:]))

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _build_track_positions_table(self) -> List[str]:
        positions = self.global_metrics.get("track_positions")
        if not isinstance(positions, dict):
            return []

        out = [f"\n### {self._next_subsection()} Track positions and temporal deviation\n"]
        out.append(
            f"Absolute mean positions in the resa track frame over {positions['n_samples']} azimuth samples "
            f"starting at index {positions['azimuth_start']}. Spans are peak-to-peak excursions per component; "
            f"the planar deviation is the distance of each sample from the per-pass mean position in the "
            f"horizontal-vertical plane, summarising how much the pass drifted over time.\n"
        )

        labels   = positions["labels"]
        n_tracks = len(labels)

        table = MarkdownTable(("Pass", "H mean [m]", "V mean [m]", "H span [m]", "V span [m]", "Planar dev RMS [m]", "Planar dev max [m]"))
        for row in zip(
            labels,
            self._padded_column(positions, "horizontal_mean", n_tracks),
            self._padded_column(positions, "vertical_mean",   n_tracks),
            self._padded_column(positions, "horizontal_span", n_tracks),
            self._padded_column(positions, "vertical_span",   n_tracks),
            self._padded_column(positions, "deviation_rms",   n_tracks),
            self._padded_column(positions, "deviation_max",   n_tracks),
        ):
            table.add_row(row[0], *(self._fmt(value) for value in row[1:]))

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _build_jepa_headline(self) -> List[str]:
        gm = self.global_metrics
        if "jepa_embedding_mse" not in gm:
            return []

        out = [f"\n### {self._next_subsection()} JEPA embedding diagnostics\n"]
        out.append(
            "The JEPA objective is measured directly in embedding space: the backbone's predicted profile "
            "embedding is compared against the checkpoint encoder's embedding of the GT curve, both in the "
            "decoder's normalised embedding space. The decoder-only error isolates the profile-autoencoder "
            "reconstruction floor; the full-chain error is the decoded prediction against the GT curve in "
            "normalised curve space, so chain − decoder gauges how much error the predictor itself adds.\n"
        )
        out.append(self._three_col_table([
            ("Embedding MSE",           gm["jepa_embedding_mse"],    "Predicted vs target embedding, normalised space"),
            ("Embedding cosine",        gm["jepa_embedding_cosine"], "Mean cosine similarity along the embedding dimension"),
            ("Decoder-only MSE (norm)", gm["jepa_decode_mse_norm"],  "decode(encode(GT)) vs GT, normalised curves"),
            ("Full-chain MSE (norm)",   gm["jepa_chain_mse_norm"],   "decode(predicted embedding) vs GT, normalised curves"),
        ], header=("Metric", "Value", "Description")))
        out.append("")

        return out

    def _build_reduced_headline(self) -> List[str]:
        gm = self.global_metrics
        if "improvement_pixel_mse_mean" not in gm:
            return []

        out = [f"\n### {self._next_subsection()} NN improvement over the classical baseline\n"]
        out.append(
            "The reduced tomogram is a classical Capon reconstruction from the primary plus the exact "
            "secondary subset this run was trained on, processed identically to the ground-truth full-stack "
            "tomogram (same pyrat `fusartomo`, height range, filter and stack). It is what tomography yields "
            "without the network. Reduced-vs-GT and improvement quantities are computed on per-elevation-profile "
            "unit-area-normalised cubes, so they measure elevation-distribution agreement independent of absolute "
            "Capon power calibration.\n"
        )
        out.append(self._three_col_table([
            ("Reduced MSE mean",       gm["reduced_pixel_mse_norm_mean"],  "Per-pixel reduced-vs-GT MSE (classical baseline)"),
            ("Pred MSE mean",          gm["pred_pixel_mse_norm_mean"],     "Per-pixel pred-vs-GT MSE (neural network)"),
            ("Improvement mean",       gm["improvement_pixel_mse_mean"],   "MSE(reduced) − MSE(pred), higher is better"),
            ("Improvement median",     gm["improvement_pixel_mse_median"], "Median per-pixel improvement"),
            ("Relative MSE reduction", gm["relative_mse_reduction"],       "1 − MSE(pred)/MSE(reduced)"),
            ("Pixels where NN wins",   gm["fraction_pred_beats_reduced"],  "Fraction with pred MSE < reduced MSE"),
        ], header=("Metric", "Value", "Description")))
        out.append("")

        out.append("**Reduced vs GT** (classical baseline, unit-area profiles)\n")
        out.append(self._three_col_table([
            ("Curve MSE",         gm["curve_mse_red"],         "Mean squared error over all (elev, az, rg)"),
            ("Overall R2",        gm["overall_r2_red"],        "Global coefficient of determination"),
            ("SSIM elev mean",    gm["ssim_red_elev_mean"],    "Intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_red_range_mean"],   "Cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_red_azimuth_mean"], "Cross-sectional planes"),
        ], header=("Metric", "Reduced vs GT", "Description")))
        out.append("")

        return out

    def _build_data_consistency_headline(self) -> List[str]:
        gm = self.global_metrics
        if "physics_coherence_error_mean" not in gm:
            return []

        out = [f"\n### {self._next_subsection()} Interferometric data consistency\n"]
        out.append(
            "The predicted and GT elevation profiles are reprojected to normalised track coherences and "
            "pair covariances through the per-pixel kz geometry field — the same steering machinery the "
            "physics loss trains on. Phase agreement compares the multilooked measured interferogram phase "
            "against the coherence phase synthesised from the curves; the flipped column uses the conjugate "
            "steering and should stay below the aligned one when the kz sign convention is correct.\n"
        )
        out.append(self._three_col_table([
            ("Coherence error mean",       gm["physics_coherence_error_mean"],   "Mean |Δγ|² over tracks, pred vs GT, power-masked"),
            ("Coherence error p95",        gm["physics_coherence_error_p95"],    "95th percentile of the per-pixel error"),
            ("Covariance error mean",      gm["physics_covariance_error_mean"],  "Relative pair-covariance mismatch, pred vs GT"),
            ("Valid fraction",             gm["physics_valid_fraction"],         "Pixels above the physics power floor"),
            ("Phase agreement GT",         gm["phase_agreement_gt_mean"],        "Measured phase vs GT-synthesised coherence"),
            ("Phase agreement GT flipped", gm["phase_agreement_gt_flipped_mean"],"Same with conjugate steering (kz sign check)"),
            ("Phase agreement pred",       gm["phase_agreement_pred_mean"],      "Measured phase vs pred-synthesised coherence"),
        ], header=("Metric", "Value", "Description")))
        out.append("")

        return out

    _IMPROVEMENT_KEYS = frozenset({
        "improvement_pixel_mse_mean", "improvement_pixel_mse_median",
        "fraction_pred_beats_reduced", "relative_mse_reduction",
        "pred_pixel_mse_norm_mean", "reduced_pixel_mse_norm_mean",
    })

    _METRIC_SKIP_KEYS = ("tracks", "track_positions", "split", "split_region")

    _SLOT_ORGANIZATION_KEYS = frozenset({"slot_usage_entropy", "slot_mu_rank_diag", "slot_gt_alignment"})

    @staticmethod
    def _is_reduced_key(k: str) -> bool:
        return ("_red" in k) or k.startswith("reduced_") or k.startswith("ssim_red") or ("predn" in k)

    _METRIC_TAXONOMY = (
        ("3.1 Dataset statistics",                               lambda k: k in Report._DATASET_KEYS or k.endswith("_status")),
        ("3.11 Interferometric data consistency",                lambda k: k.startswith(("physics_", "phase_agreement_"))),
        ("3.12 JEPA embedding diagnostics",                      lambda k: k.startswith("jepa_")),
        ("3.13 Label quality (GT fit vs raw tomogram)",          lambda k: k.startswith("label_")),
        ("3.14 Normalization & clamp health",                    lambda k: k.startswith(("norm_in_", "clamp_"))),
        ("3.15 Flip-equivariance consistency",                   lambda k: k.startswith("flip_")),
        ("3.16 Stratified errors (by scene covariate)",          lambda k: k.startswith("strat")),
        ("3.17 Failure-mode classification",                     lambda k: k.startswith("failure_")),
        ("3.18 Presence calibration",                            lambda k: k.startswith("presence_")),
        ("3.10 Matched Gaussian errors (permutation-invariant)", lambda k: k.startswith("matched_")),
        ("3.9c Slot internal organization (emergent)",           lambda k: k in Report._SLOT_ORGANIZATION_KEYS),
        ("3.9b Per-slot parameter statistics by activity",       lambda k: k.startswith("slot_") and ("_amp_" in k or "_mu_" in k or "_sig_" in k)),
        ("3.9 Slot occupancy & active count",                    lambda k: k.startswith(("active_frac", "active_count", "count_")) or (k.startswith("slot_") and "_active_" in k)),
        ("3.8 NN improvement over baseline",                     lambda k: k in Report._IMPROVEMENT_KEYS),
        ("3.7 SSIM summaries (reduced vs GT)",                   lambda k: k.startswith("ssim_red")),
        ("3.6 Per-pixel (reduced vs GT)",                        lambda k: Report._is_reduced_key(k) and k.startswith("pixel_")),
        ("3.5 Curve-level (reduced vs GT)",                      lambda k: Report._is_reduced_key(k)),
        ("3.3 Per-pixel (GT)",                                   lambda k: k.startswith("pixel_")),
        ("3.4b SSIM summaries (normalised)",                     lambda k: k.startswith("ssim_norm")),
        ("3.4 SSIM summaries (denorm)",                          lambda k: k.startswith("ssim_")),
        ("3.2 Curve-level (GT)",                                 lambda k: True),
    )

    @staticmethod
    def _section_order(title: str) -> tuple:
        number = title.split()[0].split(".")[1]
        digits = "".join(ch for ch in number if ch.isdigit())
        return (int(digits), number[len(digits):])

    @classmethod
    def group_metric_keys(cls, keys) -> List[Tuple[str, List[str]]]:
        groups: Dict[str, List[str]] = {title: [] for title, _match in cls._METRIC_TAXONOMY}

        for key in sorted(keys):
            if cls._is_per_slice_ssim(key) or key in cls._METRIC_SKIP_KEYS:
                continue

            title = next(title for title, match in cls._METRIC_TAXONOMY if match(key))
            groups[title].append(key)

        ordered = sorted((title for title in groups if groups[title]), key=cls._section_order)
        return [(title, groups[title]) for title in ordered]

    def _build_full_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 3. Full metric tables\n"]

        for title, keys in self.group_metric_keys(gm):
            out.append(f"\n### {title}\n")
            out.append(self._dict_table({key: gm[key] for key in keys}))
            out.append("")

        return out

    def _section(self, out: List[str], groups) -> None:
        fp = self.figure_paths

        for key, title in groups:
            if fp.get(key):
                out.append(f"\n### {title}\n")
                out += self.assets.images(key, fp[key])

    def _numbered_section(self, out: List[str], prefix: str, groups) -> None:
        fp        = self.figure_paths
        available = [(key, title) for key, title in groups if fp.get(key)]

        for n, (key, title) in enumerate(available, start=1):
            out.append(f"\n### {prefix}{n} {title}\n")
            out += self.assets.images(key, fp[key])

    def _build_figures(self) -> List[str]:
        fp  = self.figure_paths
        gp  = self.gif_paths
        out = []

        track_groups = [(key, title) for key, title in (
            ("track_geometry",  "Baseline geometry of the passes used"),
            ("track_profiles",  "Per-azimuth baseline profiles"),
            ("track_flight_3d", "3D flight tracks with temporal deviation envelopes"),
            ("input_channels",  "Pass amplitudes and interferogram phases over the inference region"),
        ) if fp.get(key)]

        if track_groups:
            out.append("\n## 3b. Passes and interferograms used\n")
            out.append(
                "The model inputs for this run: the primary, the selected secondary passes and "
                "their interferograms against the primary.\n"
            )
            self._numbered_section(out, "3b.", track_groups)

        profile_groups = (
            ("profiles_best",   "4.1 Best-fit profiles (lowest MSE)"),
            ("profiles_worst",  "4.2 Worst-fit profiles (highest MSE)"),
            ("profiles_random", "4.3 Random profiles"),
        )
        if any(fp.get(key) for key, _title in profile_groups):
            out.append("\n## 4. Profile reconstructions\n")
            out.append(
                "Each panel overlays the GT profile (black solid), "
                "prediction (red dashed) and individual Gaussian components. "
                "The shaded area shows the signed residual (pred \u2212 gt). "
                "The raw full-stack tomogram that the GT Gaussians are fit to is drawn as a grey reference. "
                "When a reduced baseline is available, its classical-Capon profile is overlaid "
                "(green dotted, rescaled to the GT peak) so the network's gain in elevation shape is visible.\n"
            )
            self._section(out, profile_groups)

        pixel_groups = (
            ("pixel_mse_map",       "5.1 MSE map (log scale, pred vs GT)"),
            ("pixel_r2_map",        "5.2 R\u00b2 map (pred vs GT)"),
            ("pixel_peak_map",      "5.3 Peak-location error map (|\u0394 peak index|)"),
            ("label_r2_map",          "5.4 Label fit R\u00b2 map (GT curves vs raw tomogram)"),
            ("flip_consistency_map",  "5.5 Flip-equivariance disagreement map"),
            ("stratified_errors",     "5.6 Error stratified by scene covariates"),
            ("failure_mode_map",      "5.7 Dominant failure mode per pixel"),
            ("miss_by_separation",    "5.8 Miss rate by GT scatterer separation"),
            ("metric_histograms",     "5.9 Metric distributions"),
        )
        if any(fp.get(key) for key, _title in pixel_groups):
            out.append("\n## 5. Per-pixel metric maps\n")
            out.append(
                "5.4 scores the labels themselves: the R\u00b2 of the fitted GT Gaussians against the raw Capon tomogram profile, "
                "so low-R\u00b2 areas in the model error maps above can be read against low label quality rather than model failure. "
                "5.5, when computed, is the disagreement between the prediction and the unflipped prediction on flipped inputs, "
                "a label-free confidence signal.\n"
            )
            self._section(out, pixel_groups)

        param_groups = (
            ("param_distributions",    "6.1 Parameter distributions (GT vs Pred)"),
            ("param_scatter",          "6.2 Parameter scatter plots (GT vs Pred, with R²)"),
            ("param_error_maps",       "6.3 Parameter absolute-error maps |Pred − GT|"),
            ("param_error_hists",      "6.4 Parameter error histograms (matched Pred − GT)"),
            ("active_count_map",       "6.5 Active Gaussian count map"),
            ("presence_reliability",   "6.6 Presence reliability (precision by predicted amplitude)"),
        )
        if any(fp.get(key) for key, _title in param_groups):
            out.append("\n## 6. Gaussian parameter analysis\n")
            out.append(
                "All parameter figures are permutation-invariant: the distributions (6.1), scatter (6.2), "
                "error maps (6.3) and error histograms (6.4) Hungarian-match predicted Gaussians to GT Gaussians "
                "per pixel before scoring, and the active-count map (6.5) compares counts only. The error "
                "histograms show the signed per-pixel error, so a shift away from the zero line reads directly "
                "as parameter bias. For aggregate ordering-independent accuracy see §2.6.\n"
            )
            self._section(out, param_groups)

        org_groups = [(key, title) for key, title in (
            ("slot_usage",      "Per-slot activation frequency"),
            ("slot_param_dist", "Per-slot parameter distributions"),
            ("slot_mu_rank",    "Slot-index vs μ-rank matrix"),
            ("slot_assignment", "Slot-index vs matched GT-index matrix"),
        ) if fp.get(key)]

        if org_groups:
            out.append("\n## 6b. Slot internal organization (emergent under Hungarian matching)\n")
            out.append(
                "Analyses of the raw, unsorted predicted slots. Because Hungarian matching makes the training "
                "loss permutation-invariant over slot index, any structure here is self-organised, not imposed. "
                "Usage shows per-slot activation; the per-slot distributions reveal whether each slot specialises "
                "in a height (μ), width (σ) or amplitude band; the μ-rank and matched-GT matrices show whether "
                "the channel index tracks height order (a diagonal means the model spontaneously emits slots "
                "sorted by μ). See §2.7 for the scalar summaries.\n"
            )
            self._numbered_section(out, "6b.", org_groups)

        slice_groups = [(key, title) for key, title in (
            ("slices_range",        "Range cuts"),
            ("slices_azimuth",      "Azimuth cuts"),
            ("slices_elev",         "Elevation cuts"),
            ("slices_range_norm",   "Range cuts (unit-area)"),
            ("slices_azimuth_norm", "Azimuth cuts (unit-area)"),
            ("slices_elev_norm",    "Elevation cuts (unit-area)"),
        ) if fp.get(key)]

        if slice_groups:
            out.append("\n## 7. Tomogram slices\n")
            out.append(
                "GT and prediction share a colour scale; "
                "error figures are clipped at p99 of that slice. "
                "SSIM (pred vs GT) is shown in the prediction title. "
                "The raw full-stack tomogram (the reference the GT Gaussians are fit to) is shown as an extra panel. "
                "Both denormalised slices and unit-area-normalised slices (each elevation profile normalised to unit area) are shown.\n"
            )
            self._numbered_section(out, "7.", slice_groups)

        ssim_groups = (
            ("ssim_range",        "8.1 SSIM along range axis (denorm)"),
            ("ssim_azimuth",      "8.2 SSIM along azimuth axis (denorm)"),
            ("ssim_elev",         "8.3 SSIM along elevation axis (denorm)"),
            ("ssim_range_norm",   "8.4 SSIM along range axis (unit-area)"),
            ("ssim_azimuth_norm", "8.5 SSIM along azimuth axis (unit-area)"),
            ("ssim_elev_norm",    "8.6 SSIM along elevation axis (unit-area)"),
        )
        if any(fp.get(key) for key, _title in ssim_groups) or fp.get("elev_metric_curves"):
            out.append("\n## 8. SSIM curves\n")
            out.append("SSIM plotted for every slice along each axis \u2014 pred vs GT, both denormalised and unit-area-normalised.\n")
            self._section(out, ssim_groups)

        if fp.get("elev_metric_curves"):
            out.append("\n### 8.7 Per-elevation-bin metrics (MAE, RMSE, R\u00b2, cross-entropy)\n")
            out.append(
                "Each panel shows a metric aggregated over all (az\u00d7rg) pixels for every "
                "elevation bin (pred vs GT). "
                "Dashed lines mark the mean over all bins.\n"
            )
            out += self.assets.images("elev_metric_curves", fp["elev_metric_curves"])

        out += self._build_reduced_figures()
        out += self._build_data_consistency_figures()

        if gp:
            out.append("\n## 10. Animations\n")
            out.append(
                "Each GIF walks through one axis of the stitched test cube. "
                "The colour scale is fixed across frames.\n"
            )
            for n, (name, path) in enumerate(sorted(gp.items()), start=1):
                out.append(f"\n### 10.{n} `{name}`\n")
                out += self.assets.image(name, path)

        return out

    def _build_reduced_figures(self) -> List[str]:
        fp = self.figure_paths

        keys = (
            "improvement_map", "reduced_pixel_mse_map",
            "slices_range_reduced", "slices_azimuth_reduced", "slices_elev_reduced",
            "ssim_range_reduced", "ssim_azimuth_reduced", "ssim_elev_reduced",
            "elev_metric_curves_reduced",
        )
        if not any(fp.get(k) for k in keys):
            return []

        out = ["\n## 9. Classical baseline (reduced Capon) vs ground truth\n"]
        out.append(
            "The reduced tomogram is a classical Capon reconstruction from the primary plus the exact "
            "secondary subset this run was trained on — what tomography yields without the network. "
            "Reduced and GT cubes are unit-area-normalised per elevation profile before comparison, so "
            "panels and metrics measure elevation-distribution agreement. Read the network's gain by "
            "contrasting these errors with the pred-vs-GT figures in sections 5, 7 and 8.\n"
        )

        self._section(out, (
            ("improvement_map",       "9.1 Per-pixel NN improvement map (MSE reduced − MSE pred)"),
            ("reduced_pixel_mse_map", "9.2 Per-pixel reduced-vs-GT MSE map"),
        ))

        slice_groups = [(key, title) for key, title in (
            ("slices_range_reduced",   "Range cuts (GT, reduced, error)"),
            ("slices_azimuth_reduced", "Azimuth cuts (GT, reduced, error)"),
            ("slices_elev_reduced",    "Elevation cuts (GT, reduced, error)"),
        ) if fp.get(key)]

        if slice_groups:
            out.append("\n### 9.3 Reduced tomogram slices\n")
            self._numbered_section(out, "9.3.", slice_groups)

        self._section(out, (
            ("ssim_range_reduced",   "9.4 Reduced SSIM along range axis"),
            ("ssim_azimuth_reduced", "9.5 Reduced SSIM along azimuth axis"),
            ("ssim_elev_reduced",    "9.6 Reduced SSIM along elevation axis"),
        ))

        if fp.get("elev_metric_curves_reduced"):
            out.append("\n### 9.7 Reduced per-elevation-bin metrics (MAE, RMSE, R², cross-entropy)\n")
            out += self.assets.images("elev_metric_curves_reduced", fp["elev_metric_curves_reduced"])

        return out

    def _build_data_consistency_figures(self) -> List[str]:
        fp = self.figure_paths

        keys = ("coherence_error_map", "covariance_error_map", "phase_agreement")
        if not any(fp.get(k) for k in keys):
            return []

        out = ["\n## 9b. Interferometric data consistency\n"]
        out.append(
            "Predicted and GT profiles are reprojected to track coherences and pair covariances through the "
            "per-pixel kz geometry field, and the measured multilooked interferogram phase is compared against "
            "the coherence phase synthesised from the curves. In the phase-agreement panels, the aligned bars "
            "should dominate the flipped ones; the reverse indicates an inverted kz sign convention "
            "(capon_phase_sign) for the stack.\n"
        )

        self._section(out, (
            ("coherence_error_map",  "9b.1 Per-pixel coherence-resynthesis error (pred vs GT)"),
            ("covariance_error_map", "9b.2 Per-pixel covariance-matching error (pred vs GT)"),
            ("phase_agreement",      "9b.3 Measured-interferogram phase agreement per secondary pass"),
            ("physics_spectroscopy", "9b.4 Residual spectroscopy: error and phase spread by mean |kz|"),
        ))

        return out

    def assemble(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_path

        lines: List[str] = []
        lines.append("# TomoSAR Inference Report")
        lines.append("")
        lines.append(f"_Generated on {RunTag.timestamp()}_")
        lines.append("")

        lines += self._build_run_summary()
        lines += self._build_headline_metrics()
        lines += self._build_full_metrics()
        lines += self._build_figures()

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

