"""Tests covering inference report payloads, metric taxonomy routing and markdown assembly.

Metrics are produced from synthetic two-Gaussian curves so the assembled report can be
checked section by section, including the optional extraction, JEPA and figure blocks.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from tools.data.regions                   import CropRegion
from pipelines.backbone.inference.metrics import Metrics, Result
from pipelines.backbone.inference.report  import Report, ReportPayloadBuilder
from tools.runtime.git_state              import GitState

from tests.conftest import SilentLogger


N_GAUSSIANS = 2
N_Z         = 12
H           = W       = 6


def _curves_from_params(params, z_axis, n_gaussians):
    """Renders Gaussian mixture curves from a parameter cube.

    Args:
        params      : Parameter cube of shape (3 * n_gaussians, z, width) holding amplitude, centre in metres and width in metres per slot.
        z_axis      : Z axis of shape (n_z,), in metres.
        n_gaussians : Number of Gaussian slots to sum.

    Returns:
        Curve cube of shape (n_z, z, width).
    """
    curve = np.zeros((z_axis.size, params.shape[1], params.shape[2]), dtype=np.float32)
    for k in range(n_gaussians):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = z_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return curve.astype(np.float32)


def _global_metrics():
    """Computes a full metric dictionary from noisy predictions of synthetic curves.

    Returns:
        Mapping of metric name to value, covering curve, pixel, slot and parameter
        families over a 6x6 patch with 12 z bins.
    """
    rng    = np.random.default_rng(0)
    z_axis = np.linspace(-20.0, 80.0, N_Z).astype(np.float32)
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    for k in range(N_GAUSSIANS):
        params[3 * k]     = rng.random((H, W)).astype(np.float32) + 0.5
        params[3 * k + 1] = float(k * 15)
        params[3 * k + 2] = 4.0

    gt   = _curves_from_params(params, z_axis, N_GAUSSIANS)
    pred = gt + 0.01 * rng.standard_normal(gt.shape).astype(np.float32)

    pixel = Metrics.curve_pixel_metrics(pred, gt)
    res   = Result(
        pred_curves        = pred,
        gt_curves          = gt,
        pixel_mse          = pixel["mse"],
        pixel_mae          = pixel["mae"],
        pixel_r2           = pixel["r2"],
        pixel_cosine       = pixel["cos"],
        pixel_peak_err_idx = pixel["peak"].astype(np.int32),
        cube_directory     = None,
        row_offset         = 0,
        col_offset         = 0,
        params_pred        = params.copy(),
        params_gt          = params.copy(),
    )

    m = Metrics(res, z_axis, N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())
    return m.compute(
        z_indices   = np.arange(N_Z),
        col_indices = np.arange(W),
        row_indices = np.arange(H),
        param_space = True,
    )


def _run_stub():
    """Returns a run stub carrying backbone name, channel counts, split region, grid, dataset config and no target dataset."""
    region = CropRegion(row_start=0, row_end=H, col_start=0, col_end=W)
    grid   = types.SimpleNamespace(number_of_patches=4, patch_size=(6, 6), stride=(6, 6))
    ds_cfg = types.SimpleNamespace(
        preprocessing_run_directory = "/pp/run",
        input_config                = types.SimpleNamespace(as_dict=lambda: {"use_amplitude": True}),
        batch_size                  = 8,
    )
    return types.SimpleNamespace(
        backbone_name    = "unet",
        in_channels      = 4,
        out_channels     = N_GAUSSIANS * 3,
        n_gaussians      = N_GAUSSIANS,
        x_axis_length    = N_Z,
        split_name       = "test",
        split_region     = region,
        global_crop      = region,
        grid             = grid,
        dataset_config   = ds_cfg,
        training_dataset = "/pp/run",
        target           = None,
        secondary_labels = ("CH01", "CH02"),
    )


def _cfg_stub():
    """Returns an inference config stub with plotting, slicing and device settings."""
    return types.SimpleNamespace(
        stitch_window=True, cube_dtype="float32", save_cubes=False, save_plots=True,
        save_animations=False, n_best_profiles=4, n_worst_profiles=4, n_random_profiles=4,
        n_col_slices=3, n_row_slices=3, n_z_slices=3, gif_axes=["col"],
        gif_fps=12, gif_duration=None, gif_max_frames=10, gif_separate_panels=False, device="cpu", num_workers=2,
        checkpoint_name="last_model.pt", region_override=None, render_amp_floor=0.02, extract_curve_params=True,
        extract_prominence_frac=0.05,
    )


def test_report_payload_run_summary_structure():
    """The run-summary payload carries model name, slot count, z axis bounds and secondary labels."""
    run     = _run_stub()
    z_axis  = np.linspace(-20.0, 80.0, N_Z)
    payload = ReportPayloadBuilder.run_summary(run, z_axis)

    assert payload["model_name"]       == "unet"
    assert payload["n_gaussians"]      == N_GAUSSIANS
    assert payload["x_axis_length"]    == N_Z
    assert payload["x_axis_min"]       == pytest.approx(-20.0)
    assert payload["secondary_labels"] == "CH01, CH02"
    assert isinstance(payload["input_config"], dict)


def test_report_payload_inference_config_structure():
    """The inference-config payload merges device, dataset batch size and figure settings."""
    payload = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub())

    assert payload["device"]      == "cpu"
    assert payload["batch_size"]  == 8
    assert payload["gif_axes"]    == ["col"]
    assert "n_best_profiles" in payload


def test_report_payload_inference_config_records_the_scoring_settings_and_commit():
    """The inference-config payload names the checkpoint, the settings that change the scores and the code commit."""
    payload = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub())

    assert payload["checkpoint_name"]  == "last_model.pt"
    assert payload["target_dataset"]   == "none (scored on its own split)"
    assert payload["render_amp_floor"] == 0.02
    assert payload["region_override"] is None
    assert payload["code_commit"]      == GitState.commit()


def test_report_assemble_writes_markdown(tmp_path):
    """Assembling writes a markdown report with the run summary, headline and full-metric sections."""
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 3, "best_epoch": 2, "best_val_loss": 0.1},
        global_metrics   = gm,
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )

    out = report.assemble()

    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "# ML Template Inference Report" in text
    assert "## 1. Run summary" in text
    assert "## 2. Headline metrics" in text
    assert "## 3. Full metric tables" in text
    assert "`model_name`" in text
    assert "unet" in text


def test_report_includes_slot_occupancy_section(tmp_path):
    """Slot-occupancy metrics produce their headline block and full-metric section."""
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 0, "best_epoch": 0, "best_val_loss": 1.0},
        global_metrics   = gm,
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )

    text  = report.assemble().read_text(encoding="utf-8")
    lines = report._build_full_metrics()

    assert "active_frac_gt" in gm
    assert "Slot occupancy (GT vs Pred)" in text
    assert "Active-count agreement" in text
    assert "3.9 Slot occupancy & active count" in "\n".join(lines)


def test_report_includes_param_population_section(tmp_path):
    """Per-slot parameter statistics produce their headline block and full-metric section."""
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 0, "best_epoch": 0, "best_val_loss": 1.0},
        global_metrics   = gm,
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )

    text  = report.assemble().read_text(encoding="utf-8")
    lines = report._build_full_metrics()

    assert "slot_0_amp_active_gt_mean" in gm
    assert "Per-slot parameter statistics (GT vs Pred, by activity)" in text
    assert "GT-active fraction" in text
    assert "3.9b Per-slot parameter statistics by activity" in "\n".join(lines)


def _report_for(tmp_path, metrics):
    """Builds a Report over the given metric dictionary with no figures attached.

    Args:
        tmp_path : Directory receiving the report.
        metrics  : Global metric dictionary the report is assembled from.

    Returns:
        A Report ready to assemble or to query section by section.
    """
    return Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 0, "best_epoch": 0, "best_val_loss": 1.0},
        global_metrics   = metrics,
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )


def test_report_includes_cube_ssim_section_only_when_the_sweep_ran(tmp_path):
    """The raw-cube SSIM subsection appears exactly when its metric keys are present, with both forms."""
    gm = _global_metrics()

    assert _report_for(tmp_path, gm)._build_cube_ssim_headline() == []

    sweep = {f"ssim_cube_{axis}_mean": 0.6 + 0.01 * i for i, axis in enumerate(("z", "col", "row"))}
    sweep.update({f"ssim_cube_norm_{axis}_mean": 0.7 + 0.01 * i for i, axis in enumerate(("z", "col", "row"))})
    text = "\n".join(_report_for(tmp_path, dict(gm, cube_ssim_status="computed", **sweep))._build_cube_ssim_headline())

    assert "SSIM against the raw cube" in text
    assert "0.61" in text
    assert "0.72" in text
    assert "Pred vs raw cube" in text
    assert "Pred vs GT" not in text


def test_report_routes_cube_ssim_keys_to_their_own_section():
    """Raw-cube SSIM summaries get section 3.4c and their per-slice values stay out of the tables."""
    assert _metric_group("ssim_cube_col_mean")    == "3.4c SSIM summaries (prediction vs raw cube)"
    assert _metric_group("ssim_cube_norm_z_mean") == "3.4c SSIM summaries (prediction vs raw cube)"
    assert _metric_group("cube_ssim_status")      == "3.1 Dataset statistics"

    assert Report._is_per_slice_ssim("ssim_cube_col_3")        is True
    assert Report._is_per_slice_ssim("ssim_cube_norm_row_0")   is True
    assert Report._is_per_slice_ssim("ssim_cube_col_mean")     is False
    assert Report._is_per_slice_ssim("ssim_cube_col_n_scored") is False

    groups = dict(Report.group_metric_keys(["ssim_cube_z_mean", "ssim_cube_z_0", "ssim_cube_norm_z_1"]))
    assert groups == {"3.4c SSIM summaries (prediction vs raw cube)": ["ssim_cube_z_mean"]}


def test_report_metric_tables_leave_out_the_target_and_training_dataset_provenance():
    """The target and training dataset are shown in the configuration sections, so the metric tables do not repeat them."""
    groups = dict(Report.group_metric_keys(["curve_mse_gt", "target_dataset", "training_dataset"]))

    assert groups == {"3.2 Curve-level (GT)": ["curve_mse_gt"]}


def _metric_group(key: str) -> str:
    """Returns the taxonomy section title that claims the given metric key."""
    return next(title for title, match in Report._METRIC_TAXONOMY if match(key))


def test_report_metric_taxonomy_routes_keys():
    """Each metric family is routed to its taxonomy section."""
    assert _metric_group("gt_mean")                   == "3.1 Dataset statistics"
    assert _metric_group("curve_mse_gt")              == "3.2 Curve-level (GT)"
    assert _metric_group("pixel_mse_gt_mean")         == "3.3 Per-pixel (GT)"
    assert _metric_group("ssim_gt_z_mean")         == "3.4 SSIM summaries (denorm)"
    assert _metric_group("ssim_norm_z_mean")       == "3.4b SSIM summaries (normalised)"
    assert _metric_group("active_frac_gt")            == "3.9 Slot occupancy & active count"
    assert _metric_group("count_exact_frac")          == "3.9 Slot occupancy & active count"
    assert _metric_group("slot_0_active_pred_frac")   == "3.9 Slot occupancy & active count"
    assert _metric_group("slot_0_mu_active_pred_mean") == "3.9b Per-slot parameter statistics by activity"
    assert _metric_group("slot_0_amp_inactive_gt_std") == "3.9b Per-slot parameter statistics by activity"
    assert _metric_group("slot_0_sig_global_gt_mean")  == "3.9b Per-slot parameter statistics by activity"
    assert _metric_group("slot_usage_entropy")        == "3.9c Slot internal organization (emergent)"
    assert _metric_group("slot_mu_rank_diag")         == "3.9c Slot internal organization (emergent)"
    assert _metric_group("slot_gt_alignment")         == "3.9c Slot internal organization (emergent)"
    assert _metric_group("matched_mu_mae")            == "3.10 Matched Gaussian errors (permutation-invariant)"
    assert _metric_group("jepa_embedding_mse")        == "3.12 JEPA embedding diagnostics"


def test_report_metric_sections_render_in_display_order():
    """Taxonomy titles sort into the documented display order."""
    titles = sorted((title for title, _match in Report._METRIC_TAXONOMY), key=Report._section_order)

    assert titles[0]  == "3.1 Dataset statistics"
    assert titles[4]  == "3.4b SSIM summaries (normalised)"
    assert titles[-1] == "3.18 Presence calibration"


def test_report_is_per_slice_ssim_classifier():
    """Per-slice SSIM and z keys are separated from their summary counterparts."""
    assert Report._is_per_slice_ssim("ssim_gt_z_3")    is True
    assert Report._is_per_slice_ssim("z_mae_gt_7")     is True
    assert Report._is_per_slice_ssim("ssim_gt_z_mean") is False
    assert Report._is_per_slice_ssim("curve_mse_gt")      is False




def test_report_figure_sections_absent_without_figures(tmp_path):
    """Figure sections are omitted when no figure paths are supplied."""
    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 0, "best_epoch": 0, "best_val_loss": 1.0},
        global_metrics   = _global_metrics(),
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )

    text = report.assemble().read_text(encoding="utf-8")

    assert "## 4. Profile reconstructions"     not in text
    assert "## 5. Per-pixel metric maps"       not in text
    assert "## 6. Gaussian parameter analysis" not in text
    assert "## 8. SSIM curves"                 not in text


def test_report_full_metrics_excludes_per_slice(tmp_path):
    """Per-slice SSIM entries are excluded from the full-metric tables."""
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 0, "best_epoch": 0, "best_val_loss": 1.0},
        global_metrics   = gm,
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )

    lines  = report._build_full_metrics()
    joined = "\n".join(lines)

    assert "ssim_gt_z_0`" not in joined
    assert "3.1 Dataset statistics" in joined


def _extraction_metrics():
    """Returns global metrics extended with curve-extraction fit and measurement-floor values."""
    gm = _global_metrics()

    gm["params_source"]                  = "extracted"
    gm["extract_fit_r2_mean"]            = 0.98
    gm["extract_fit_r2_median"]          = 0.99
    gm["extract_fit_r2_p5"]              = 0.91
    gm["extract_floor_matched_mu_mae"]   = 0.12
    gm["extract_floor_matched_sig_mae"]  = 0.08
    gm["extract_floor_matched_amp_mae"]  = 0.03
    gm["extract_floor_matched_recall"]   = 0.97

    return gm


def _jepa_metrics():
    """Returns global metrics extended with the JEPA embedding, decode and chain diagnostics."""
    gm = _global_metrics()

    gm.update({
        "jepa_embedding_mse"          : 0.02,
        "jepa_embedding_rmse"         : 0.1414,
        "jepa_embedding_nmse"         : 0.05,
        "jepa_embedding_cosine"       : 0.97,
        "jepa_embedding_dim_rmse_min" : 0.05,
        "jepa_embedding_dim_rmse_max" : 0.30,
        "jepa_embedding_dim_rmse_std" : 0.07,
        "jepa_decode_mse_norm"        : 0.010,
        "jepa_chain_mse_norm"         : 0.025,
        "jepa_predictor_excess_mse"   : 0.015,
        "jepa_decoder_floor_frac"     : 0.4,
        "jepa_decode_amplification"   : 0.75,
        "jepa_embedding_err_p95"      : 0.06,
        "jepa_embedding_cosine_p5"    : 0.90,
        "jepa_embedding_error_corr"   : 0.42,
    })

    return gm


def _report_with(gm, tmp_path, figure_paths=None):
    """Returns a report bound to the given metrics, figure paths and output directory."""
    return Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_Z)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 3, "best_epoch": 2, "best_val_loss": 0.1},
        global_metrics   = gm,
        figure_paths     = figure_paths or {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )


def test_report_includes_the_extraction_floor_section(tmp_path):
    """Extraction metrics add the curve-parameter extraction and measurement-floor section."""
    text = _report_with(_extraction_metrics(), tmp_path).assemble().read_text(encoding="utf-8")

    assert "Curve-parameter extraction and its measurement floor" in text
    assert "measurement floor" in text
    assert "Floor μ MAE" in text


def test_cross_references_follow_the_extraction_section(tmp_path):
    """Inserting the extraction section shifts the slot-occupancy anchor and the cross-references that point at it."""
    plain = _report_with(_global_metrics(), tmp_path)
    plain.assemble()

    shifted = _report_with(_extraction_metrics(), tmp_path)
    text    = shifted.assemble().read_text(encoding="utf-8")

    assert plain._anchors["slot_occupancy"]   == "2.4"
    assert shifted._anchors["extraction"]     == "2.4"
    assert shifted._anchors["slot_occupancy"] == "2.5"

    assert "§2.5 gives the corresponding emission fractions" in text
    assert "§2.4 gives the corresponding emission fractions" not in text


def test_report_omits_the_extraction_section_for_parameter_models(tmp_path):
    """Runs without extraction metrics omit the extraction section."""
    text = _report_with(_global_metrics(), tmp_path).assemble().read_text(encoding="utf-8")

    assert "Curve-parameter extraction" not in text


def test_report_renders_the_deepened_jepa_headline(tmp_path):
    """JEPA metrics render the predictor-excess, decoder-floor, amplification and error-correlation rows."""
    text = _report_with(_jepa_metrics(), tmp_path).assemble().read_text(encoding="utf-8")

    assert "Predictor excess MSE"   in text
    assert "Decoder floor fraction" in text
    assert "Decode amplification"   in text
    assert "Embedding-error corr"   in text


def test_report_renders_the_roughness_section(tmp_path):
    """The curve-smoothness section and its roughness ratio are always rendered."""
    text = _report_with(_global_metrics(), tmp_path).assemble().read_text(encoding="utf-8")

    assert "Curve smoothness" in text
    assert "Roughness ratio"  in text


def test_report_lists_the_embedding_figure_section(tmp_path):
    """Embedding figure paths add the embedding-space diagnostics figure section."""
    figures = {"embedding_error_map": [tmp_path / "err.png"], "embedding_chain_map": [tmp_path / "chain.png"]}

    for path in (tmp_path / "err.png", tmp_path / "chain.png"):
        path.write_bytes(b"")

    text = _report_with(_jepa_metrics(), tmp_path, figures).assemble().read_text(encoding="utf-8")

    assert "6c. Embedding-space diagnostics" in text
    assert "6c.1 Per-pixel embedding error"  in text


def test_report_taxonomy_routes_the_new_metric_families():
    """Extraction and roughness keys route to their own taxonomy sections."""
    assert _metric_group("extract_fit_r2_mean")          == "3.12b Curve-parameter extraction"
    assert _metric_group("extract_floor_matched_mu_mae") == "3.12b Curve-parameter extraction"
    assert _metric_group("params_source")                == "3.12b Curve-parameter extraction"
    assert _metric_group("roughness_ratio")              == "3.12c Curve roughness"
