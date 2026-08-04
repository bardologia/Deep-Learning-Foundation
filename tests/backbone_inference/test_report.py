from __future__ import annotations

import types

import numpy as np
import pytest

from tools.data.regions                   import CropRegion
from pipelines.backbone.inference.metrics import Metrics, Result
from pipelines.backbone.inference.report  import Report, ReportPayloadBuilder


N_GAUSSIANS = 2
N_ELEV      = 12
H = W       = 6


def _curves_from_params(params, x_axis, n_gaussians):
    curve = np.zeros((x_axis.size, params.shape[1], params.shape[2]), dtype=np.float32)
    for k in range(n_gaussians):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = x_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return curve.astype(np.float32)


def _global_metrics():
    rng    = np.random.default_rng(0)
    x_axis = np.linspace(-20.0, 80.0, N_ELEV).astype(np.float32)
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    for k in range(N_GAUSSIANS):
        params[3 * k]     = rng.random((H, W)).astype(np.float32) + 0.5
        params[3 * k + 1] = float(k * 15)
        params[3 * k + 2] = 4.0

    gt   = _curves_from_params(params, x_axis, N_GAUSSIANS)
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
        azimuth_offset     = 0,
        range_offset       = 0,
        params_pred        = params.copy(),
        params_gt          = params.copy(),
    )

    m = Metrics(res, x_axis, N_GAUSSIANS)
    return m.compute(
        elev_indices  = np.arange(N_ELEV),
        range_indices = np.arange(W),
        az_indices    = np.arange(H),
        param_space   = True,
    )


def _run_stub():
    region = CropRegion(azimuth_start=0, azimuth_end=H, range_start=0, range_end=W)
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
        x_axis_length    = N_ELEV,
        split_name       = "test",
        split_region     = region,
        global_crop      = region,
        grid             = grid,
        dataset_config   = ds_cfg,
        secondary_labels = ("PS04", "PS06"),
    )


def _cfg_stub():
    return types.SimpleNamespace(
        stitch_window=True, cube_dtype="float32", save_cubes=False, save_plots=True,
        save_animations=False, n_best_profiles=4, n_worst_profiles=4, n_random_profiles=4,
        n_range_slices=3, n_azimuth_slices=3, n_elevation_slices=3, gif_axes=["range"],
        gif_fps=12, gif_max_frames=10, device="cpu", num_workers=2,
    )


def test_report_payload_run_summary_structure():
    run     = _run_stub()
    x_axis  = np.linspace(-20.0, 80.0, N_ELEV)
    payload = ReportPayloadBuilder.run_summary(run, x_axis)

    assert payload["model_name"]       == "unet"
    assert payload["n_gaussians"]      == N_GAUSSIANS
    assert payload["x_axis_length"]    == N_ELEV
    assert payload["x_axis_min"]       == pytest.approx(-20.0)
    assert payload["secondary_labels"] == "PS04, PS06"
    assert isinstance(payload["input_config"], dict)


def test_report_payload_inference_config_structure():
    payload = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub())

    assert payload["device"]      == "cpu"
    assert payload["batch_size"]  == 8
    assert payload["gif_axes"]    == ["range"]
    assert "n_best_profiles" in payload


def test_report_assemble_writes_markdown(tmp_path):
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_ELEV)),
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
    assert "# TomoSAR Inference Report" in text
    assert "## 1. Run summary" in text
    assert "## 2. Headline metrics" in text
    assert "## 3. Full metric tables" in text
    assert "`model_name`" in text
    assert "unet" in text


def test_report_includes_slot_occupancy_section(tmp_path):
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_ELEV)),
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
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_ELEV)),
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


def _metric_group(key: str) -> str:
    return next(title for title, match in Report._METRIC_TAXONOMY if match(key))


def test_report_metric_taxonomy_routes_keys():
    assert _metric_group("gt_mean")                   == "3.1 Dataset statistics"
    assert _metric_group("curve_mse_gt")              == "3.2 Curve-level (GT)"
    assert _metric_group("pixel_mse_gt_mean")         == "3.3 Per-pixel (GT)"
    assert _metric_group("ssim_gt_elev_mean")         == "3.4 SSIM summaries (denorm)"
    assert _metric_group("ssim_norm_elev_mean")       == "3.4b SSIM summaries (normalised)"
    assert _metric_group("curve_mse_red")             == "3.5 Curve-level (reduced vs GT)"
    assert _metric_group("ssim_red_elev_mean")        == "3.7 SSIM summaries (reduced vs GT)"
    assert _metric_group("relative_mse_reduction")    == "3.8 NN improvement over baseline"
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
    assert _metric_group("physics_valid_fraction")    == "3.11 Interferometric data consistency"
    assert _metric_group("phase_agreement_gt_mean")   == "3.11 Interferometric data consistency"
    assert _metric_group("jepa_embedding_mse")        == "3.12 JEPA embedding diagnostics"


def test_report_metric_sections_render_in_display_order():
    titles = sorted((title for title, _match in Report._METRIC_TAXONOMY), key=Report._section_order)

    assert titles[0]  == "3.1 Dataset statistics"
    assert titles[4]  == "3.4b SSIM summaries (normalised)"
    assert titles[-1] == "3.18 Presence calibration"


def test_report_is_per_slice_ssim_classifier():
    assert Report._is_per_slice_ssim("ssim_gt_elev_3")    is True
    assert Report._is_per_slice_ssim("elev_mae_gt_7")     is True
    assert Report._is_per_slice_ssim("ssim_gt_elev_mean") is False
    assert Report._is_per_slice_ssim("curve_mse_gt")      is False


def test_report_is_reduced_key():
    assert Report._is_reduced_key("curve_mse_red")  is True
    assert Report._is_reduced_key("ssim_red_elev_3") is True
    assert Report._is_reduced_key("pixel_mse_predn") is True
    assert Report._is_reduced_key("reduced_orientation_corr_aligned") is True
    assert Report._is_reduced_key("reduced_orientation_corr_flipped") is True
    assert Report._is_reduced_key("curve_mse_gt")    is False


def test_report_figure_sections_absent_without_figures(tmp_path):
    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_ELEV)),
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
    gm = _global_metrics()

    report = Report(
        output_dir       = tmp_path,
        run_summary      = ReportPayloadBuilder.run_summary(_run_stub(), np.linspace(-20.0, 80.0, N_ELEV)),
        inference_config = ReportPayloadBuilder.inference_config(_cfg_stub(), _run_stub()),
        checkpoint_meta  = {"epoch": 0, "best_epoch": 0, "best_val_loss": 1.0},
        global_metrics   = gm,
        figure_paths     = {},
        gif_paths        = {},
        report_path      = tmp_path / "report.md",
    )

    lines  = report._build_full_metrics()
    joined = "\n".join(lines)

    assert "ssim_gt_elev_0`" not in joined
    assert "3.1 Dataset statistics" in joined
