"""Tests for ModelSurvey, the region-wide sweep behind the webui autopsy tab.

Covers tile coverage and progress accounting, the aggregated fit, detection and
matched-parameter statistics, the ablation, attribution, flip, noise, occlusion,
effective-receptive-field and mean-gradient sections, the request validation on start,
and cancellation and thread-count handling in the worker.
"""

from __future__ import annotations

import base64

import pytest

from model_survey import ModelSurvey

from pipelines.backbone.inference.probes import PredictionCurves

from tests.conftest           import SilentLogger
from tests.webui._probe_fakes import N_ROWS, N_COLS, PH, PW, fake_run


def _survey(n_slots: int = 1, with_gt: bool = True) -> ModelSurvey:
    """Returns a ModelSurvey holding a fake loaded run.

    Args:
        n_slots : Number of Gaussian component slots the fake model predicts.
        with_gt : Whether the fake run carries ground-truth parameters.

    Returns:
        A survey whose loaded state and status are set as if a run had been loaded.
    """
    survey = ModelSurvey(SilentLogger())
    run    = fake_run(n_slots, with_gt)

    survey.loaded = {
        "run"      : run,
        "path"     : "fake",
        "split"    : "test",
        "device"   : "cpu",
        "labels"   : ["primary", "sec CH01"],
        "renderer" : PredictionCurves(n_slots, run.z_axis),
        "patch"    : (PH, PW),
        "backbone" : "bare",
    }
    survey.status = {"state": "running", "path": "fake", "progress": 0.0, "stage": "", "error": ""}

    return survey


def test_survey_covers_the_region_and_finishes():
    """A full survey reaches the done state and reports the region, tile and pixel coverage, channel labels and elapsed time."""
    survey = _survey()
    survey._survey()

    assert survey.status["state"]    == "done"
    assert survey.status["progress"] == pytest.approx(1.0)

    result = survey.result
    assert result["region"]   == [N_ROWS, N_COLS]
    assert result["coverage"] == {"tiles": 4, "total_tiles": 4, "pixels": 256}
    assert result["channels"] == ["primary", "sec CH01"]
    assert result["seconds"]  > 0.0


def test_survey_fit_detection_and_matches():
    """The fit, detection and matched sections report curve errors, active-slot counts and per-parameter errors over every surveyed pixel."""
    survey = _survey()
    survey._survey()

    result = survey.result

    fit = result["fit"]
    assert fit["curve_mse_gt"]["mean"] > 0.0
    assert fit["curve_mse_gt"]["p90"] >= fit["curve_mse_gt"]["median"]
    assert fit["curve_mse_raw"]["mean"] > 0.0
    assert fit["base_power"] > 0.0

    detection = result["detection"]
    assert detection["mean_pred_active"] == pytest.approx(1.0)
    assert detection["mean_gt_active"]   == pytest.approx(1.0)
    assert detection["count_match_frac"] == pytest.approx(1.0)

    matched = result["matched"]
    assert matched["n_matched"] == 256
    assert matched["mu"] > 1.0
    assert matched["sigma"] == pytest.approx(1.0, abs=1e-5)


def test_survey_ranks_ablation_and_attribution_on_the_used_channel():
    """Ablation and attribution both concentrate on the channel the model reads and mark the unused parameter family dead."""
    survey = _survey()
    survey._survey()

    result = survey.result

    channels = result["ablation"]["channels"]
    assert channels[0]["label"]      == "primary"
    assert channels[0]["delta_mse"]  > 1e-8
    assert channels[1]["delta_mse"]  == pytest.approx(0.0, abs=1e-12)
    assert channels[1]["flips_frac"] == pytest.approx(0.0)

    mu = {f["family"]: f for f in result["attribution"]["families"]}["mu"]
    assert mu["dead"] is False
    assert mu["shares"][0] == pytest.approx(1.0)

    sigma = {f["family"]: f for f in result["attribution"]["families"]}["sigma"]
    assert sigma["dead"] is True


def test_survey_flips_noise_occlusion_and_erf_profiles():
    """The flip, noise, occlusion, receptive-field and vitals sections report the behaviour expected of a pointwise model."""
    survey = _survey()
    survey._survey()

    result = survey.result

    flips = {entry["flip"]: entry["delta_mse"] for entry in result["flips"]}
    assert flips["none"] == pytest.approx(0.0, abs=1e-12)
    assert flips["row"] == pytest.approx(0.0, abs=1e-10)
    assert flips["col"] == pytest.approx(0.0, abs=1e-10)

    noise = result["noise"]
    assert noise[0]["value"]     == pytest.approx(0.0)
    assert noise[0]["delta_mse"] == pytest.approx(0.0, abs=1e-12)
    assert noise[-1]["delta_mse"] > noise[1]["delta_mse"]

    occlusion = result["occlusion"]
    assert occlusion["occluder"] == [2, 2]
    assert occlusion["delta"][0] > 0.0
    assert occlusion["delta"][-1] == pytest.approx(0.0, abs=1e-12)

    erf = {f["family"]: f for f in result["erf"]["families"]}
    assert erf["mu"]["dead"] is False
    assert erf["mu"]["r50"] == 0
    assert erf["mu"]["samples"] == ModelSurvey.ERF_SAMPLES
    assert erf["sigma"]["dead"] is True

    vitals = result["vitals"]
    assert vitals["summary"]["n_layers"] == 1
    assert vitals["entries"][0]["type"]  == "Identity"


def test_survey_mean_gradient_maps_follow_the_used_channel():
    """The mean gradient maps carry the sample count and patch geometry, render a PNG only for the read channel and mark the unused family dead."""
    survey = _survey()
    survey._survey()

    gradients = survey.result["gradients"]
    assert gradients["samples"] == ModelSurvey.ERF_SAMPLES
    assert gradients["patch"]   == [PH, PW]
    assert gradients["center"]  == [PH // 2, PW // 2]

    families = {f["family"]: f for f in gradients["families"]}

    assert families["mu"]["dead"] is False
    assert families["mu"]["shares"][0] == pytest.approx(1.0)
    assert base64.b64decode(families["mu"]["cells"][0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert families["mu"]["cells"][1] is None

    assert families["sigma"]["dead"]  is True
    assert families["sigma"]["cells"] == [None, None]


def test_survey_without_gt_omits_fit_gt_sections():
    """A run without ground truth omits the ground-truth fit, detection and matched sections."""
    survey = _survey(with_gt=False)
    survey._survey()

    result = survey.result
    assert result["fit"]["curve_mse_gt"] is None
    assert result["detection"] is None
    assert result["matched"]   is None


def test_survey_tile_cap_strides_evenly_and_reports_totals():
    """A tile cap surveys a strided subset while still reporting the total tile count."""
    survey          = _survey()
    survey.tile_cap = 2
    survey._survey()

    coverage = survey.result["coverage"]
    assert coverage["tiles"]       == 2
    assert coverage["total_tiles"] == 4
    assert coverage["pixels"]      == 128


def test_survey_cancel_stops_the_run():
    """A cancel request ends the survey in the cancelled state with no result."""
    survey = _survey()
    survey.cancel_flag = True
    survey._survey()

    assert survey.status["state"] == "cancelled"
    assert survey.result is None


def test_worker_caps_torch_threads_and_restores_them(monkeypatch):
    """The worker caps the torch thread count for the duration of the run and restores the previous value on failure."""
    import torch

    survey = ModelSurvey(SilentLogger())
    seen   = {}

    def observing_load(run_path, split, device):
        """Records the active torch thread count and aborts the worker."""
        seen["threads"] = torch.get_num_threads()
        raise RuntimeError("stop")

    survey.cpu_threads = 2
    monkeypatch.setattr(survey, "_load", observing_load)

    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        survey._worker("x", "test", "cpu")

        assert seen["threads"] == 2
        assert torch.get_num_threads() == 4
        assert survey.status["state"] == "error"
    finally:
        torch.set_num_threads(previous)


def test_survey_start_refuses_non_runs(tmp_path):
    """Starting a survey on a missing directory is refused."""
    survey = ModelSurvey(SilentLogger())

    out = survey.start({"path": str(tmp_path / "nowhere")})
    assert out["ok"] is False
    assert "not a directory" in out["error"]


def _loadable_run(tmp_path, survey):
    """Creates a catalogued run directory holding a checkpoint and a persisted model config."""
    run_dir = tmp_path / "run_ok"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"x")
    (run_dir / "meta").mkdir()
    (run_dir / "meta" / "model_config.json").write_text("{}")

    survey.scanner.roots.add(tmp_path.resolve())
    return str(run_dir)


def test_survey_start_refuses_bad_configs(tmp_path):
    """Unknown splits and devices, negative tile and sample counts, and a thread count below one are all refused."""
    survey = ModelSurvey(SilentLogger())
    path   = _loadable_run(tmp_path, survey)

    bad_split = survey.start({"path": path, "split": "holdout"})
    assert bad_split["ok"] is False
    assert "split 'holdout'" in bad_split["error"]

    bad_device = survey.start({"path": path, "device": "tpu"})
    assert bad_device["ok"] is False
    assert "device 'tpu'" in bad_device["error"]

    bad_tiles = survey.start({"path": path, "tiles": -1})
    assert bad_tiles["ok"] is False
    assert "negative" in bad_tiles["error"]

    bad_erf = survey.start({"path": path, "erf_samples": -3})
    assert bad_erf["ok"] is False
    assert "negative" in bad_erf["error"]

    bad_threads = survey.start({"path": path, "threads": 0})
    assert bad_threads["ok"] is False
    assert "at least 1" in bad_threads["error"]


def test_survey_with_zero_erf_samples_skips_the_phase():
    """Zero receptive-field samples skips that phase, still completing with every family marked dead."""
    survey             = _survey()
    survey.erf_samples = 0
    survey._survey()

    assert survey.status["progress"] == pytest.approx(1.0)

    erf = survey.result["erf"]
    assert erf["samples"] == 0
    assert all(f["dead"] for f in erf["families"])

    gradients = survey.result["gradients"]
    assert gradients["samples"] == 0
    assert all(f["dead"] for f in gradients["families"])


def test_survey_result_before_any_run_fails():
    """Fetching a result or cancelling before any survey has started fails."""
    assert ModelSurvey(SilentLogger()).survey_result()["ok"] is False
    assert ModelSurvey(SilentLogger()).cancel()["ok"] is False
