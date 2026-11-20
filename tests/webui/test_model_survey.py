
from __future__ import annotations

from unittest.mock import Mock
import base64

import pytest

from model_survey import ModelSurvey

from tests.conftest           import SilentLogger
from tests.webui._probe_fakes import N_ROWS, N_COLS, PH, PW, fake_run


def _survey() -> ModelSurvey:
    survey = ModelSurvey(SilentLogger())

    survey.loaded = {
        "run"      : fake_run(),
        "path"     : "fake",
        "split"    : "test",
        "device"   : "cpu",
        "labels"   : ["a", "b"],
        "targets"  : ["first", "second"],
        "patch"    : (PH, PW),
        "backbone" : "bare",
    }
    survey.status = {"state": "running", "path": "fake", "progress": 0.0, "stage": "", "error": ""}

    return survey


def test_survey_covers_the_region_and_finishes():
    survey = _survey()
    survey._survey()

    assert survey.status["state"]    == "done"
    assert survey.status["progress"] == pytest.approx(1.0)

    result = survey.result
    assert result["region"]   == [N_ROWS, N_COLS]
    assert result["coverage"] == {"tiles": 4, "total_tiles": 4, "pixels": 256}
    assert result["channels"] == ["a", "b"]
    assert result["targets"]  == ["first", "second"]
    assert result["seconds"]  > 0.0


def test_survey_fit_scores_every_target_channel():
    survey = _survey()
    survey._survey()

    fit = {entry["name"]: entry for entry in survey.result["fit"]}

    assert set(fit) == {"first", "second"}
    assert fit["first"]["rmse"] == pytest.approx(0.25, abs=1e-5)
    assert fit["first"]["mae"]  == pytest.approx(0.25, abs=1e-5)
    assert fit["second"]["r2"]  < 1.0


def test_survey_ranks_ablation_and_attribution_on_the_used_channel():
    survey = _survey()
    survey._survey()

    result = survey.result

    channels = result["ablation"]["channels"]
    assert channels[0]["label"] == "a"
    assert channels[0]["delta"] > 1e-8
    assert channels[1]["delta"] == pytest.approx(0.0, abs=1e-12)

    for target in result["attribution"]["targets"]:
        assert target["dead"] is False
        assert target["shares"][0] == pytest.approx(1.0)


def test_survey_flips_noise_occlusion_and_erf_profiles():
    survey = _survey()
    survey._survey()

    result = survey.result

    flips = {entry["flip"]: entry["delta"] for entry in result["flips"]}
    assert flips["none"] == pytest.approx(0.0, abs=1e-12)
    assert flips["row"]  == pytest.approx(0.0, abs=1e-10)
    assert flips["col"]  == pytest.approx(0.0, abs=1e-10)

    noise = result["noise"]
    assert noise[0]["value"] == pytest.approx(0.0)
    assert noise[0]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert noise[-1]["delta"] > noise[1]["delta"]

    occlusion = result["occlusion"]
    assert occlusion["occluder"] == [2, 2]
    assert occlusion["delta"][0] > 0.0
    assert occlusion["delta"][-1] == pytest.approx(0.0, abs=1e-12)

    erf = {entry["name"]: entry for entry in result["erf"]["targets"]}
    assert erf["first"]["dead"]    is False
    assert erf["first"]["r50"]     == 0
    assert erf["first"]["samples"] == ModelSurvey.ERF_SAMPLES

    vitals = result["vitals"]
    assert vitals["summary"]["n_layers"] == 1
    assert vitals["entries"][0]["type"]  == "Identity"


def test_survey_mean_gradient_maps_follow_the_used_channel():
    survey = _survey()
    survey._survey()

    gradients = survey.result["gradients"]
    assert gradients["samples"] == ModelSurvey.ERF_SAMPLES
    assert gradients["patch"]   == [PH, PW]
    assert gradients["center"]  == [PH // 2, PW // 2]

    first = gradients["targets"][0]

    assert first["dead"] is False
    assert first["shares"][0] == pytest.approx(1.0)
    assert base64.b64decode(first["cells"][0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert first["cells"][1] is None


def test_survey_tile_cap_strides_evenly_and_reports_totals():
    survey          = _survey()
    survey.tile_cap = 2
    survey._survey()

    coverage = survey.result["coverage"]
    assert coverage["tiles"]       == 2
    assert coverage["total_tiles"] == 4
    assert coverage["pixels"]      == 128


def test_survey_cancel_stops_the_run():
    survey = _survey()
    survey.cancel_flag = True
    survey._survey()

    assert survey.status["state"] == "cancelled"
    assert survey.result is None


def test_worker_caps_torch_threads_and_restores_them(monkeypatch):
    import torch

    survey = ModelSurvey(SilentLogger())
    seen   = {}

    def observing_load(run_path, split, device):
        seen["threads"] = torch.get_num_threads()
        Mock(side_effect=RuntimeError("stop"))()

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
    survey = ModelSurvey(SilentLogger())

    out = survey.start({"path": str(tmp_path / "nowhere")})
    assert out["ok"] is False
    assert "not a directory" in out["error"]


def _loadable_run(tmp_path, survey):
    run_dir = tmp_path / "run_ok"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"x")
    (run_dir / "meta").mkdir()
    (run_dir / "meta" / "model_config.json").write_text("{}")

    survey.scanner.roots.add(tmp_path.resolve())
    return str(run_dir)


def test_survey_start_refuses_bad_configs(tmp_path):
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
    survey             = _survey()
    survey.erf_samples = 0
    survey._survey()

    assert survey.status["progress"] == pytest.approx(1.0)

    erf = survey.result["erf"]
    assert erf["samples"] == 0
    assert all(entry["dead"] for entry in erf["targets"])

    gradients = survey.result["gradients"]
    assert gradients["samples"] == 0
    assert all(entry["dead"] for entry in gradients["targets"])


def test_survey_result_before_any_run_fails():
    assert ModelSurvey(SilentLogger()).survey_result()["ok"] is False
    assert ModelSurvey(SilentLogger()).cancel()["ok"] is False
