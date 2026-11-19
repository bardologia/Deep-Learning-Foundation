
from __future__ import annotations

import base64

import numpy as np
import pytest
import torch

from model_probe import ModelProbe

from pipelines.backbone.inference.probes import PredictionCurves

from tests.conftest           import SilentLogger
from tests.webui._probe_fakes import PH, PW, N_Z, AzShiftModel, TinyConvModel, Wrapper, fake_run


def _probe(n_slots: int = 1, with_gt: bool = True) -> ModelProbe:
    probe = ModelProbe(SilentLogger())
    run   = fake_run(n_slots, with_gt)

    probe.loaded = {
        "run"      : run,
        "labels"   : ["primary", "sec CH01"],
        "layers"   : ["conv"],
        "types"    : {"conv": "Conv2d"},
        "renderer" : PredictionCurves(n_slots, run.z_axis),
        "patch"    : (PH, PW),
        "device"   : "cpu",
    }
    probe.status = {"state": "ready", "path": "fake", "progress": 1.0, "stage": "ready", "error": "", "info": None}

    return probe


def test_predict_returns_slots_curves_and_references():
    result = _probe().predict({"row": 10, "col": 8})

    assert result["ok"] is True
    assert len(result["slots"])     == 1
    assert result["slots"][0]["active"] is True
    assert len(result["curve"])     == N_Z
    assert len(result["gt_curve"])  == N_Z
    assert result["gt_slots"][0]["mu"] == pytest.approx(20.0)
    assert len(result["raw_curve"]) == N_Z


def test_predict_reports_fit_and_hungarian_matches():
    result = _probe().predict({"row": 10, "col": 8})

    fit = result["fit"]
    assert fit["curve_mse_gt"]  > 0.0
    assert fit["curve_mse_raw"] > 0.0
    assert fit["pred_active"] == 1
    assert fit["gt_active"]   == 1

    match = result["matches"]
    assert len(match) == 1
    assert match[0]["gt_slot"]   == 0
    assert match[0]["pred_slot"] == 0
    assert match[0]["d_amp"]   == pytest.approx(result["slots"][0]["amp"] - 1.5, abs=1e-5)
    assert match[0]["d_mu"]    == pytest.approx(result["slots"][0]["mu"] - 20.0, abs=1e-4)
    assert match[0]["d_sigma"] == pytest.approx(-1.0, abs=1e-5)


def test_predict_matches_cover_every_active_gt_slot():
    result = _probe(n_slots=2).predict({"row": 10, "col": 8})

    match = result["matches"]
    assert [m["gt_slot"] for m in match]        == [0, 1]
    assert sorted(m["pred_slot"] for m in match) == [0, 1]


def test_predict_without_gt_omits_matches_and_gt_fit():
    result = _probe(with_gt=False).predict({"row": 10, "col": 8})

    assert result["ok"] is True
    assert result["gt_curve"] is None
    assert result["gt_slots"] is None
    assert result["matches"]  is None
    assert result["fit"]["curve_mse_gt"] is None
    assert result["fit"]["gt_active"]    is None


def test_predict_outside_region_fails():
    result = _probe().predict({"row": 99, "col": 0})

    assert result["ok"] is False
    assert "outside" in result["error"]


def test_predict_without_loaded_model_fails():
    probe = ModelProbe(SilentLogger())

    assert probe.predict({"row": 0, "col": 0})["ok"]     is False
    assert probe.attribution({"row": 0, "col": 0})["ok"] is False


def test_fields_returns_per_slot_maps_and_overviews():
    result = _probe().fields({"row": 10, "col": 8})

    assert result["ok"]     is True
    assert result["center"] == [4, 4]
    assert result["patch"]  == [PH, PW]
    assert len(result["slots"]) == 1

    slot = result["slots"][0]
    assert slot["active"] is True
    assert [f["family"] for f in slot["families"]] == ["amp", "mu", "sigma"]

    amp = slot["families"][0]
    assert base64.b64decode(amp["png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert amp["min"] >= 1.5
    assert amp["max"] <= 2.5

    sigma = slot["families"][2]
    assert sigma["min"] == pytest.approx(3.0)
    assert sigma["max"] == pytest.approx(3.0)

    assert result["activity"]["min"] == pytest.approx(1.0)
    assert result["activity"]["max"] == pytest.approx(1.0)
    assert result["error"]["max"] > 0.0


def test_fields_without_gt_has_no_error_map():
    result = _probe(with_gt=False).fields({"row": 10, "col": 8})

    assert result["ok"]    is True
    assert result["error"] is None


def test_fields_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).fields({"row": 0, "col": 0})["ok"] is False


def test_attribution_concentrates_on_the_used_channel():
    result = _probe().attribution({"row": 10, "col": 8})

    assert result["ok"]       is True
    assert result["channels"] == ["primary", "sec CH01"]
    assert result["center"]   == [4, 4]
    assert result["patch"]    == [PH, PW]
    assert result["slot"]     == -1

    families = {payload["family"]: payload for payload in result["families"]}

    assert families["mu"]["dead"] is False
    assert families["mu"]["shares"][0] == pytest.approx(1.0)
    assert families["mu"]["shares"][1] == pytest.approx(0.0)
    assert base64.b64decode(families["mu"]["cells"][0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert families["mu"]["cells"][1] is None


def test_attribution_per_slot_isolates_each_components_inputs():
    probe = _probe(n_slots=2)

    slot0    = probe.attribution({"row": 10, "col": 8, "slot": 0})
    slot1    = probe.attribution({"row": 10, "col": 8, "slot": 1})
    combined = probe.attribution({"row": 10, "col": 8})

    assert slot0["ok"] and slot1["ok"] and combined["ok"]
    assert (slot0["slot"], slot1["slot"], combined["slot"]) == (0, 1, -1)

    mu0  = {payload["family"]: payload for payload in slot0["families"]}["mu"]
    mu1  = {payload["family"]: payload for payload in slot1["families"]}["mu"]
    both = {payload["family"]: payload for payload in combined["families"]}["mu"]

    assert mu0["shares"][0]  == pytest.approx(1.0)
    assert mu0["shares"][1]  == pytest.approx(0.0)
    assert mu1["shares"][0]  == pytest.approx(0.0)
    assert mu1["shares"][1]  == pytest.approx(1.0)
    assert both["shares"][0] == pytest.approx(0.5)
    assert both["shares"][1] == pytest.approx(0.5)


def test_attribution_rejects_out_of_range_slots():
    probe = _probe()

    high = probe.attribution({"row": 10, "col": 8, "slot": 1})
    low  = probe.attribution({"row": 10, "col": 8, "slot": -2})

    assert high["ok"] is False
    assert "out of range" in high["error"]
    assert low["ok"] is False
    assert "out of range" in low["error"]


def test_attribution_marks_dead_families_instead_of_failing():
    result = _probe().attribution({"row": 10, "col": 8})

    families = {payload["family"]: payload for payload in result["families"]}

    assert families["sigma"]["dead"]   is True
    assert families["sigma"]["shares"] == [0.0, 0.0]
    assert families["sigma"]["cells"]  == [None, None]
    assert families["sigma"]["radial"] is None


def test_attribution_radial_profile_collapses_to_the_center_for_a_pointwise_model():
    result = _probe().attribution({"row": 10, "col": 8})

    radial = {payload["family"]: payload for payload in result["families"]}["mu"]["radial"]

    assert radial["radii"][0]         == 0
    assert radial["r50"]              == 0
    assert radial["r90"]              == 0
    assert radial["cumulative"][0]    == pytest.approx(1.0)
    assert radial["cumulative"][-1]   == pytest.approx(1.0)
    assert len(radial["radii"])       == len(radial["cumulative"])


def test_ablation_ranks_the_used_channel_first():
    result = _probe().ablation({"row": 10, "col": 8})

    assert result["ok"] is True
    assert result["base_power"] > 0.0

    ranked = result["channels"]
    assert [c["label"] for c in ranked] == ["primary", "sec CH01"]
    assert ranked[0]["channel"]   == 0
    assert ranked[0]["delta_mse"] > 1e-6
    assert ranked[0]["max_mu_shift"] > 1.0
    assert ranked[1]["delta_mse"] == pytest.approx(0.0, abs=1e-12)
    assert ranked[1]["flips"]     == 0


def test_ablation_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).ablation({"row": 0, "col": 0})["ok"] is False


def test_occlusion_peaks_on_the_cell_holding_the_probed_pixel():
    result = _probe().occlusion({"row": 10, "col": 8})

    assert result["ok"]          is True
    assert result["patch"]       == [PH, PW]
    assert result["occluder"]    == [2, 2]
    assert result["grid"]        == [4, 4]
    assert result["center_cell"] == [2, 2]

    assert result["worst"]["row"] == 2
    assert result["worst"]["col"] == 2
    assert result["worst"]["delta_mse"] > 1e-6
    assert base64.b64decode(result["map"]["png"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_occlusion_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).occlusion({"row": 0, "col": 0})["ok"] is False


def test_flips_report_zero_deltas_for_a_pointwise_model():
    result = _probe().flips({"row": 10, "col": 8})

    assert result["ok"] is True
    assert [e["flip"] for e in result["entries"]] == ["none", "row", "col", "both"]
    assert result["base_power"] > 0.0

    for entry in result["entries"]:
        assert entry["delta_mse"] == pytest.approx(0.0, abs=1e-10)
        assert len(entry["curve"]) == N_Z
        assert len(entry["slots"]) == 1


def test_flips_expose_a_row_asymmetric_model():
    probe = _probe()
    probe.loaded["run"].model = Wrapper(AzShiftModel())

    result = probe.flips({"row": 10, "col": 8})

    deltas = {e["flip"]: e["delta_mse"] for e in result["entries"]}

    assert deltas["none"]    == pytest.approx(0.0, abs=1e-12)
    assert deltas["row"] > 1e-8
    assert deltas["col"]   == pytest.approx(0.0, abs=1e-10)
    assert deltas["both"]    > 1e-8


def test_flips_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).flips({"row": 0, "col": 0})["ok"] is False


def test_whatif_drop_of_used_channel_shifts_the_prediction():
    probe = _probe()

    dropped   = probe.whatif({"row": 10, "col": 8, "perturbation": {"kind": "drop_channel", "channel": 0}})
    untouched = probe.whatif({"row": 10, "col": 8, "perturbation": {"kind": "drop_channel", "channel": 1}})

    assert dropped["ok"] and untouched["ok"]
    assert dropped["delta_mse"]   > 1e-6
    assert untouched["delta_mse"] == pytest.approx(0.0, abs=1e-12)


def test_sweep_scale_channel_is_quiet_at_factor_one_and_loud_at_zero():
    result = _probe().sweep({"row": 10, "col": 8, "kind": "scale_channel", "channel": 0})

    assert result["ok"]      is True
    assert result["kind"]    == "scale_channel"
    assert result["channel"] == 0
    assert result["label"]   == "primary"

    points = result["points"]
    assert [p["value"] for p in points] == pytest.approx(list(np.linspace(0.0, 2.0, 9)))
    assert points[4]["value"]     == pytest.approx(1.0)
    assert points[4]["delta_mse"] == pytest.approx(0.0, abs=1e-12)
    assert points[0]["delta_mse"] > 1e-6
    assert points[8]["delta_mse"] > 1e-6


def test_sweep_noise_averages_seeds_and_grows_with_sigma():
    result = _probe().sweep({"row": 10, "col": 8, "kind": "noise"})

    assert result["ok"]      is True
    assert result["channel"] is None
    assert result["label"]   is None

    points = result["points"]
    assert points[0]["delta_mse"] == pytest.approx(0.0, abs=1e-12)
    assert points[0]["spread"]    == pytest.approx(0.0, abs=1e-12)
    assert points[8]["delta_mse"] > points[1]["delta_mse"]
    assert points[8]["spread"]    >= 0.0


def test_sweep_rejects_unsweepable_kinds_and_bad_channels():
    probe = _probe()

    drop = probe.sweep({"row": 10, "col": 8, "kind": "drop_channel"})
    assert drop["ok"] is False
    assert "no strength to sweep" in drop["error"]

    bad = probe.sweep({"row": 10, "col": 8, "kind": "scale_channel", "channel": 7})
    assert bad["ok"] is False
    assert "out of range" in bad["error"]


def test_whatif_unknown_perturbation_fails():
    result = _probe().whatif({"row": 10, "col": 8, "perturbation": {"kind": "warp"}})

    assert result["ok"] is False
    assert "unknown perturbation" in result["error"]


def test_vitals_reports_layers_in_forward_order_with_params_and_flags():
    probe = _probe()
    probe.loaded["run"].model = Wrapper(TinyConvModel())

    result = probe.vitals({"row": 10, "col": 8})

    assert result["ok"] is True
    assert [e["name"] for e in result["entries"]] == ["conv", "act", "head"]
    assert [e["type"] for e in result["entries"]] == ["Conv2d", "ReLU", "Conv2d"]
    assert [e["params"] for e in result["entries"]] == [152, 0, 27]
    assert result["entries"][0]["shape"] == [8, PH, PW]

    summary = result["summary"]
    assert summary["n_layers"]     == 3
    assert summary["total_params"] == 179
    assert 0 <= summary["flagged"] <= 3

    for entry in result["entries"]:
        assert 0.0 <= entry["zero_frac"] <= 1.0
        assert isinstance(entry["flags"], list)


def test_vitals_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).vitals({"row": 0, "col": 0})["ok"] is False


def _conv_probe() -> ModelProbe:
    probe = _probe()
    probe.loaded["run"].model = Wrapper(TinyConvModel())
    probe.loaded["layers"]    = ["conv", "act", "head"]
    probe.loaded["types"]     = {"conv": "Conv2d", "act": "ReLU", "head": "Conv2d"}
    return probe


def test_kernels_render_conv_grids_and_one_by_one_matrices():
    probe = _conv_probe()

    grid = probe.kernels_png("conv")
    assert grid[:8] == b"\x89PNG\r\n\x1a\n"

    matrix = probe.kernels_png("head")
    assert matrix[:8] == b"\x89PNG\r\n\x1a\n"


def test_kernels_refuse_layers_without_conv_weights():
    probe = _conv_probe()

    with pytest.raises(ValueError, match="no 4-D convolution weight"):
        probe.kernels_png("act")

    assert probe.kernels_png("missing") is None
    assert ModelProbe(SilentLogger()).kernels_png("conv") is None


class _WithContainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv       = torch.nn.Conv2d(1, 1, 1)
        self.downsample = torch.nn.ModuleList()


def test_probe_layers_skips_container_modules():
    layers = ModelProbe(SilentLogger())._probe_layers(_WithContainer())

    assert layers == ["conv"]


def test_probe_layers_skip_the_attention_dropout_the_fused_kernel_never_calls():
    from models.blocks import MultiHeadSelfAttention

    layers = ModelProbe(SilentLogger())._probe_layers(MultiHeadSelfAttention(8, num_heads=2))

    assert layers == ["query_key_value", "output_projection", "output_dropout"]


def test_layers_and_map_need_a_loaded_model():
    probe = ModelProbe(SilentLogger())

    assert probe.layers()["ok"] is False
    assert probe.map_png() is None


def test_map_png_renders_for_a_loaded_run():
    png = _probe().map_png()

    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def _loadable_run(tmp_path, group: str = "backbone", name: str = "run_ok", config_name: str = "model_config.json"):
    run_dir = tmp_path / group / name
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"x")
    (run_dir / "meta").mkdir()
    (run_dir / "meta" / config_name).write_text("{}")
    return run_dir


def test_runs_lists_loadable_backbone_runs(tmp_path):
    backbone = _loadable_run(tmp_path)

    bare = tmp_path / "backbone" / "run_bare"
    bare.mkdir(parents=True)
    (bare / "best_model.pt").write_bytes(b"x")

    _loadable_run(tmp_path, group="profile_ae", name="run_ae", config_name="profile_autoencoder_config.json")

    out = ModelProbe(SilentLogger()).runs(str(tmp_path))

    assert out["ok"] is True
    assert [run["id"] for run in out["runs"]] == [str(backbone)]


def test_runs_reports_root_errors(tmp_path):
    out = ModelProbe(SilentLogger()).runs(str(tmp_path / "nowhere"))

    assert out["ok"] is False
    assert out["runs"] == []


def test_load_refuses_a_directory_that_is_not_a_run(tmp_path):
    probe = ModelProbe(SilentLogger())
    probe.scanner.roots.add(tmp_path.resolve())

    missing = probe.start_load(str(tmp_path / "nowhere"))
    assert missing["ok"] is False
    assert "not a directory" in missing["error"]

    (tmp_path / "runs_root").mkdir()
    rootish = probe.start_load(str(tmp_path / "runs_root"))
    assert rootish["ok"] is False
    assert "best_model.pt" in rootish["error"]


def test_load_refuses_a_run_outside_every_catalogued_root(tmp_path):
    run_dir = tmp_path / "run_stray"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"x")
    (run_dir / "meta").mkdir()
    (run_dir / "meta" / "model_config.json").write_text("{}")

    out = ModelProbe(SilentLogger()).start_load(str(run_dir))

    assert out["ok"] is False
    assert "outside every catalogued runs root" in out["error"]


def test_load_refuses_a_run_without_the_persisted_config(tmp_path):
    run_dir = tmp_path / "run_bare"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"x")

    probe = ModelProbe(SilentLogger())
    probe.scanner.roots.add(tmp_path.resolve())

    out = probe.start_load(str(run_dir))

    assert out["ok"] is False
    assert "meta/model_config.json" in out["error"]




class _SharedPoolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_a = torch.nn.Conv2d(2, 4, kernel_size=1)
        self.conv_b = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.pool   = torch.nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.conv_b(self.pool(self.conv_a(x))))


def test_vitals_payload_resolves_a_reused_module_call_site():
    from tools.diagnostics.activation_recorder import ActivationRecorder

    model    = _SharedPoolModel()
    recorder = ActivationRecorder(model)
    recorder.attach_stats()

    with torch.no_grad():
        model(torch.randn(1, 2, 8, 8))

    recorder.detach()
    payload = ModelProbe._vitals_payload(model, recorder.stats())
    rows    = {entry["name"]: entry for entry in payload["entries"]}

    assert {"pool", "pool#1"} <= set(rows)
    assert rows["pool#1"]["params"] == 0
    assert rows["conv_b"]["params"] == 20
