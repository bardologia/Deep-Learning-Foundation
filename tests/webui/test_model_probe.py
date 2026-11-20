from __future__ import annotations

import base64

import numpy as np
import pytest
import torch

from model_probe import ModelProbe

from tests.conftest           import SilentLogger
from tests.webui._probe_fakes import PH, PW, RowShiftModel, TinyConvModel, TwoChannelModel, fake_run


def _probe(model: torch.nn.Module | None = None) -> ModelProbe:
    probe = ModelProbe(SilentLogger())

    probe.loaded = {
        "run"    : fake_run(model),
        "layers" : ["conv"],
        "types"  : {"conv": "Conv2d"},
        "patch"  : (PH, PW),
        "device" : "cpu",
    }
    probe.status = {"state": "ready", "path": "fake", "progress": 1.0, "stage": "ready", "error": "", "info": None}

    return probe


def test_predict_returns_every_target_channel_against_its_target():
    probe = _probe()
    run   = probe.loaded["run"]

    result = probe.predict({"row": 10, "col": 8})

    assert result["ok"] is True
    assert [t["name"] for t in result["targets"]] == ["first", "second"]
    assert result["targets"][0]["prediction"] == pytest.approx(float(run.inputs[0, 10, 8]) + 1.0)
    assert result["targets"][1]["error"]      == pytest.approx(-0.25)
    assert result["inputs"] == [{"name": "a", "value": pytest.approx(float(run.inputs[0, 10, 8]))}, {"name": "b", "value": pytest.approx(float(run.inputs[1, 10, 8]))}]


def test_predict_outside_region_fails():
    result = _probe().predict({"row": 999, "col": 0})

    assert result["ok"] is False
    assert "outside" in result["error"]


def test_predict_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).predict({"row": 0, "col": 0})["ok"] is False


def test_fields_return_prediction_target_and_error_maps_per_target():
    result = _probe().fields({"row": 10, "col": 8})

    assert result["ok"] is True
    assert result["patch"] == [PH, PW]
    assert [t["name"] for t in result["targets"]] == ["first", "second"]

    first = result["targets"][0]
    assert base64.b64decode(first["prediction"]["png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert first["error"]["min"] == pytest.approx(0.25, abs=1e-5)
    assert first["error"]["max"] == pytest.approx(0.25, abs=1e-5)


def test_fields_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).fields({"row": 0, "col": 0})["ok"] is False


def test_attribution_concentrates_on_the_used_channel():
    result = _probe().attribution({"row": 10, "col": 8})

    assert result["ok"] is True
    assert result["channels"] == ["a", "b"]

    for target in result["targets"]:
        assert target["dead"] is False
        assert target["shares"][0] == pytest.approx(1.0)
        assert target["cells"][1] is None


def test_attribution_isolates_each_targets_inputs():
    result = _probe(TwoChannelModel()).attribution({"row": 10, "col": 8})

    first, second = result["targets"]

    assert first["shares"][0]  == pytest.approx(1.0)
    assert second["shares"][1] == pytest.approx(1.0)


def test_attribution_radial_profile_collapses_to_the_center_for_a_pointwise_model():
    result = _probe().attribution({"row": 10, "col": 8})
    radial = result["targets"][0]["radial"]

    assert radial["r50"] == 0
    assert radial["r90"] == 0
    assert radial["cumulative"][0] == pytest.approx(1.0)


def test_ablation_ranks_the_used_channel_first():
    result = _probe().ablation({"row": 10, "col": 8})

    assert result["ok"] is True
    assert result["channels"][0]["label"] == "a"
    assert result["channels"][0]["delta"] > 0.0
    assert result["channels"][1]["delta"] == pytest.approx(0.0)


def test_ablation_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).ablation({"row": 0, "col": 0})["ok"] is False


def test_occlusion_peaks_on_the_cell_holding_the_probed_pixel():
    result = _probe().occlusion({"row": 10, "col": 8})

    assert result["ok"] is True
    assert [result["worst"]["row"], result["worst"]["col"]] == result["center_cell"]
    assert result["worst"]["delta"] > 0.0


def test_occlusion_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).occlusion({"row": 0, "col": 0})["ok"] is False


def test_flips_report_zero_shifts_for_a_pointwise_model():
    result = _probe().flips({"row": 10, "col": 8})

    assert result["ok"] is True
    assert result["channels"] == ["first", "second"]
    assert [e["flip"] for e in result["entries"]] == ["none", "row", "col", "both"]
    assert all(e["delta"] == pytest.approx(0.0, abs=1e-12) for e in result["entries"])


def test_flips_expose_a_row_asymmetric_model():
    result = _probe(RowShiftModel()).flips({"row": 10, "col": 8})
    shifts = {e["flip"]: e["delta"] for e in result["entries"]}

    assert shifts["col"] == pytest.approx(0.0, abs=1e-12)
    assert shifts["row"] > 1e-6


def test_flips_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).flips({"row": 0, "col": 0})["ok"] is False


def test_vitals_reports_layers_in_forward_order_with_params_and_flags():
    probe = _probe(TinyConvModel())

    result = probe.vitals({"row": 10, "col": 8})

    assert result["ok"] is True
    assert [e["name"] for e in result["entries"]] == ["conv", "act", "head"]
    assert [e["type"] for e in result["entries"]] == ["Conv2d", "ReLU", "Conv2d"]
    assert [e["params"] for e in result["entries"]] == [152, 0, 18]
    assert result["entries"][0]["shape"] == [8, PH, PW]

    summary = result["summary"]
    assert summary["n_layers"]     == 3
    assert summary["total_params"] == 170
    assert 0 <= summary["flagged"] <= 3

    for entry in result["entries"]:
        assert 0.0 <= entry["zero_frac"] <= 1.0
        assert isinstance(entry["flags"], list)


def test_vitals_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).vitals({"row": 0, "col": 0})["ok"] is False


def _conv_probe() -> ModelProbe:
    probe = _probe(TinyConvModel())
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

    with pytest.raises(AttributeError):
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

    _loadable_run(tmp_path, group="other", name="run_other", config_name="trainer_config.json")

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
