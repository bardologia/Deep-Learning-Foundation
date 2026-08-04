from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.diagnostics                      import AttentionCaptureConfig
from models.backbone                                import get_backbone
from models.blocks                                  import AttentionTap
from pipelines.backbone.inference.analysis.attention_capture import AttentionCapture, AttentionCaptureRun, AttentionSummary

from tests.conftest                 import SilentLogger
from tests.models_backbone._helpers import SMALL_OVERRIDES, WINDOW


def _capture(name: str, in_channels: int = 3, out_channels: int = 6):
    model, _ = get_backbone(name, in_channels=in_channels, out_channels=out_channels, **SMALL_OVERRIDES[name])
    model    = model.eval()

    capture = AttentionCapture(model)
    capture.attach()

    try:
        with torch.no_grad():
            model(torch.randn(1, in_channels, WINDOW, WINDOW))
    finally:
        capture.detach()

    return capture.records()


def test_attention_unet_gates_are_captured():
    records = _capture("attention_unet")

    assert records["gates"]
    assert not records["attention"]

    for gate in records["gates"].values():
        assert gate.shape[1] == 1
        assert float(gate.min()) >= 0.0
        assert float(gate.max()) <= 1.0


def test_swin_attention_weights_are_captured_and_labelled():
    records = _capture("swin_unet")

    assert records["attention"]
    for name, weight_list in records["attention"].items():
        assert "attention" in name.lower() or weight_list
        for weights in weight_list:
            sums = weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_segformer_multihead_attention_is_patched():
    records = _capture("segformer")

    assert records["attention"]
    total_calls = sum(len(weight_list) for weight_list in records["attention"].values())
    assert total_calls >= 4


def test_capture_detach_restores_the_model():
    model, _ = get_backbone("segformer", in_channels=3, out_channels=6, **SMALL_OVERRIDES["segformer"])
    model    = model.eval()

    mha_forwards = {name: module.forward for name, module in model.named_modules() if isinstance(module, torch.nn.MultiheadAttention)}

    capture = AttentionCapture(model)
    capture.attach()
    capture.detach()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            assert module.forward == mha_forwards[name]
            assert "forward" not in module.__dict__


def test_detach_disables_the_process_wide_tap():
    model, _ = get_backbone("swin_unet", in_channels=3, out_channels=6, **SMALL_OVERRIDES["swin_unet"])

    capture = AttentionCapture(model.eval())
    capture.attach()

    assert AttentionTap._sink is not None

    capture.detach()

    assert AttentionTap._sink is None


def test_capture_releases_the_tap_when_the_forward_fails(tmp_path):
    model, _ = get_backbone("swin_unet", in_channels=3, out_channels=6, **SMALL_OVERRIDES["swin_unet"])

    class _FailingModel:
        def __init__(self, module) -> None:
            self.module = module

        def __call__(self, images):
            raise RuntimeError("the forward failed")

    run = SimpleNamespace(model=_FailingModel(model.eval()), loader=[(torch.randn(1, 3, WINDOW, WINDOW),)])

    with pytest.raises(RuntimeError):
        AttentionCaptureRun(tmp_path, AttentionCaptureConfig(), SilentLogger())._capture(run)

    assert AttentionTap._sink is None
    assert all("forward" not in module.__dict__ for module in run.model.module.modules())


def test_model_without_attention_raises():
    model, _ = get_backbone("unet", in_channels=3, out_channels=6, **SMALL_OVERRIDES["unet"])

    with pytest.raises(ValueError):
        AttentionCapture(model).attach()


def test_entropy_bounds():
    uniform = torch.full((1, 2, 4, 8), 1.0 / 8.0)
    onehot  = torch.zeros(1, 2, 4, 8)
    onehot[..., 0] = 1.0

    assert AttentionSummary.entropy(uniform) == pytest.approx(1.0, abs=1e-6)
    assert AttentionSummary.entropy(onehot)  == pytest.approx(0.0, abs=1e-6)
    assert AttentionSummary.peak(onehot)     == pytest.approx(1.0)


def test_gate_stats_report_mean_and_active_fraction():
    gate  = torch.zeros(1, 1, 4, 4)
    gate[0, 0, :2] = 1.0

    stats = AttentionSummary.gate_stats(gate)

    assert stats["mean"]        == pytest.approx(0.5)
    assert stats["frac_active"] == pytest.approx(0.5)
    assert stats["spatial_shape"] == [4, 4]
