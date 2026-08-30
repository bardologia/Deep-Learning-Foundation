"""Tests for capturing attention weights and attention gates out of backbone models.

Covers the hook/patch lifecycle of `AttentionCapture` (attach, detach, release on a
failing forward) and the summary statistics `AttentionSummary` derives from captured
weight tensors of shape (batch, heads, queries, keys).
"""

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
    """Runs one forward pass of a small backbone with attention capture attached.

    Args:
        name         : Backbone registry name to instantiate with the small test overrides.
        in_channels  : Channel count of the random probe window.
        out_channels : Channel count the backbone predicts.

    Returns:
        The capture records mapping, with a "gates" and an "attention" entry.
    """
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
    """Attention U-Net exposes single-channel gates bounded to [0, 1] and no attention weights."""
    records = _capture("attention_unet")

    assert records["gates"]
    assert not records["attention"]

    for gate in records["gates"].values():
        assert gate.shape[1] == 1
        assert float(gate.min()) >= 0.0
        assert float(gate.max()) <= 1.0


def test_swin_attention_weights_are_captured_and_labelled():
    """Swin U-Net records named attention weights whose key axis sums to one."""
    records = _capture("swin_unet")

    assert records["attention"]
    for name, weight_list in records["attention"].items():
        assert "attention" in name.lower() or weight_list
        for weights in weight_list:
            sums = weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_segformer_multihead_attention_is_patched():
    """SegFormer records at least four multi-head attention calls per forward pass."""
    records = _capture("segformer")

    assert records["attention"]
    total_calls = sum(len(weight_list) for weight_list in records["attention"].values())
    assert total_calls >= 4


def test_capture_detach_restores_the_model():
    """Detaching restores the original MultiheadAttention.forward bound methods."""
    model, _ = get_backbone("segformer", in_channels=3, out_channels=6, **SMALL_OVERRIDES["segformer"])
    model    = model.eval()

    mha_forwards = {name: module.forward for name, module in model.named_modules() if isinstance(module, torch.nn.MultiheadAttention)}

    assert mha_forwards

    capture = AttentionCapture(model)
    capture.attach()
    capture.detach()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            assert module.forward == mha_forwards[name]
            assert "forward" not in module.__dict__


def test_detach_disables_the_process_wide_tap():
    """Detaching clears the process-wide AttentionTap sink installed on attach."""
    model, _ = get_backbone("swin_unet", in_channels=3, out_channels=6, **SMALL_OVERRIDES["swin_unet"])

    capture = AttentionCapture(model.eval())
    capture.attach()

    assert AttentionTap._sink is not None

    capture.detach()

    assert AttentionTap._sink is None


def test_capture_releases_the_tap_when_the_forward_fails(tmp_path):
    """A raising forward still releases the tap and unpatches every module."""
    model, _ = get_backbone("swin_unet", in_channels=3, out_channels=6, **SMALL_OVERRIDES["swin_unet"])

    class _FailingModel:
        """Callable model stub that always raises, standing in for a failing forward pass."""
        def __init__(self, module) -> None:
            """Stores the wrapped module so its patched state can be inspected."""
            self.module = module

        def __call__(self, images):
            """Raises a RuntimeError instead of running the wrapped module."""
            raise RuntimeError("the forward failed")

    run = SimpleNamespace(model=_FailingModel(model.eval()), loader=[(torch.randn(1, 3, WINDOW, WINDOW),)])

    with pytest.raises(RuntimeError):
        AttentionCaptureRun(tmp_path, AttentionCaptureConfig(), SilentLogger())._capture(run)

    assert AttentionTap._sink is None
    assert all("forward" not in module.__dict__ for module in run.model.module.modules())


def test_model_without_attention_raises():
    """Attaching to a plain U-Net, which has neither gates nor attention, raises ValueError."""
    model, _ = get_backbone("unet", in_channels=3, out_channels=6, **SMALL_OVERRIDES["unet"])

    with pytest.raises(ValueError):
        AttentionCapture(model).attach()


def test_entropy_bounds():
    """Normalised entropy is one for uniform attention and zero for one-hot attention."""
    uniform = torch.full((1, 2, 4, 8), 1.0 / 8.0)
    onehot  = torch.zeros(1, 2, 4, 8)
    onehot[..., 0] = 1.0

    assert AttentionSummary.entropy_values(uniform).mean() == pytest.approx(1.0, abs=1e-6)
    assert AttentionSummary.entropy_values(onehot).mean()  == pytest.approx(0.0, abs=1e-6)
    assert AttentionSummary.peak(onehot)                   == pytest.approx(1.0)


def test_entropy_values_cover_every_query():
    """Entropy values are returned once per (batch, head, query) triple."""
    weights = torch.full((2, 3, 4, 8), 1.0 / 8.0)

    assert AttentionSummary.entropy_values(weights).shape == (2 * 3 * 4,)


def test_stack_calls_rejects_mismatched_shapes():
    """Stacking rejects calls of differing shapes and tensors that are not four-dimensional."""
    with pytest.raises(ValueError):
        AttentionSummary.stack_calls([torch.rand(1, 2, 4, 4), torch.rand(1, 2, 8, 8)])

    with pytest.raises(ValueError):
        AttentionSummary.stack_calls([torch.rand(1, 4, 4)])


def test_head_redundancy_separates_identical_from_distinct_heads():
    """Redundancy is one for cloned heads, below 0.9 for independent heads and None for a single head."""
    base      = torch.rand(1, 1, 6, 6).softmax(dim=-1)
    identical = base.expand(1, 3, 6, 6).clone()

    distinct         = torch.rand(2, 3, 6, 6).softmax(dim=-1)
    single_head      = torch.rand(1, 1, 6, 6).softmax(dim=-1)

    assert AttentionSummary.head_redundancy(identical) == pytest.approx(1.0, abs=1e-6)
    assert AttentionSummary.head_redundancy(distinct)  < 0.9
    assert AttentionSummary.head_redundancy(single_head) is None


def test_head_entropies_are_per_head():
    """Per-head entropies separate a uniform head from a one-hot head in the same tensor."""
    weights          = torch.zeros(1, 2, 4, 4)
    weights[:, 0]    = 1.0 / 4.0
    weights[:, 1, :, 0] = 1.0

    entropies = AttentionSummary.head_entropies(weights)

    assert entropies[0] == pytest.approx(1.0, abs=1e-6)
    assert entropies[1] == pytest.approx(0.0, abs=1e-6)


def test_attention_distance_zero_for_self_and_uniform_reference():
    """Mean attention distance is zero for identity attention and matches the uniform reference otherwise."""
    self_attention = torch.eye(16).reshape(1, 1, 16, 16)
    uniform        = torch.full((1, 1, 16, 16), 1.0 / 16.0)

    self_distance    = AttentionSummary.attention_distance(self_attention)
    uniform_distance = AttentionSummary.attention_distance(uniform)

    assert self_distance["tokens"]      == pytest.approx(0.0, abs=1e-9)
    assert uniform_distance["tokens"]   == pytest.approx(uniform_distance["uniform_tokens"])
    assert uniform_distance["relative"] == pytest.approx(uniform_distance["tokens"] / 4.0)


def test_attention_distance_undefined_off_square_grids():
    """Attention distance returns None when the token count is not a square or the map is not self-attention."""
    assert AttentionSummary.attention_distance(torch.rand(1, 1, 8, 8).softmax(dim=-1)) is None
    assert AttentionSummary.attention_distance(torch.rand(1, 1, 4, 16).softmax(dim=-1)) is None


def test_mean_attention_map_reshapes_to_the_token_grid():
    """The averaged attention map is folded back onto the square token grid."""
    weights = torch.full((2, 2, 16, 16), 1.0 / 16.0)

    mean_map = AttentionSummary.mean_attention_map(weights)

    assert mean_map.shape == (4, 4)
    assert np.allclose(mean_map, 1.0 / 16.0)


def test_gate_stats_report_mean_and_active_fraction():
    """Gate statistics report the mean gate value, the active fraction and the spatial shape."""
    gate  = torch.zeros(1, 1, 4, 4)
    gate[0, 0, :2] = 1.0

    stats = AttentionSummary.gate_stats(gate)

    assert stats["mean"]        == pytest.approx(0.5)
    assert stats["frac_active"] == pytest.approx(0.5)
    assert stats["spatial_shape"] == [4, 4]
