from __future__ import annotations

import pytest
import torch

from models.backbone.nafnet  import NAFNet
from models.backbone.resunet import ResUNetBackbone
from models.backbone.unet    import UNetBackbone
from models.dual             import DUAL_MODEL_REGISTRY, get_dual


WINDOW = 32
BATCH  = 2

OVERRIDES = {
    "in_channels"         : 5,
    "params_channels"     : (0, 1, 2, 3, 4),
    "existence_channels"  : (3, 4),
    "params_features"     : [8, 16],
    "existence_features"  : [8],
    "params_overrides"    : {"bottleneck_factor": 1, "dropout": 0.0},
    "existence_overrides" : {"bottleneck_factor": 1, "dropout": 0.0},
}


def _build(**extra):
    model, config = get_dual("dual_resunet", **{**OVERRIDES, **extra})
    return model.eval(), config


def _force_gate(model, logit_value: float) -> None:
    final_conv = model.existence_head.mlp[-1]
    torch.nn.init.zeros_(final_conv.weight)
    torch.nn.init.constant_(final_conv.bias, logit_value)


def _gate(model, x: torch.Tensor) -> torch.Tensor:
    embedding = model.trunk_existence.encode_decode(x.index_select(1, model.existence_index))
    return torch.sigmoid(model.existence_head(embedding))


def test_registry_holds_dual_resunet():
    assert set(DUAL_MODEL_REGISTRY) == {"dual_resunet"}


def test_default_build_is_the_ninety_ten_full_full_standard():
    model, config = get_dual("dual_resunet")

    assert config.params_backbone    == "unet_skip"
    assert config.existence_backbone == "unet_skip"
    assert config.params_channels    == tuple(range(9))
    assert config.existence_channels == tuple(range(9))
    assert config.params_features    == [64, 128, 248, 472]
    assert config.existence_features == [24, 48, 84, 156]
    assert len(model.trunk_params.encoder_blocks)    == 4
    assert len(model.trunk_existence.encoder_blocks) == 4

    params_arm = sum(p.numel() for name, p in model.named_parameters() if name.startswith(("trunk_params.", "gaussian_heads.")))
    total      = sum(p.numel() for p in model.parameters())

    assert abs(params_arm / total - 0.9) < 0.01


def test_trunk_architectures_are_selectable():
    model, config = _build(params_backbone="unet", existence_backbone="unet_skip")

    assert isinstance(model.trunk_params, UNetBackbone)
    assert isinstance(model.trunk_existence, ResUNetBackbone)
    assert model.trunk_existence.downsample_mode == "maxpool"

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_trunks_take_independent_widths():
    model, config = _build(existence_features=[4])

    assert model.trunk_existence.embedding_channels == 4
    assert model.existence_head.mlp[0].in_channels  == 4
    assert model.gaussian_heads[0].mlp[0].in_channels == model.trunk_params.embedding_channels

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_forward_shape_is_three_k_channels():
    model, config = _build()

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_existence_trunk_sees_only_its_channels():
    model, config = _build()

    x         = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)
    perturbed = x.clone()
    perturbed[:, :3] = torch.randn(BATCH, 3, WINDOW, WINDOW)

    with torch.no_grad():
        assert torch.equal(_gate(model, x), _gate(model, perturbed))


def test_params_trunk_sees_only_its_channels():
    model, config = _build(params_channels=(0, 1, 2))

    x         = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)
    perturbed = x.clone()
    perturbed[:, 3:] = torch.randn(BATCH, 2, WINDOW, WINDOW)

    with torch.no_grad():
        embedding_a = model.trunk_params.encode_decode(x.index_select(1, model.params_index))
        embedding_b = model.trunk_params.encode_decode(perturbed.index_select(1, model.params_index))

    assert torch.equal(embedding_a, embedding_b)


def test_params_trunk_sees_all_channels_by_default():
    model, config = _build()

    x         = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)
    perturbed = x.clone()
    perturbed[:, :3] = torch.randn(BATCH, 3, WINDOW, WINDOW)

    with torch.no_grad():
        assert not torch.equal(model(x), model(perturbed))


def test_closed_gate_pins_amplitude_to_off_level():
    model, config = _build()
    _force_gate(model, logit_value=-30.0)

    with torch.no_grad():
        model.amp_off.copy_(torch.tensor([-1.5, 0.75]))
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    ppg = config.params_per_gaussian
    amp = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)[:, :, 0]

    assert torch.allclose(amp[:, 0], torch.full_like(amp[:, 0], -1.5), atol=1e-5)
    assert torch.allclose(amp[:, 1], torch.full_like(amp[:, 1],  0.75), atol=1e-5)


def test_open_gate_passes_raw_amplitude():
    model, config = _build()
    _force_gate(model, logit_value=30.0)

    x = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        out       = model(x)
        embedding = model.trunk_params.encode_decode(x.index_select(1, model.params_index))
        raw       = torch.stack([head(embedding) for head in model.gaussian_heads], dim=1)

    ppg = config.params_per_gaussian
    amp = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)[:, :, 0]

    assert torch.allclose(amp, raw[:, :, 0], atol=1e-5)


def test_gate_leaves_mu_sigma_untouched():
    model, config = _build()
    _force_gate(model, logit_value=-30.0)

    x = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        out       = model(x)
        embedding = model.trunk_params.encode_decode(x.index_select(1, model.params_index))
        raw       = torch.stack([head(embedding) for head in model.gaussian_heads], dim=1)

    ppg      = config.params_per_gaussian
    reshaped = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)

    assert torch.allclose(reshaped[:, :, 1:], raw[:, :, 1:], atol=1e-5)


def test_gradients_reach_both_trunks_and_heads():
    model, config = _build()
    model.train()

    out  = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))
    loss = out.pow(2).mean()
    loss.backward()

    for module in (model.trunk_params, model.trunk_existence, model.existence_head, model.gaussian_heads):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

    assert model.amp_off.grad is not None


def test_param_groups_cover_every_parameter_exactly_once():
    model, config = _build()

    grouped = [id(p) for group in config.get_param_groups(model) for p in group["params"]]
    every   = [id(p) for p in model.parameters()]

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(every)


def test_any_zoo_backbone_is_a_valid_trunk():
    small_nafnet  = {"width": 8, "enc_blocks": [1, 1], "middle_blocks": 1, "dec_blocks": [1, 1]}
    model, config = _build(existence_backbone="nafnet", existence_features=[], existence_overrides=small_nafnet)

    assert isinstance(model.trunk_existence, NAFNet)
    assert model.trunk_existence.embedding_channels == 8

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_zoo_trunks_carry_no_head_parameters():
    model, _ = _build()

    trunk_keys = [name for name, _ in model.trunk_params.named_parameters()]

    assert all(not key.startswith(("output_head", "gaussian_heads", "existence_head", "head_")) for key in trunk_keys)
    assert model.trunk_params.head_parameters() == []


def test_featureless_trunk_rejects_a_features_ladder():
    with pytest.raises(ValueError, match="no 'features' ladder"):
        _build(existence_backbone="nafnet", existence_features=[8])


def test_reserved_trunk_override_rejected():
    with pytest.raises(ValueError, match="set by the dual model itself"):
        _build(params_overrides={"features": [8, 16]})


def test_unknown_trunk_override_rejected():
    with pytest.raises(AttributeError, match="Unknown trunk override"):
        _build(params_overrides={"bottleneck_factor": 1, "no_such_field": 1})


def test_unknown_trunk_backbone_rejected():
    with pytest.raises(ValueError, match="params_backbone"):
        _build(params_backbone="resnet_trunk")


def test_empty_trunk_channels_rejected():
    with pytest.raises(ValueError, match="existence_channels"):
        _build(existence_channels=())


def test_out_of_range_trunk_channel_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _build(existence_channels=(3, 5))


def test_non_set_pred_head_rejected():
    with pytest.raises(ValueError, match="set_pred"):
        _build(head="conv")
