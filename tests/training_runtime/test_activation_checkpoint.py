
from __future__ import annotations

import torch
import torch.nn as nn

from tools.training.activation_checkpoint import ActivationCheckpointing, CheckpointedModule

from tests.conftest import SilentLogger


class _Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
        self.head    = nn.Conv2d(4, 6, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


def _pair() -> tuple[_Net, _Net]:
    torch.manual_seed(0)
    plain   = _Net()
    wrapped = _Net()
    wrapped.load_state_dict(plain.state_dict())
    ActivationCheckpointing.apply(wrapped, ("encoder",), SilentLogger())

    return plain, wrapped


def test_wrapper_keeps_state_dict_keys_identical():
    plain, wrapped = _pair()

    assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())

    fresh = _Net()
    ActivationCheckpointing.apply(fresh, ("encoder",), SilentLogger())
    fresh.load_state_dict(plain.state_dict())

    assert isinstance(fresh.encoder, CheckpointedModule)


def test_wrapper_reproduces_loss_and_gradients():
    plain, wrapped = _pair()
    x = torch.randn(2, 2, 8, 8, generator=torch.Generator().manual_seed(1))

    plain.train()
    wrapped.train()

    loss_plain = plain(x).square().mean()
    loss_plain.backward()
    loss_wrapped = wrapped(x).square().mean()
    loss_wrapped.backward()

    assert torch.allclose(loss_plain, loss_wrapped)
    assert torch.allclose(plain.encoder[0].weight.grad, wrapped.encoder.module[0].weight.grad, atol=1e-6)


def test_wrapper_runs_the_module_directly_in_eval():
    _plain, wrapped = _pair()
    wrapped.eval()

    with torch.no_grad():
        out = wrapped(torch.randn(1, 2, 8, 8))

    assert out.shape == (1, 6, 8, 8)


def test_matching_never_nests_wrappers():
    names = ActivationCheckpointing.matching_names(_Net(), ("encoder*",))

    assert names == ["encoder"]


def test_module_list_matches_expand_to_children():
    class _ListNet(nn.Module):

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Conv2d(2, 2, 3, padding=1) for _ in range(2)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    net   = _ListNet()
    names = ActivationCheckpointing.apply(net, ("blocks",), SilentLogger())

    assert names == ["blocks.0", "blocks.1"]
    assert all(isinstance(block, CheckpointedModule) for block in net.blocks)

    net.train()
    out = net(torch.randn(1, 2, 8, 8, requires_grad=True))
    out.mean().backward()
