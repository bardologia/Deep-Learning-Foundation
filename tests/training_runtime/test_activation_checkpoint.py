"""Tests for the state-dict-transparent gradient-checkpoint wrapper."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tools.training.activation_checkpoint import ActivationCheckpointing, CheckpointedModule

from tests.conftest import SilentLogger


class _Net(nn.Module):
    """Tiny conv net with an encoder block and a head, mirroring the backbone layout."""

    def __init__(self):
        """Builds a two-conv encoder and a 1x1 head."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
        self.head    = nn.Conv2d(4, 6, 1)

    def forward(self, x):
        """Returns the head applied to the encoded input."""
        return self.head(self.encoder(x))


def _pair() -> tuple[_Net, _Net]:
    """Returns a plain net and a weight-identical copy with a wrapped encoder."""
    torch.manual_seed(0)
    plain   = _Net()
    wrapped = _Net()
    wrapped.load_state_dict(plain.state_dict())
    ActivationCheckpointing.apply(wrapped, ("encoder",), SilentLogger())

    return plain, wrapped


def test_wrapper_keeps_state_dict_keys_identical():
    """A wrapped model saves and loads the exact keys of an unwrapped one."""
    plain, wrapped = _pair()

    assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())

    fresh = _Net()
    ActivationCheckpointing.apply(fresh, ("encoder",), SilentLogger())
    fresh.load_state_dict(plain.state_dict())

    assert isinstance(fresh.encoder, CheckpointedModule)


def test_wrapper_reproduces_loss_and_gradients():
    """Checkpointed training produces the same loss and gradients as the plain forward."""
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
    """Evaluation and no-grad passes bypass the checkpointing machinery."""
    _plain, wrapped = _pair()
    wrapped.eval()

    with torch.no_grad():
        out = wrapped(torch.randn(1, 2, 8, 8))

    assert out.shape == (1, 6, 8, 8)


def test_matching_never_nests_wrappers():
    """A pattern matching a module and its children wraps only the outermost module."""
    names = ActivationCheckpointing.matching_names(_Net(), ("encoder*",))

    assert names == ["encoder"]


def test_unmatched_pattern_fails_loudly():
    """A pattern matching no submodule raises and names the top-level modules."""
    with pytest.raises(ValueError, match="encoder"):
        ActivationCheckpointing.apply(_Net(), ("decoder*",), SilentLogger())


def test_module_list_matches_expand_to_children():
    """Matching a ModuleList wraps its children, keeping the parent's iteration intact."""
    class _ListNet(nn.Module):
        """Net whose forward iterates a ModuleList of blocks."""

        def __init__(self):
            """Builds two conv blocks in a ModuleList."""
            super().__init__()
            self.blocks = nn.ModuleList([nn.Conv2d(2, 2, 3, padding=1) for _ in range(2)])

        def forward(self, x):
            """Applies every block in sequence."""
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
