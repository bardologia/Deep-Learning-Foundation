"""Tests for the run label built by the dual single-run training launcher.

Covers the trunk backbone tag, the dotted form when the two trunks differ, the
input group tag, and the loss tag position.
"""

from __future__ import annotations

from types import SimpleNamespace

from configuration.training              import DualEntryConfig
from pipelines.dual.training.launcher    import DualSingleTrainRunner
from pipelines.shared.training.run_naming import RunNaming


def _runner(monkeypatch, **fields) -> DualSingleTrainRunner:
    """Returns a dual single-run runner with the given config fields and a stubbed Gaussian count of two."""
    config = DualEntryConfig()
    for name, value in fields.items():
        setattr(config, name, value)

    runner = DualSingleTrainRunner(config)
    monkeypatch.setattr(runner.factory, "gaussian_config", lambda: SimpleNamespace(n_default_gaussians=2))
    return runner


def test_label_carries_the_default_trunk_routing(monkeypatch):
    """The default label names the shared trunk backbone and the full input routing."""
    runner = _runner(monkeypatch)

    expected = RunNaming.training_tag("dual_unet_skip", "set_pred", runner.config.curriculum, 2, runner.config.augmentation, extras=("full.full",))

    assert runner.label == expected
    assert runner.label.startswith("dual_unet_skip-")
    assert "-full.full-" in runner.label


def test_label_carries_the_selected_trunk_backbones(monkeypatch):
    """A shared non-default trunk backbone appears in the label."""
    runner = _runner(monkeypatch, params_backbone="resunet", existence_backbone="resunet")

    assert runner.label.startswith("dual_resunet-")


def test_label_names_both_trunks_when_they_differ(monkeypatch):
    """Differing trunk backbones are joined with a dot in the label."""
    runner = _runner(monkeypatch, params_backbone="unet_skip", existence_backbone="local_cnn")

    assert runner.label.startswith("dual_unet_skip.local_cnn-")


def test_label_follows_the_selected_inputs(monkeypatch):
    """The input group tag reflects the per-trunk input selection."""
    runner = _runner(monkeypatch, params_input=("derived",), existence_input=("chan", "derived", "aux"))

    assert "-derived.full-" in runner.label


def test_label_keeps_the_loss_tag_last(monkeypatch):
    """The segment after the input tag is the loss tag of the last active curriculum stage."""
    runner = _runner(monkeypatch)

    assert runner.label.split("-full.full-")[1] == RunNaming.loss_tag(runner.config.curriculum.active_stages()[-1])
