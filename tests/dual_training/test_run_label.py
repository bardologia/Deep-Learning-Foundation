from __future__ import annotations

from types import SimpleNamespace

from configuration.training              import DualEntryConfig
from pipelines.dual.training.launcher    import DualSingleTrainRunner
from pipelines.shared.training.run_naming import RunNaming


def _runner(monkeypatch, **fields) -> DualSingleTrainRunner:
    config = DualEntryConfig()
    for name, value in fields.items():
        setattr(config, name, value)

    runner = DualSingleTrainRunner(config)
    monkeypatch.setattr(runner.factory, "gaussian_config", lambda: SimpleNamespace(n_default_gaussians=2))
    return runner


def test_label_carries_the_default_trunk_routing(monkeypatch):
    runner = _runner(monkeypatch)

    expected = RunNaming.training_tag("dual_unet_skip", "set_pred", runner.config.curriculum, 2, runner.config.augmentation, extras=("full.full",))

    assert runner.label == expected
    assert runner.label.startswith("dual_unet_skip-")
    assert "-full.full-" in runner.label


def test_label_carries_the_selected_trunk_backbones(monkeypatch):
    runner = _runner(monkeypatch, params_backbone="resunet", existence_backbone="resunet")

    assert runner.label.startswith("dual_resunet-")


def test_label_names_both_trunks_when_they_differ(monkeypatch):
    runner = _runner(monkeypatch, params_backbone="unet_skip", existence_backbone="local_cnn")

    assert runner.label.startswith("dual_unet_skip.local_cnn-")


def test_label_follows_the_selected_inputs(monkeypatch):
    runner = _runner(monkeypatch, params_input=("ifg",), existence_input=("pass", "ifg"))

    assert "-ifg.full-" in runner.label


def test_label_keeps_the_loss_tag_last(monkeypatch):
    runner = _runner(monkeypatch)

    assert runner.label.split("-full.full-")[1] == RunNaming.loss_tag(runner.config.curriculum.active_stages()[-1])
