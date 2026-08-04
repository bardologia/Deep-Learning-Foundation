from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pipelines.shared.training.pretrain_preflight as preflight_module
import pipelines.shared.training.training_runner    as runner_module

from pipelines.shared.training.pretrain_preflight import PretrainPreflight
from pipelines.shared.training.training_runner    import EntryConfigTrainRunner


def _pretrain(**overrides):
    base = dict(find_batch_size=False, tune_loader=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _training():
    return SimpleNamespace(batch_size=1, num_workers=0, prefetch_factor=2, resume=False)


class _FakeOrchestrator:
    captured = {}

    def __init__(self, **kwargs):
        type(self).captured = kwargs

    def run(self):
        pass


class _FakePreflight:
    captured = {}

    def __init__(self, **kwargs):
        type(self).captured = kwargs

    def run(self):
        pass


class _Pipeline:
    run_label = "image_ae"

    def __init__(self, config):
        self.config = config

    def run(self):
        return ("trained", self.config.run_name)


class _Runner(EntryConfigTrainRunner):
    pipeline_class = _Pipeline

    @property
    def label(self) -> str:
        return "image_ae"


def test_preflight_writes_inside_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)

    run_directory = tmp_path / "run_image_ae_x"

    PretrainPreflight(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "image_ae",
    ).run()

    assert _FakeOrchestrator.captured["label"] == "image_ae"
    assert (run_directory / "pretrain").is_dir()
    assert (run_directory / "pretrain" / "logs").is_dir()


def test_disabled_preflight_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)

    run_directory = tmp_path / "run_image_ae_x"

    PretrainPreflight(
        pretrain_config = _pretrain(),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "image_ae",
    ).run()

    assert not run_directory.exists()


def test_runner_resolves_run_directory_before_the_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "PretrainPreflight", _FakePreflight)

    config = SimpleNamespace(logdir=str(tmp_path), run_name=None, resume=True, pretrain=_pretrain(find_batch_size=True), training=_training())
    runner = _Runner(config)

    result, resolved_name = runner.run()

    assert result == "trained"
    assert resolved_name.startswith("run_image_ae_")
    assert config.run_name == resolved_name
    assert _FakePreflight.captured["run_directory"] == Path(tmp_path) / resolved_name
    assert runner.run_directory                     == Path(tmp_path) / resolved_name


def test_runner_keeps_an_explicit_run_name(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "PretrainPreflight", _FakePreflight)

    config = SimpleNamespace(logdir=str(tmp_path), run_name="my_run", resume=True, pretrain=_pretrain(), training=_training())

    result, resolved_name = _Runner(config).run()

    assert resolved_name == "my_run"
    assert _FakePreflight.captured["run_directory"] == Path(tmp_path) / "my_run"
