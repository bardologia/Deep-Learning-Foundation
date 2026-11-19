
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

import pipelines.shared.training.pretrain_preflight as preflight_module
import pipelines.shared.training.training_runner    as runner_module

from pipelines.shared.training.pretrain_preflight import PretrainPreflight
from pipelines.shared.training.training_runner    import EntryConfigTrainRunner


@dataclass
class _PretrainStub:

    find_batch_size : bool = False
    tune_loader     : bool = False
    find_lr         : bool = False


@dataclass
class _TrainingStub:

    batch_size          : int  = 1
    num_workers         : int  = 0
    prefetch_factor     : int  = 2
    resume              : bool = False
    scale_lr_with_batch : bool = True


@dataclass
class _EntryStub:

    logdir      : str            = ""
    run_name    : str | None     = None
    resume      : bool           = True
    resume_from : str            = ""
    pretrain    : _PretrainStub  = field(default_factory=_PretrainStub)
    training    : _TrainingStub  = field(default_factory=_TrainingStub)


def _pretrain(**overrides):
    return _PretrainStub(**overrides)


def _training():
    return _TrainingStub()


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


def test_resumed_run_refuses_to_probe_and_records_the_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)

    run_directory = tmp_path / "run_image_ae_x"
    training      = _TrainingStub(batch_size=32, resume=True)

    with pytest.raises(RuntimeError, match=r"pretraining probes \(find_batch_size, find_lr\) cannot run on a run resuming from last.pt.*training.batch_size=32"):
        PretrainPreflight(
            pretrain_config = _pretrain(find_batch_size=True, find_lr=True),
            training_config = training,
            build_trainer   = lambda logger: None,
            run_directory   = run_directory,
            label           = "image_ae",
        ).run()

    written = (run_directory / "pretrain" / "logs" / "pretrain.log").read_text(encoding="utf-8")

    assert ">>> ABORTED pretrain" in written
    assert "[Aborted] RuntimeError: pretraining probes" in written


def test_occupied_gpu_context_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(preflight_module.TrainStepMemoryProbe, "measure_context", staticmethod(lambda device: 3.25))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    run_directory = tmp_path / "run_image_ae_x"

    PretrainPreflight(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "image_ae",
    ).run()

    written = (run_directory / "pretrain" / "logs" / "pretrain.log").read_text(encoding="utf-8")

    assert "WARNING" in written
    assert "CUDA context 3.25 GB on CUDA_VISIBLE_DEVICES=1 exceeds a bare context" in written
    assert "smaller training.batch_size" in written


def test_bare_gpu_context_is_not_warned(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(preflight_module.TrainStepMemoryProbe, "measure_context", staticmethod(lambda device: 0.4))

    run_directory = tmp_path / "run_image_ae_x"

    PretrainPreflight(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "image_ae",
    ).run()

    written = (run_directory / "pretrain" / "logs" / "pretrain.log").read_text(encoding="utf-8")

    assert "CUDA context before the probe builds anything: 0.40 GB" in written
    assert "exceeds a bare context" not in written


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

    config = _EntryStub(logdir=str(tmp_path), pretrain=_pretrain(find_batch_size=True))
    runner = _Runner(config)

    result, resolved_name = runner.run()

    assert result == "trained"
    assert resolved_name.startswith("run_image_ae_")
    assert config.run_name == resolved_name
    assert _FakePreflight.captured["run_directory"] == Path(tmp_path) / resolved_name
    assert runner.run_directory                     == Path(tmp_path) / resolved_name


def test_runner_keeps_an_explicit_run_name(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "PretrainPreflight", _FakePreflight)

    config = _EntryStub(logdir=str(tmp_path), run_name="my_run")

    result, resolved_name = _Runner(config).run()

    assert resolved_name == "my_run"
    assert _FakePreflight.captured["run_directory"] == Path(tmp_path) / "my_run"
