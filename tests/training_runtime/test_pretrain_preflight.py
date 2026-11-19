
from __future__ import annotations

from dataclasses import dataclass

import pytest

import pipelines.shared.training.pretrain_preflight as preflight_module

from pipelines.shared.training.pretrain_preflight import PretrainPreflight


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


def test_preflight_writes_inside_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)

    run_directory = tmp_path / "run_resunet_x"

    PretrainPreflight(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "resunet",
    ).run()

    assert _FakeOrchestrator.captured["label"] == "resunet"
    assert (run_directory / "pretrain").is_dir()
    assert (run_directory / "pretrain" / "logs").is_dir()


def test_resumed_run_refuses_to_probe_and_records_the_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)

    run_directory = tmp_path / "run_resunet_x"
    training      = _TrainingStub(batch_size=32, resume=True)

    with pytest.raises(RuntimeError, match=r"pretraining probes \(find_batch_size, find_lr\) cannot run on a run resuming from last.pt.*training.batch_size=32"):
        PretrainPreflight(
            pretrain_config = _pretrain(find_batch_size=True, find_lr=True),
            training_config = training,
            build_trainer   = lambda logger: None,
            run_directory   = run_directory,
            label           = "resunet",
        ).run()

    written = (run_directory / "pretrain" / "logs" / "pretrain.log").read_text(encoding="utf-8")

    assert ">>> ABORTED pretrain" in written
    assert "[Aborted] RuntimeError: pretraining probes" in written


def test_occupied_gpu_context_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(preflight_module.TrainStepMemoryProbe, "measure_context", staticmethod(lambda device: 3.25))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    run_directory = tmp_path / "run_resunet_x"

    PretrainPreflight(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "resunet",
    ).run()

    written = (run_directory / "pretrain" / "logs" / "pretrain.log").read_text(encoding="utf-8")

    assert "WARNING" in written
    assert "CUDA context 3.25 GB on CUDA_VISIBLE_DEVICES=1 exceeds a bare context" in written
    assert "smaller training.batch_size" in written


def test_bare_gpu_context_is_not_warned(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(preflight_module.TrainStepMemoryProbe, "measure_context", staticmethod(lambda device: 0.4))

    run_directory = tmp_path / "run_resunet_x"

    PretrainPreflight(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "resunet",
    ).run()

    written = (run_directory / "pretrain" / "logs" / "pretrain.log").read_text(encoding="utf-8")

    assert "CUDA context before the probe builds anything: 0.40 GB" in written
    assert "exceeds a bare context" not in written


def test_disabled_preflight_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "PretrainOrchestrator", _FakeOrchestrator)

    run_directory = tmp_path / "run_resunet_x"

    PretrainPreflight(
        pretrain_config = _pretrain(),
        training_config = _training(),
        build_trainer   = lambda logger: None,
        run_directory   = run_directory,
        label           = "resunet",
    ).run()

    assert not run_directory.exists()
