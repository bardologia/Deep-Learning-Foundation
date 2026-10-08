"""Tests for the pre-training preflight step wired into the train runner.

Covers where preflight artefacts are written, the refusal on a resumed run,
the warning on a GPU other processes occupy, the disabled no-op path and run
directory resolution before the preflight is constructed.
"""

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
    """Pretrain settings stub with every probe disabled by default."""

    find_batch_size : bool = False
    tune_loader     : bool = False
    find_lr         : bool = False


@dataclass
class _TrainingStub:
    """Training settings stub with resume disabled."""

    batch_size          : int  = 1
    num_workers         : int  = 0
    prefetch_factor     : int  = 2
    resume              : bool = False
    scale_lr_with_batch : bool = True


@dataclass
class _EntryStub:
    """Entry config stub carrying the fields the train runner touches."""

    logdir      : str            = ""
    run_name    : str | None     = None
    resume      : bool           = True
    resume_from : str            = ""
    pretrain    : _PretrainStub  = field(default_factory=_PretrainStub)
    training    : _TrainingStub  = field(default_factory=_TrainingStub)


def _pretrain(**overrides):
    """Returns a pretrain settings stub with the given fields overridden."""
    return _PretrainStub(**overrides)


def _training():
    """Returns a training settings stub with resume disabled."""
    return _TrainingStub()


class _FakeOrchestrator:
    """Orchestrator stub capturing the keyword arguments it was built with."""
    captured = {}

    def __init__(self, **kwargs):
        """Records the construction keyword arguments on the class."""
        type(self).captured = kwargs

    def run(self):
        """Performs no orchestration."""
        pass


class _FakePreflight:
    """Preflight stub capturing the keyword arguments it was built with."""
    captured = {}

    def __init__(self, **kwargs):
        """Records the construction keyword arguments on the class."""
        type(self).captured = kwargs

    def run(self):
        """Performs no preflight."""
        pass


class _Pipeline:
    """Pipeline stub returning a marker and the resolved run name."""
    run_label = "image_ae"

    def __init__(self, config):
        """Stores the entry config."""
        self.config = config

    def run(self):
        """Returns a marker tuple carrying the resolved run name."""
        return ("trained", self.config.run_name)


class _Runner(EntryConfigTrainRunner):
    """Train runner bound to the stub pipeline."""
    pipeline_class = _Pipeline

    @property
    def label(self) -> str:
        """Returns the run label used in generated run names."""
        return "image_ae"


def test_preflight_writes_inside_the_run_directory(tmp_path, monkeypatch):
    """Verifies that preflight logs land under <run_directory>/pretrain."""
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
    """Verifies a run continuing from its trainer state refuses every enabled probe, naming them, and that pretrain.log ends in an ABORTED banner."""
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
    """Verifies a CUDA context above a bare one is warned about in pretrain.log, naming the mask and the batch-size consequence."""
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
    """Verifies a context within a bare CUDA context logs only the measurement."""
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
    """Verifies that a disabled preflight creates no run directory."""
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
    """Verifies that the run name is resolved before the preflight receives it."""
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
    """Verifies that an explicit run name is kept instead of being generated."""
    monkeypatch.setattr(runner_module, "PretrainPreflight", _FakePreflight)

    config = _EntryStub(logdir=str(tmp_path), run_name="my_run")

    result, resolved_name = _Runner(config).run()

    assert resolved_name == "my_run"
    assert _FakePreflight.captured["run_directory"] == Path(tmp_path) / "my_run"
