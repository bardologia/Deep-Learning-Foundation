"""Tests for resuming a training run or sweep from a pasted path with the configuration it recorded."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.training                        import BackboneEntryConfig
from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.training.resume_source       import ResumeLauncher, SweepRecording, TrainingResume
from tools.runtime.config_cli                      import ConfigCli

from tests.conftest import RecordingLogger


def _recorded(path: Path, **leaves) -> Path:
    """Writes a backbone entry recording at the path with the given leaf values."""
    config = BackboneEntryConfig()
    for key, value in leaves.items():
        ConfigCli.set_path(config, key, value)
    return ConfigCli.save_resolved(config, path)


def _launch(resume_from: str, *argv: str) -> tuple:
    """Returns the launch config and its CLI for a resume of the path with the given overrides."""
    cli    = ConfigCli(BackboneEntryConfig())
    config = cli.apply(["--resume_from", resume_from, *argv])
    return config, cli


def test_a_run_reads_its_own_recording(tmp_path):
    """A pasted run directory is the run kind and loads docs/resolved_entry_config.json."""
    run = tmp_path / "label_myrun"
    _recorded(ResolvedEntryConfigWriter.path_for(run), **{"training.epochs": 7})

    launch, cli = _launch(str(run))
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)
    config      = resume.configure()

    assert resume.kind            == TrainingResume.RUN
    assert resume.root            == run
    assert resume.recording       == ResolvedEntryConfigWriter.path_for(run)
    assert config.training.epochs == 7
    assert config.resume_from     == str(run)


def test_a_unit_without_its_own_recording_reads_the_sweep_recording(tmp_path):
    """A fresh unit handed its directory by a sweep reads the sweep's recording found above it."""
    root = tmp_path / "ema"
    SweepRecording.write(root, BackboneEntryConfig(training=BackboneEntryConfig().training.__class__(epochs=9)))
    (root / "rung_a").mkdir()

    launch, cli = _launch(str(root / "rung_a" / "seed0"))
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False, unit=True)
    config      = resume.configure()

    assert resume.kind            == TrainingResume.RUN
    assert resume.recording       == SweepRecording.path(root)
    assert config.training.epochs == 9


def test_a_unit_prefers_its_own_recording_over_the_sweep_one(tmp_path):
    """A unit that already recorded its own configuration reads that, not the sweep recording above it."""
    root = tmp_path / "ema"
    _recorded(SweepRecording.path(root), **{"training.epochs": 9})
    _recorded(ResolvedEntryConfigWriter.path_for(root / "rung_a" / "seed0"), **{"training.epochs": 2})

    launch, cli = _launch(str(root / "rung_a" / "seed0"))
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False, unit=True)

    assert resume.recording               == ResolvedEntryConfigWriter.path_for(root / "rung_a" / "seed0")
    assert resume.configure().training.epochs == 2


def test_a_seed_sweep_reads_its_recording_and_takes_the_seeds_from_the_launch(tmp_path):
    """A sweep root, or one of its seed runs, is the seed sweep kind and reads the sweep recording."""
    root = tmp_path / "label_exp"
    (root / "seed0").mkdir(parents=True)
    _recorded(SweepRecording.path(root), **{"training.epochs": 11, "seeds": [0, 1, 2]})

    launch, cli = _launch(str(root / "seed0"), "--seeds", "[0, 1, 2, 3]")
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)
    config      = resume.configure()

    assert resume.kind            == TrainingResume.SEED_SWEEP
    assert resume.root            == root
    assert config.training.epochs == 11
    assert config.seeds           == [0, 1, 2, 3]


def test_a_seed_sweep_without_a_recording_reads_its_first_seed_run(tmp_path):
    """A sweep launched before recordings existed is read through its first seed run, with the per-unit fields from the launch."""
    root = tmp_path / "label_exp"
    _recorded(ResolvedEntryConfigWriter.path_for(root / "seed1"), **{"training.epochs": 5, "seeds": [1], "seed": 1, "run_name": "exp/seed1"})
    _recorded(ResolvedEntryConfigWriter.path_for(root / "seed0"), **{"training.epochs": 5, "seeds": [0], "seed": 0, "run_name": "exp/seed0"})

    launch, cli = _launch(str(root))
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)
    config      = resume.configure()

    assert resume.recording       == ResolvedEntryConfigWriter.path_for(root / "seed0")
    assert config.training.epochs == 5
    assert config.seeds           == launch.seeds
    assert config.run_name        == launch.run_name
    assert "seeds" not in resume.changed


def test_a_trial_sweep_reads_the_sweep_recording_above_any_unit(tmp_path):
    """A trial sweep launch reads batch_train_logs/resolved_config.json found at or above the pasted unit."""
    root = tmp_path / "patch"
    _recorded(SweepRecording.path(root), **{"training.epochs": 3, "trials_enabled": True, "trials_mode": "patch"})
    (root / "rung_a" / "seed0").mkdir(parents=True)

    launch, cli = _launch(str(root / "rung_a" / "seed0"), "--trials_enabled", "true", "--trials_mode", "patch")
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=True)
    config      = resume.configure()

    assert resume.kind            == TrainingResume.TRIAL_SWEEP
    assert resume.root            == root
    assert config.training.epochs == 3
    assert config.trials_mode     == "patch"


def test_a_sweep_root_is_a_trial_sweep_without_the_launch_saying_so(tmp_path):
    """A pasted sweep root is recognised from its own recording, so the sweep settings need not be retyped."""
    root = tmp_path / "patch"
    _recorded(SweepRecording.path(root), **{"training.epochs": 3, "trials_enabled": True, "trials_mode": "patch"})
    (root / "rung_a").mkdir(parents=True)

    launch, cli = _launch(str(root))
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)
    config      = resume.configure()

    assert resume.kind             == TrainingResume.TRIAL_SWEEP
    assert resume.root             == root
    assert resume.recording        == SweepRecording.path(root)
    assert config.trials_enabled   is True
    assert config.trials_mode      == "patch"
    assert config.training.epochs  == 3


def test_a_run_inside_a_trial_sweep_stays_a_run(tmp_path):
    """A unit directory below a sweep root resumes that unit, not the whole sweep, even though the sweep recorded itself."""
    root = tmp_path / "patch"
    _recorded(SweepRecording.path(root), **{"training.epochs": 3, "trials_enabled": True})
    unit = root / "rung_a"
    unit.mkdir(parents=True)

    launch, cli = _launch(str(unit))
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)

    assert resume.kind      == TrainingResume.RUN
    assert resume.root      == unit
    assert resume.recording == SweepRecording.path(root)


def test_hardware_and_typed_overrides_win_over_the_recording(tmp_path):
    """The GPU fields follow the launch and a typed override replaces the recorded value, both reported."""
    run = tmp_path / "run"
    _recorded(ResolvedEntryConfigWriter.path_for(run), **{"training.epochs": 7, "gpus": [5, 6], "gpu": 5})

    launch, cli = _launch(str(run), "--training.epochs", "9")
    launch.gpus = [0]
    launch.gpu  = 0
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)
    config      = resume.configure()

    assert config.gpus            == [0]
    assert config.gpu             == 0
    assert config.training.epochs == 9
    assert resume.changed         == {"training.epochs": (7, 9)}


def test_a_directory_without_any_recording_keeps_the_launch_settings(tmp_path):
    """A run that recorded nothing resumes from the launch configuration, and the report warns about it."""
    run = tmp_path / "bare_run"
    run.mkdir()

    launch, cli = _launch(str(run), "--training.epochs", "4")
    resume      = TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)
    config      = resume.configure()

    logger = RecordingLogger()
    resume.log(logger)

    assert config is launch
    assert resume.recording is None
    assert any(level == "warning" for level, _ in logger.messages)


def test_a_path_without_a_parent_directory_is_refused(tmp_path):
    """A resume path whose parent does not exist cannot be created, so it is refused."""
    launch, cli = _launch(str(tmp_path / "nope" / "unit"))

    with pytest.raises(FileNotFoundError, match="no parent directory"):
        TrainingResume(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)


def test_sweep_recording_keeps_the_earlier_configuration_when_it_changes(tmp_path):
    """Recording a different configuration moves the previous one into the history directory."""
    root = tmp_path / "sweep"
    first  = BackboneEntryConfig()
    second = BackboneEntryConfig()
    second.training.epochs = 99

    SweepRecording.write(root, first)
    SweepRecording.write(root, first)
    SweepRecording.write(root, second)

    history = sorted((root / "batch_train_logs" / "config_history").glob("*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text())["training.epochs"] == first.training.epochs
    assert json.loads(SweepRecording.path(root).read_text())["training.epochs"] == 99


def test_launcher_returns_the_config_untouched_without_a_resume_path(tmp_path):
    """Without resume_from the launcher hands the parsed configuration back as it is."""
    cli    = ConfigCli(BackboneEntryConfig())
    config = cli.apply([])

    assert ResumeLauncher.configure(BackboneEntryConfig, config, cli.overrides, trial_sweep=False) is config


def test_launcher_logs_the_resume_into_the_resumed_directory(tmp_path):
    """The resume report lands in the resumed run's logs directory."""
    run = tmp_path / "run"
    _recorded(ResolvedEntryConfigWriter.path_for(run), **{"training.epochs": 7})

    launch, cli = _launch(str(run))
    config      = ResumeLauncher.configure(BackboneEntryConfig, launch, cli.overrides, trial_sweep=False)

    assert config.training.epochs == 7
    assert (run / "logs" / "resume_from.log").is_file()
