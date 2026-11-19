
from __future__ import annotations

from enum    import Enum
from pathlib import Path

import pytest

from configuration.training                     import BackboneEntryConfig, DualEntryConfig
from configuration.training.jepa                import JepaEntryConfig
from configuration.training.profile_autoencoder import ProfileAeEntryConfig
from pipelines.backbone.training.launcher       import BackboneTrainingLauncher, TrainScheduler
from pipelines.dual.training.launcher           import DualTrainingLauncher, DualTrainScheduler
from pipelines.jepa.training.launcher           import JepaTrainingLauncher, JepaTrialScheduler
from pipelines.profile_autoencoder.training.launcher import ProfileAeTrainingLauncher, ProfileAeTrialScheduler
from tools.runtime.config_cli                   import ConfigCli


DATASET_MODES = ("secondary", "input", "patch", "reach", "ratio", "ablation")

EXTRA_ARGS = {
    "ae_run"  : ["--ae_run_trials.runs", "['runA', 'runB']"],
    "masking" : ["--profile_autoencoder_run", "runA"],
}

ENTRIES = [
    ("train_profile_autoencoder", ProfileAeEntryConfig, ProfileAeTrainingLauncher, ProfileAeTrialScheduler),
    ("train_jepa",                JepaEntryConfig,      JepaTrainingLauncher,      JepaTrialScheduler),
    ("train_backbone",            BackboneEntryConfig,  BackboneTrainingLauncher,  TrainScheduler),
    ("train_dual",                DualEntryConfig,      DualTrainingLauncher,      DualTrainScheduler),
]


def _resolve(entry_config, overrides: dict):
    cli = ConfigCli(entry_config(), description="trial override round trip")

    return cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])


def _value_at(config, path: str):
    value = config

    for part in path.split("."):
        value = getattr(value, part)

    return value


def _matches(resolved, planned) -> bool:
    if isinstance(resolved, Enum):
        resolved = resolved.value

    if isinstance(planned, dict):
        return dict(resolved) == planned

    if isinstance(planned, (list, tuple)) and isinstance(resolved, (list, tuple)):
        return list(resolved) == list(planned)

    return resolved == planned


@pytest.mark.parametrize(("key", "entry_config", "launcher_class", "scheduler"), ENTRIES)
def test_every_planned_override_lands_on_the_resolved_config(key, entry_config, launcher_class, scheduler):
    launcher = launcher_class(entry_script=Path(f"main/training/{key}.py"))
    checked  = 0

    for mode in scheduler.MODE_SUBDIRS:
        if mode in DATASET_MODES:
            continue

        argv      = ["--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]"] + EXTRA_ARGS.get(mode, [])
        plans     = launcher.build_scheduler(argv).planner().plan()

        assert plans, f"{key}/{mode} planned no unit"

        for run_name, overrides in plans:
            config = _resolve(entry_config, overrides)

            for path, planned in overrides.items():
                resolved = _value_at(config, path)
                assert _matches(resolved, planned), f"{key}/{mode}/{run_name}: {path} resolved to {resolved!r}, planner asked for {planned!r}"
                checked += 1

    assert checked > 0
