"""Round-trip test taking every planned trial override back through the entry CLI.

A planner emitting a valid config path is not enough: the value it emits has to
survive `ConfigCli.to_argv` and land on the resolved entry config unchanged,
because that argv is exactly what each sweep unit is launched with. This module
plans every mode of every trials-capable entry and checks each override against
the config the unit would actually train under.
"""

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
"""Modes whose planner reads the preprocessing run or instantiates models at plan
time, so they cannot be planned without the server-side dataset."""

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
    """Applies one unit's overrides to a fresh entry config the way its sweep unit would.

    Args:
        entry_config : Entry configuration class of the training entry.
        overrides    : Per-unit overrides planned for that unit.

    Returns:
        The resolved entry configuration.
    """
    cli = ConfigCli(entry_config(), description="trial override round trip")

    return cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])


def _value_at(config, path: str):
    """Returns the value the resolved config holds at a dotted override path.

    Args:
        config : Resolved entry configuration.
        path   : Dotted path of the override.

    Returns:
        The value stored at that path.
    """
    value = config

    for part in path.split("."):
        value = getattr(value, part)

    return value


def _matches(resolved, planned) -> bool:
    """Returns whether a resolved config value equals the value the planner asked for.

    Enums compare by their value, because the CLI turns a planned string into the
    enum member the config declares. Sequences compare element-wise so a planned
    tuple and a resolved list agree, and dicts compare by content, which is how
    model overrides arrive.

    Args:
        resolved : Value read back off the resolved config.
        planned  : Value the planner emitted.

    Returns:
        True when the two express the same setting.
    """
    if isinstance(resolved, Enum):
        resolved = resolved.value

    if isinstance(planned, dict):
        return dict(resolved) == planned

    if isinstance(planned, (list, tuple)) and isinstance(resolved, (list, tuple)):
        return list(resolved) == list(planned)

    return resolved == planned


@pytest.mark.parametrize(("key", "entry_config", "launcher_class", "scheduler"), ENTRIES)
def test_every_planned_override_lands_on_the_resolved_config(key, entry_config, launcher_class, scheduler):
    """Every override a planner emits resolves to that exact value on the unit's entry config."""
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
