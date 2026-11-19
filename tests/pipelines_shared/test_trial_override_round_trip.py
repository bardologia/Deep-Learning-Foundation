
from __future__ import annotations

from enum    import Enum
from pathlib import Path

from configuration.training               import BackboneEntryConfig
from pipelines.backbone.training.launcher import BackboneTrainingLauncher, TrainScheduler
from tools.runtime.config_cli             import ConfigCli


DATASET_MODES = ("secondary", "input", "patch", "reach", "ratio", "ablation")


def _resolve(overrides: dict):
    cli = ConfigCli(BackboneEntryConfig(), description="trial override round trip")

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


def test_every_planned_override_lands_on_the_resolved_config():
    launcher = BackboneTrainingLauncher(entry_script=Path("main/training/train_backbone.py"))
    checked  = 0

    for mode in TrainScheduler.MODE_SUBDIRS:
        if mode in DATASET_MODES:
            continue

        argv  = ["--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]"]
        plans = launcher.build_scheduler(argv).planner().plan()

        assert plans, f"{mode} planned no unit"

        for run_name, overrides in plans:
            config = _resolve(overrides)

            for path, planned in overrides.items():
                resolved = _value_at(config, path)
                assert _matches(resolved, planned), f"{mode}/{run_name}: {path} resolved to {resolved!r}, planner asked for {planned!r}"
                checked += 1

    assert checked > 0
