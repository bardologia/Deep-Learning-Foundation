"""Tests for the shared trial sweep scheduler: naming, resume-aware units and the staged inference pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.training.backbone                    import BackboneEntryConfig
from configuration.training.dual                        import DualEntryConfig
from configuration.training.jepa                        import JepaEntryConfig
from configuration.training.profile_autoencoder         import ProfileAeEntryConfig
from pipelines.backbone.training.launcher               import BackboneTrainingLauncher, TrainScheduler
from pipelines.dual.training.launcher                   import DualTrainScheduler
from pipelines.jepa.training.launcher                   import JepaTrainingLauncher, JepaTrialScheduler
from pipelines.profile_autoencoder.training.launcher    import ProfileAeTrainingLauncher, ProfileAeTrialScheduler
from tools.runtime.completion                           import CompletionMarker


JEPA_ENTRY    = Path("main/training/train_jepa.py")
PROFILE_ENTRY = Path("main/training/train_profile_autoencoder.py")


class FakeStage:
    """Records the jobs a sweep submits instead of running them on the GPU pool.

    Attributes:
        statuses  : Job status per run name, defaulting to DONE.
        pool_file : Path reported as the GPU pool file in the configuration table.
        submitted : One list of job names per queue submission, in submission order.
        written   : Result payloads per written file name.
    """

    def __init__(self, statuses: dict | None = None) -> None:
        """Stores the per-run statuses the fake queue reports back."""
        self.statuses  = statuses or {}
        self.pool_file = Path("gpus.json")
        self.submitted = []
        self.written   = {}

    def _run_queue(self, jobs: list) -> list[dict]:
        """Records the submitted job names and returns their configured statuses."""
        names = [job.name for job in jobs]
        self.submitted.append(names)

        return [{"name": name, "status": self.statuses.get(name, "DONE"), "duration_s": 60.0} for name in names]

    def _order_results(self, results: list[dict], names: list[str]) -> list[dict]:
        """Returns the results unchanged, already in submission order."""
        return results

    def _write_results(self, results: list[dict], path: Path) -> None:
        """Records the results under the file name they would be written to."""
        self.written[Path(path).name] = results

    def _log_failures(self, failed: list[dict], name_key: str = "name") -> None:
        """Ignores the failure report, which the fake queue does not render."""


def _scheduler(logdir: Path, launcher_class=JepaTrainingLauncher, entry: Path = JEPA_ENTRY, mode: str = "ema", **overrides):
    """Returns a trial scheduler built from an entry launcher and CLI overrides.

    Args:
        logdir          : Log directory the sweep roots its runs in.
        launcher_class  : Training launcher whose trial scheduler is built.
        entry           : Entry script the units re-invoke.
        mode            : Trial mode the sweep runs.
        **overrides     : Extra CLI flags appended as --key value pairs.

    Returns:
        The trial scheduler built from the assembled argv.
    """
    argv = ["--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]", "--logdir", str(logdir)]

    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]

    return launcher_class(entry_script=entry).build_scheduler(argv)


def _complete_run(root: Path, name: str, inference: bool = False, split: str = "test") -> Path:
    """Creates a completed run directory, optionally carrying a completed inference stamp."""
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(run, {})

    if inference:
        stamp = run / "inference" / "20260101_000000"
        stamp.mkdir(parents=True)
        CompletionMarker.stamp(stamp, {"split": split})

    return run


def test_sweep_name_qualifies_the_mode_with_its_entry(tmp_path):
    """The same trial mode launched from two entries resolves to two sweep names."""
    jepa    = _scheduler(tmp_path)
    profile = _scheduler(tmp_path, ProfileAeTrainingLauncher, PROFILE_ENTRY)

    assert jepa.sweep_name    == "sweep_jepa_ema"
    assert profile.sweep_name == "sweep_profile_autoencoder_ema"
    assert jepa.sweep_name != profile.sweep_name


def test_sweep_name_still_names_the_mode(tmp_path):
    """Two modes of one entry keep resolving to two sweep names."""
    assert _scheduler(tmp_path, mode="amp").sweep_name == "sweep_jepa_amp"


def test_run_directories_stay_under_the_mode_subdirectory(tmp_path):
    """Qualifying the sweep name leaves the on-disk run tree at logdir/mode."""
    assert _scheduler(tmp_path).runs_root == tmp_path / "ema"


def test_planned_units_keep_the_units_a_resume_reuses(tmp_path):
    """A resumed sweep drops completed units from units() but keeps them in planned_units()."""
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    _complete_run(root, names[0])
    (root / names[1]).mkdir(parents=True)

    scheduler = _scheduler(tmp_path, resume_from=root)

    assert [name for name, _ in scheduler.planned_units()] == names
    assert [name for name, _ in scheduler.units()]         == names[1:]


def test_resumed_sweep_infers_over_the_reused_units_missing_inference(tmp_path):
    """A fully resumed sweep trains nothing and still infers over the units without inference output."""
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    _complete_run(root, names[0], inference=True)
    _complete_run(root, names[1])

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_at_end="true")
    scheduler._stage = FakeStage()
    reports          = []

    scheduler._write_mode_report = lambda: reports.append(True)

    scheduler.run()

    assert scheduler.units()          == []
    assert scheduler._stage.submitted == [[names[1]]]
    assert reports                    == [True]
    assert "already trained" in (scheduler.log_dir / "train_scheduler.log").read_text()


def test_resumed_sweep_without_pending_inference_launches_nothing(tmp_path):
    """A fully resumed sweep whose units all carry inference output submits no job at all."""
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, name, inference=True)

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_at_end="true")
    scheduler._stage = FakeStage()

    scheduler.run()

    assert scheduler._stage.submitted == []


def test_inference_is_pending_when_the_stamp_records_another_split(tmp_path):
    """An inference stamp scored on another split does not count as the configured split's output."""
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, name, inference=True, split="val")

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_at_end="true")
    scheduler._stage = FakeStage()

    scheduler.run()

    assert scheduler._stage.submitted == [names]


def test_inference_covers_freshly_trained_units_and_skips_failed_ones(tmp_path):
    """A fresh sweep infers over the units that trained and warns off the ones that failed."""
    names            = [name for name, _ in _scheduler(tmp_path).planned_units()]
    scheduler        = _scheduler(tmp_path, infer_at_end="true")
    scheduler._stage = FakeStage(statuses={names[1]: "FAILED"})

    with pytest.raises(SystemExit, match="training trials failed"):
        scheduler.run()

    assert scheduler._stage.submitted == [names, [names[0]]]


def test_run_directories_carrying_a_label_prefix_count_as_inferred(tmp_path):
    """A completed run stored under its composed label prefix is matched by the inference check."""
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, f"resunet-conv_{name}", inference=True)

    scheduler = _scheduler(tmp_path, resume_from=root, infer_at_end="true")

    assert all(not scheduler._inference_pending(name) for name in names)


def test_trial_sweeps_stage_no_prepare_or_compare_job(tmp_path):
    """A trial sweep declares neither a prepare nor a compare stage to the cluster submitter."""
    scheduler = _scheduler(tmp_path)

    assert scheduler.setup_argv()    is None
    assert scheduler.finalize_argv() is None


def test_trial_argv_pins_the_unit_and_drops_the_scheduler_fields(tmp_path):
    """The unit command carries the run name and sweep root but none of the scheduler-owned fields."""
    scheduler       = _scheduler(tmp_path, gpus="[0, 1]", poll_interval_s="1.0", **{"training.epochs": "2"})
    run_name, extra = scheduler.units()[0]
    argv            = scheduler.trial_argv(run_name, extra)

    assert argv[argv.index("--run_name") + 1]        == run_name
    assert argv[argv.index("--logdir") + 1]          == str(scheduler.runs_root)
    assert argv[argv.index("--training.epochs") + 1] == "2"

    assert "--trials_enabled"    not in argv
    assert "--trials_mode"       not in argv
    assert "--ema_trials.decays" not in argv
    assert "--gpus"              not in argv
    assert "--poll_interval_s"   not in argv


def test_seed_expanded_units_override_the_forwarded_seed_list(tmp_path):
    """The per-unit seed wins over the seed list the scheduler forwards to every unit."""
    scheduler       = _scheduler(tmp_path, seeds="[3, 7]")
    run_name, extra = scheduler.units()[0]
    argv            = scheduler.trial_argv(run_name, extra)

    assert run_name.endswith("/seed3")
    assert argv[argv.index("--seed") + 1]  == "3"
    assert argv[argv.index("--seeds") + 1] == "[3]"


@pytest.mark.parametrize(("scheduler_class", "config_class"), (
    (TrainScheduler,          BackboneEntryConfig),
    (DualTrainScheduler,      DualEntryConfig),
    (JepaTrialScheduler,      JepaEntryConfig),
    (ProfileAeTrialScheduler, ProfileAeEntryConfig),
))
def test_scheduler_fields_name_real_entry_config_fields(scheduler_class, config_class):
    """Every field a scheduler claims exists on its entry config, so nothing is swallowed by a typo."""
    config  = config_class()
    unknown = [field for field in scheduler_class.SCHEDULER_FIELDS if not hasattr(config, field)]

    assert unknown == []


@pytest.mark.parametrize(("scheduler_class", "config_class"), (
    (TrainScheduler,          BackboneEntryConfig),
    (DualTrainScheduler,      DualEntryConfig),
    (JepaTrialScheduler,      JepaEntryConfig),
    (ProfileAeTrialScheduler, ProfileAeEntryConfig),
))
def test_every_trial_mode_claims_its_planner_settings(scheduler_class, config_class):
    """The trials config field each mode reads is claimed by the scheduler and never forwarded."""
    config = config_class()

    for mode in scheduler_class.MODE_SUBDIRS:
        field = f"{mode}_trials"

        if hasattr(config, field):
            assert field in scheduler_class.SCHEDULER_FIELDS


def test_backbone_and_jepa_sweeps_do_not_share_a_mode_directory(tmp_path):
    """The same mode launched from the backbone and the JEPA entry yields two cluster sweep names."""
    backbone = _scheduler(tmp_path, BackboneTrainingLauncher, Path("main/training/train_backbone.py"))
    jepa     = _scheduler(tmp_path)

    assert backbone.sweep_name == "sweep_backbone_ema"
    assert jepa.sweep_name     == "sweep_jepa_ema"
