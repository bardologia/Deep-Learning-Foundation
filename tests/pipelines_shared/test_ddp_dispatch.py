"""Tests for the shared DDP dispatch across every training family.

Covers the seed-sweep entries (JEPA and the two autoencoders), the trial-sweep
entries (backbone and dual), the combinations each of them must refuse, and the
local-only guard on the cluster schedulers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.training.image_autoencoder    import ImageAeEntryConfig
from configuration.training.jepa                 import JepaEntryConfig
from configuration.training.profile_autoencoder  import ProfileAeEntryConfig
from pipelines.backbone.training                 import launcher as backbone_mod
from pipelines.backbone.training.launcher        import BackboneTrainingLauncher
from pipelines.dual.training                     import launcher as dual_mod
from pipelines.dual.training.launcher            import DualTrainingLauncher
from pipelines.image_autoencoder.training.launcher import ImageAeTrainingLauncher
from pipelines.image_autoencoder.training.pipeline import SingleTrainRunner as ImageAeRunner
from pipelines.jepa.training.launcher            import JepaTrainingLauncher
from pipelines.jepa.training.pipeline            import SingleTrainRunner as JepaRunner
from pipelines.profile_autoencoder.training.launcher import ProfileAeTrainingLauncher
from pipelines.profile_autoencoder.training.pipeline import SingleTrainRunner as ProfileAeRunner
from pipelines.shared.training                   import training_launcher as launcher_mod
from pipelines.shared.training.ddp_fanout        import DdpFanout
from pipelines.shared.training.training_launcher import SeedSweepLauncher
from tools.training.distributed                  import DistributedContext


ENTRY = Path("/entry/train.py")

SEED_SWEEP_FAMILIES = [
    (JepaEntryConfig,       JepaRunner,      "train_jepa.py"),
    (ProfileAeEntryConfig,  ProfileAeRunner, "train_profile_autoencoder.py"),
    (ImageAeEntryConfig,    ImageAeRunner,   "train_image_autoencoder.py"),
]

TRIAL_LAUNCHERS = [
    (BackboneTrainingLauncher, backbone_mod),
    (DualTrainingLauncher,     dual_mod),
]


@pytest.fixture(autouse=True)
def _inert_context():
    """Clears the cached distributed context so each launcher sees a plain process."""
    DistributedContext.reset()
    yield
    DistributedContext.reset()


class _Recorder:
    """Stands in for a fan-out or runner, recording the config it was built with.

    Attributes:
        calls : Configs handed to the recorded constructor, in call order.
        ran   : Number of `run` calls made on the recorded objects.
    """

    def __init__(self) -> None:
        """Starts with no recorded construction and no run."""
        self.calls = []
        self.ran   = 0

    def factory(self, config, *args, **kwargs):
        """Records the config and returns self so `run` lands on this recorder."""
        self.calls.append(config)
        return self

    def run(self) -> None:
        """Counts one run of the recorded object."""
        self.ran += 1


def _stubs(monkeypatch, module) -> tuple:
    """Replaces the fan-out and the in-process runner of a launcher module with recorders.

    Args:
        monkeypatch : Pytest monkeypatch fixture.
        module      : Launcher module whose `SeedSweepRunner` is stubbed.

    Returns:
        Tuple of the fan-out recorder and the in-process runner recorder.
    """
    fanout = _Recorder()
    sweep  = _Recorder()

    monkeypatch.setattr(DdpFanout, "for_runner",      fanout.factory)
    monkeypatch.setattr(module,    "SeedSweepRunner", sweep.factory)

    return fanout, sweep


@pytest.mark.parametrize("entry_config, runner, entry_script", SEED_SWEEP_FAMILIES)
def test_seed_sweep_entries_dispatch_a_single_ddp_seed(monkeypatch, tmp_path, entry_config, runner, entry_script):
    """Every seed-sweep family fans its single seed out as a rank group when ddp is set."""
    fanout, sweep = _stubs(monkeypatch, launcher_mod)

    SeedSweepLauncher(entry_config(), runner, "family training", entry_script=Path(entry_script)).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert fanout.ran == 1
    assert sweep.ran  == 0


@pytest.mark.parametrize("entry_config, runner, entry_script", SEED_SWEEP_FAMILIES)
def test_seed_sweep_entries_reject_ddp_with_several_seeds(monkeypatch, tmp_path, entry_config, runner, entry_script):
    """A multi-seed sweep already claims one GPU per seed, so ddp is refused."""
    scheduler = _Recorder()
    monkeypatch.setattr(launcher_mod, "SeedFanoutScheduler", scheduler)

    with pytest.raises(ValueError, match="ddp=True trains exactly one seed"):
        SeedSweepLauncher(entry_config(), runner, "family training", entry_script=Path(entry_script)).run(["--ddp", "true", "--seeds", "[0, 1]", "--logdir", str(tmp_path)])

    assert scheduler.ran == 0


@pytest.mark.parametrize("entry_config, runner, entry_script", SEED_SWEEP_FAMILIES)
def test_seed_sweep_entries_train_in_process_inside_a_rank(monkeypatch, tmp_path, entry_config, runner, entry_script):
    """A process that already is a rank of a running group trains its shard in place."""
    fanout, sweep = _stubs(monkeypatch, launcher_mod)

    for key, value in DistributedContext.environment(rank=1, world_size=2, master_port=29998).items():
        monkeypatch.setenv(key, value)
    DistributedContext.reset()

    SeedSweepLauncher(entry_config(), runner, "family training", entry_script=Path(entry_script)).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert sweep.ran  == 1
    assert fanout.ran == 0


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_dispatch_a_single_ddp_seed(monkeypatch, tmp_path, launcher_class, module):
    """Backbone and dual fan their single seed out as a rank group when trials stay off."""
    fanout, sweep = _stubs(monkeypatch, module)

    launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert fanout.ran == 1
    assert sweep.ran  == 0


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_train_in_process_inside_a_rank(monkeypatch, tmp_path, launcher_class, module):
    """A backbone or dual rank process trains its shard instead of fanning out again."""
    fanout, sweep = _stubs(monkeypatch, module)

    for key, value in DistributedContext.environment(rank=1, world_size=2, master_port=29997).items():
        monkeypatch.setenv(key, value)
    DistributedContext.reset()

    launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert sweep.ran  == 1
    assert fanout.ran == 0


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_reject_ddp_with_several_seeds(monkeypatch, tmp_path, launcher_class, module):
    """A multi-seed backbone or dual fan-out cannot also split one run across the pool."""
    _stubs(monkeypatch, module)

    with pytest.raises(ValueError, match="ddp=True trains exactly one seed"):
        launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[0, 1]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_reject_ddp_with_a_trial_sweep(monkeypatch, tmp_path, launcher_class, module):
    """A trial sweep fans its units out one GPU each, so it refuses ddp before planning."""
    _stubs(monkeypatch, module)

    with pytest.raises(ValueError, match="fans its units out one GPU each"):
        launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--trials_enabled", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_reject_ddp_on_a_trial_unit(monkeypatch, tmp_path, launcher_class, module):
    """A --trial unit owns a single device, so it refuses ddp instead of fanning out."""
    _stubs(monkeypatch, module)

    with pytest.raises(ValueError, match="not available for a --trial unit"):
        launcher_class(entry_script=ENTRY).run(["--trial", "--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


@pytest.mark.parametrize("launcher_class", [JepaTrainingLauncher, ProfileAeTrainingLauncher, ImageAeTrainingLauncher])
def test_cluster_seed_sweeps_refuse_ddp(launcher_class):
    """Data-parallel training is local-only, so a cluster seed sweep refuses it loudly."""
    with pytest.raises(SystemExit, match="local-only"):
        launcher_class(entry_script=ENTRY).build_scheduler(["--ddp", "true", "--run_name", "base", "--logdir", "/dss/runs/family"])


@pytest.mark.parametrize("launcher_class", [BackboneTrainingLauncher, DualTrainingLauncher])
def test_cluster_trial_sweeps_refuse_ddp(launcher_class, tmp_path):
    """The trial scheduler the cluster submitter builds refuses ddp before planning units."""
    with pytest.raises(ValueError, match="fans its units out one GPU each"):
        launcher_class(entry_script=ENTRY).build_scheduler(["--ddp", "true", "--run_name", "base", "--logdir", str(tmp_path)])
