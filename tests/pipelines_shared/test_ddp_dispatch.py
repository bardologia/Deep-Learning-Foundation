
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
    DistributedContext.reset()
    yield
    DistributedContext.reset()


class _Recorder:

    def __init__(self) -> None:
        self.calls = []
        self.ran   = 0

    def factory(self, config, *args, **kwargs):
        self.calls.append(config)
        return self

    def run(self) -> None:
        self.ran += 1


def _stubs(monkeypatch, module) -> tuple:
    fanout = _Recorder()
    sweep  = _Recorder()

    monkeypatch.setattr(DdpFanout, "for_runner",      fanout.factory)
    monkeypatch.setattr(module,    "SeedSweepRunner", sweep.factory)

    return fanout, sweep


@pytest.mark.parametrize("entry_config, runner, entry_script", SEED_SWEEP_FAMILIES)
def test_seed_sweep_entries_dispatch_a_single_ddp_seed(monkeypatch, tmp_path, entry_config, runner, entry_script):
    fanout, sweep = _stubs(monkeypatch, launcher_mod)

    SeedSweepLauncher(entry_config(), runner, "family training", entry_script=Path(entry_script)).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert fanout.ran == 1
    assert sweep.ran  == 0


@pytest.mark.parametrize("entry_config, runner, entry_script", SEED_SWEEP_FAMILIES)
def test_seed_sweep_entries_reject_ddp_with_several_seeds(monkeypatch, tmp_path, entry_config, runner, entry_script):
    scheduler = _Recorder()
    monkeypatch.setattr(launcher_mod, "SeedFanoutScheduler", scheduler)

    with pytest.raises(ValueError, match="ddp=True trains exactly one seed"):
        SeedSweepLauncher(entry_config(), runner, "family training", entry_script=Path(entry_script)).run(["--ddp", "true", "--seeds", "[0, 1]", "--logdir", str(tmp_path)])

    assert scheduler.ran == 0


@pytest.mark.parametrize("entry_config, runner, entry_script", SEED_SWEEP_FAMILIES)
def test_seed_sweep_entries_train_in_process_inside_a_rank(monkeypatch, tmp_path, entry_config, runner, entry_script):
    fanout, sweep = _stubs(monkeypatch, launcher_mod)

    for key, value in DistributedContext.environment(rank=1, world_size=2, master_port=29998).items():
        monkeypatch.setenv(key, value)
    DistributedContext.reset()

    SeedSweepLauncher(entry_config(), runner, "family training", entry_script=Path(entry_script)).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert sweep.ran  == 1
    assert fanout.ran == 0


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_dispatch_a_single_ddp_seed(monkeypatch, tmp_path, launcher_class, module):
    fanout, sweep = _stubs(monkeypatch, module)

    launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert fanout.ran == 1
    assert sweep.ran  == 0


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_train_in_process_inside_a_rank(monkeypatch, tmp_path, launcher_class, module):
    fanout, sweep = _stubs(monkeypatch, module)

    for key, value in DistributedContext.environment(rank=1, world_size=2, master_port=29997).items():
        monkeypatch.setenv(key, value)
    DistributedContext.reset()

    launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert sweep.ran  == 1
    assert fanout.ran == 0


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_reject_ddp_with_several_seeds(monkeypatch, tmp_path, launcher_class, module):
    _stubs(monkeypatch, module)

    with pytest.raises(ValueError, match="ddp=True trains exactly one seed"):
        launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[0, 1]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_reject_ddp_with_a_trial_sweep(monkeypatch, tmp_path, launcher_class, module):
    _stubs(monkeypatch, module)

    with pytest.raises(ValueError, match="fans its units out one GPU each"):
        launcher_class(entry_script=ENTRY).run(["--ddp", "true", "--trials_enabled", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


@pytest.mark.parametrize("launcher_class, module", TRIAL_LAUNCHERS)
def test_trial_entries_reject_ddp_on_a_trial_unit(monkeypatch, tmp_path, launcher_class, module):
    _stubs(monkeypatch, module)

    with pytest.raises(ValueError, match="not available for a --trial unit"):
        launcher_class(entry_script=ENTRY).run(["--trial", "--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


@pytest.mark.parametrize("launcher_class", [JepaTrainingLauncher, ProfileAeTrainingLauncher, ImageAeTrainingLauncher])
def test_cluster_seed_sweeps_refuse_ddp(launcher_class):
    with pytest.raises(SystemExit, match="local-only"):
        launcher_class(entry_script=ENTRY).build_scheduler(["--ddp", "true", "--run_name", "base", "--logdir", "/dss/runs/family"])


@pytest.mark.parametrize("launcher_class", [BackboneTrainingLauncher, DualTrainingLauncher])
def test_cluster_trial_sweeps_refuse_ddp(launcher_class, tmp_path):
    with pytest.raises(ValueError, match="fans its units out one GPU each"):
        launcher_class(entry_script=ENTRY).build_scheduler(["--ddp", "true", "--run_name", "base", "--logdir", str(tmp_path)])
