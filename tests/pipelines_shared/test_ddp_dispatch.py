
from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.backbone.training          import launcher as backbone_mod
from pipelines.backbone.training.launcher import BackboneTrainingLauncher
from pipelines.shared.training.ddp_fanout import DdpFanout
from tools.training.distributed           import DistributedContext


ENTRY = Path("/entry/train.py")


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


def _stubs(monkeypatch) -> tuple:
    fanout = _Recorder()
    sweep  = _Recorder()

    monkeypatch.setattr(DdpFanout,    "for_runner",      fanout.factory)
    monkeypatch.setattr(backbone_mod, "SeedSweepRunner", sweep.factory)

    return fanout, sweep


def test_trial_entries_dispatch_a_single_ddp_seed(monkeypatch, tmp_path):
    fanout, sweep = _stubs(monkeypatch)

    BackboneTrainingLauncher(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert fanout.ran == 1
    assert sweep.ran  == 0


def test_trial_entries_train_in_process_inside_a_rank(monkeypatch, tmp_path):
    fanout, sweep = _stubs(monkeypatch)

    for key, value in DistributedContext.environment(rank=1, world_size=2, master_port=29997).items():
        monkeypatch.setenv(key, value)
    DistributedContext.reset()

    BackboneTrainingLauncher(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])

    assert sweep.ran  == 1
    assert fanout.ran == 0


def test_trial_entries_reject_ddp_with_several_seeds(monkeypatch, tmp_path):
    _stubs(monkeypatch)

    with pytest.raises(ValueError, match="ddp=True trains exactly one seed"):
        BackboneTrainingLauncher(entry_script=ENTRY).run(["--ddp", "true", "--seeds", "[0, 1]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


def test_trial_entries_reject_ddp_with_a_trial_sweep(monkeypatch, tmp_path):
    _stubs(monkeypatch)

    with pytest.raises(ValueError, match="fans its units out one GPU each"):
        BackboneTrainingLauncher(entry_script=ENTRY).run(["--ddp", "true", "--trials_enabled", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


def test_trial_entries_reject_ddp_on_a_trial_unit(monkeypatch, tmp_path):
    _stubs(monkeypatch)

    with pytest.raises(ValueError, match="not available for a --trial unit"):
        BackboneTrainingLauncher(entry_script=ENTRY).run(["--trial", "--ddp", "true", "--seeds", "[3]", "--gpus", "[0, 1]", "--logdir", str(tmp_path)])


def test_cluster_trial_sweeps_refuse_ddp(tmp_path):
    with pytest.raises(ValueError, match="fans its units out one GPU each"):
        BackboneTrainingLauncher(entry_script=ENTRY).build_scheduler(["--ddp", "true", "--run_name", "base", "--logdir", str(tmp_path)])
