"""Tests for the DDP fan-out that launches one training rank per local GPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from configuration.training              import BackboneEntryConfig
from pipelines.shared.training.ddp_fanout import DdpFanout
from tools.runtime.completion             import CompletionMarker
from tools.training.distributed           import DistributedContext


def _config(tmp_path, **overrides) -> BackboneEntryConfig:
    """Returns an unrolled entry config with a tmp logdir and two GPUs."""
    config        = BackboneEntryConfig()
    config.logdir = tmp_path / "runs"
    config.gpus   = [0, 1]
    config.seeds  = [3]
    config.seed   = 3

    for key, value in overrides.items():
        setattr(config, key, value)

    return config


class _Runner:
    """Runner stub composing run names exactly like the unrolled entry."""

    def __init__(self, config):
        """Stores the entry config."""
        self.config = config

    def _resolve_run_name(self) -> str:
        """Returns the explicit run name, mirroring TrainingRunMetadata.resolve_name."""
        return self.config.run_name


class _LabellingRunner(_Runner):
    """Runner stub prefixing a descriptive label, as every non-unrolled entry does."""

    LABEL = "resunet_set_pred"

    def _resolve_run_name(self) -> str:
        """Returns the label joined to the configured run name."""
        return f"{self.LABEL}_{self.config.run_name}"


def test_validate_rejects_fewer_than_two_distinct_gpus(tmp_path):
    """One device, or a repeated device, cannot form a rank group."""
    with pytest.raises(ValueError, match="at least two distinct devices"):
        DdpFanout.for_runner(_config(tmp_path, gpus=[0]), {}, Path("train.py"), _Runner)

    with pytest.raises(ValueError, match="at least two distinct devices"):
        DdpFanout.for_runner(_config(tmp_path, gpus=[1, 1]), {}, Path("train.py"), _Runner)


def test_validate_rejects_an_indivisible_batch(tmp_path):
    """A global batch with a remainder across the ranks is rejected."""
    config = _config(tmp_path, gpus=[0, 1, 2])

    with pytest.raises(ValueError, match="does not divide across 3 DDP ranks"):
        DdpFanout.for_runner(config, {}, Path("train.py"), _Runner)


def test_validate_rejects_enabled_pretrain_probes(tmp_path):
    """Rank-local probe outcomes could diverge, so enabled probes are refused."""
    config = _config(tmp_path)
    config.pretrain.find_lr = True

    with pytest.raises(ValueError, match="find_lr"):
        DdpFanout.for_runner(config, {}, Path("train.py"), _Runner)


def test_for_runner_composes_a_fresh_run_name(tmp_path):
    """Without resume_from the fan-out names the run through the runner's scheme."""
    fanout = DdpFanout.for_runner(_config(tmp_path, run_name="gamma_ddp"), {}, Path("train.py"), _Runner)

    assert fanout.run_name == "gamma_ddp"
    assert fanout.seed     == 3
    assert fanout.run_dir  == tmp_path / "runs" / "gamma_ddp"


def test_for_runner_resumes_an_existing_run_directory(tmp_path):
    """resume_from repoints the logdir and forces both resume flags."""
    target = tmp_path / "elsewhere" / "old_run"
    (target / "ddp" / "rank1").mkdir(parents=True)

    config = _config(tmp_path, resume_from=str(target))
    fanout = DdpFanout.for_runner(config, {}, Path("train.py"), _Runner)

    assert fanout.run_name        == "old_run"
    assert config.logdir          == target.parent
    assert config.resume          is True
    assert config.training.resume is True


def test_resume_rejects_a_run_with_fewer_mirrors_than_ranks(tmp_path):
    """Resuming with more GPUs than the run trained on refuses instead of desynchronising."""
    target = tmp_path / "elsewhere" / "old_run"
    (target / "ddp" / "rank1").mkdir(parents=True)

    config = _config(tmp_path, resume_from=str(target), gpus=[0, 1, 2])
    config.training.batch_size = 6

    with pytest.raises(SystemExit, match="different rank group"):
        DdpFanout.for_runner(config, {}, Path("train.py"), _Runner)


def test_resume_rejects_a_run_with_more_mirrors_than_ranks(tmp_path):
    """Resuming with fewer GPUs than the run trained on refuses instead of dropping a trained rank."""
    target = tmp_path / "elsewhere" / "old_run"
    (target / "ddp" / "rank1").mkdir(parents=True)
    (target / "ddp" / "rank2").mkdir(parents=True)

    with pytest.raises(SystemExit, match="different rank group"):
        DdpFanout.for_runner(_config(tmp_path, resume_from=str(target)), {}, Path("train.py"), _Runner)


def test_resume_rejects_a_run_without_ddp_mirrors(tmp_path):
    """A run that never trained data-parallel cannot be resumed as a rank group."""
    target = tmp_path / "elsewhere" / "old_run"
    target.mkdir(parents=True)

    with pytest.raises(SystemExit, match="different rank group"):
        DdpFanout.for_runner(_config(tmp_path, resume_from=str(target)), {}, Path("train.py"), _Runner)


def test_for_runner_rejects_a_missing_resume_directory(tmp_path):
    """resume_from pointing at nothing raises instead of silently starting fresh."""
    with pytest.raises(FileNotFoundError, match="not an existing run directory"):
        DdpFanout.for_runner(_config(tmp_path, resume_from=str(tmp_path / "missing")), {}, Path("train.py"), _Runner)


def test_rank_overrides_isolate_the_secondary_ranks(tmp_path):
    """Rank 0 owns the canonical run directory; other ranks land in its ddp subtree."""
    fanout = DdpFanout.for_runner(_config(tmp_path, run_name="gamma_ddp"), {"training.epochs": 3, "gpus": [0, 1]}, Path("train.py"), _Runner)

    zero = fanout._rank_overrides(0)
    one  = fanout._rank_overrides(1)

    assert zero["run_name"] == "gamma_ddp"
    assert zero["logdir"]   == str(tmp_path / "runs")
    assert one["run_name"]  == "rank1"
    assert one["logdir"]    == str(tmp_path / "runs" / "gamma_ddp" / "ddp")

    for overrides in (zero, one):
        assert overrides["seed"]            == 3
        assert overrides["seeds"]           == (3,)
        assert overrides["training.epochs"] == 3
        assert "gpus" not in overrides
        assert "ddp"  not in overrides


def test_rank_overrides_carry_the_bare_name_a_labelling_entry_composes(tmp_path):
    """An entry that prefixes a label gets the bare name back, so it re-composes the predicted directory."""
    fanout = DdpFanout.for_runner(_config(tmp_path, run_name="ddp_run"), {}, Path("train.py"), _LabellingRunner)

    assert fanout.run_name              == "resunet_set_pred_ddp_run"
    assert fanout.run_dir               == tmp_path / "runs" / "resunet_set_pred_ddp_run"
    assert fanout._rank_overrides(0)["run_name"] == "ddp_run"
    assert fanout._rank_overrides(1)["run_name"] == "rank1"
    assert fanout._mirror_name(1)                == "resunet_set_pred_rank1"


def test_resume_recovers_the_bare_name_of_a_labelling_entry(tmp_path):
    """Resuming a labelled run directory strips the label back off for the ranks."""
    target = tmp_path / "elsewhere" / "resunet_set_pred_old_run"
    (target / "ddp" / "resunet_set_pred_rank1").mkdir(parents=True)

    fanout = DdpFanout.for_runner(_config(tmp_path, resume_from=str(target)), {}, Path("train.py"), _LabellingRunner)

    assert fanout.run_name                       == "resunet_set_pred_old_run"
    assert fanout._rank_overrides(0)["run_name"] == "old_run"


def test_resume_rejects_a_directory_from_another_naming_scheme(tmp_path):
    """A resumed directory whose label disagrees with the current settings raises instead of renaming the run."""
    target = tmp_path / "elsewhere" / "other_model_old_run"
    target.mkdir(parents=True)

    with pytest.raises(SystemExit, match="does not match"):
        DdpFanout.for_runner(_config(tmp_path, resume_from=str(target)), {}, Path("train.py"), _LabellingRunner)


def test_environment_stamps_the_group_coordinates():
    """The child environment carries rank, world size and rendezvous port."""
    environment = DistributedContext.environment(rank=1, world_size=2, master_port=29777)

    assert environment[DistributedContext.RANK_VAR]  == "1"
    assert environment[DistributedContext.WORLD_VAR] == "2"
    assert environment[DistributedContext.PORT_VAR]  == "29777"


def test_launch_restores_the_inherited_cuda_mask(tmp_path, monkeypatch):
    """A rank's environment carries the pre-pin mask, not the parent's single-device pin."""
    monkeypatch.setenv("TOMOSAR_INHERITED_CUDA_MASK", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    captured = {}

    def _popen(command, env=None, **kwargs):
        captured["env"] = env
        return object()

    import pipelines.shared.training.ddp_fanout as module
    monkeypatch.setattr(module.subprocess, "Popen", _popen)

    fanout = DdpFanout.for_runner(_config(tmp_path, run_name="mask_run"), {}, Path("train.py"), _Runner)
    fanout._launch(0, gpu=1, port=29777, handles=[])

    assert "CUDA_VISIBLE_DEVICES" not in captured["env"]
    assert "TOMOSAR_INHERITED_CUDA_MASK" not in captured["env"]
    assert captured["env"][DistributedContext.RANK_VAR] == "0"


_STUB_ENTRY = """
import os, sys, time
from pathlib import Path

rank    = int(os.environ["TOMOSAR_DDP_RANK"])
mode    = Path(sys.argv[0]).with_name("mode.txt").read_text().strip()
run_dir = Path(sys.argv[sys.argv.index("--logdir") + 1]) / sys.argv[sys.argv.index("--run_name") + 1]

if rank == 0:
    if mode == "fail":
        sys.exit(2)
    if mode == "skip":
        sys.exit(0)
    (run_dir / "docs").mkdir(parents=True, exist_ok=True)
    (run_dir / "docs" / "run_directory_settled.marker").write_text("settled")

if mode == "rank_fail":
    if rank == 0:
        time.sleep(120)
    sys.exit(2)

run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / f"rank{rank}.done").write_text("done")
"""


def _stub_fanout(tmp_path, mode: str, monkeypatch) -> DdpFanout:
    """Returns a fast-polling fan-out whose entry script is a stub with the given mode."""
    entry = tmp_path / "entry.py"
    entry.write_text(_STUB_ENTRY)
    (tmp_path / "mode.txt").write_text(mode)

    monkeypatch.setattr(DdpFanout, "POLL_S", 0.05)

    return DdpFanout.for_runner(_config(tmp_path, run_name="stub_run"), {}, entry, _Runner)


def test_run_sequences_rank_zero_before_the_others(tmp_path, monkeypatch):
    """Rank zero settles the run directory, then every rank finishes its work."""
    fanout = _stub_fanout(tmp_path, "train", monkeypatch)
    fanout.run()

    assert (fanout.run_dir / "rank0.done").is_file()
    assert (fanout.ddp_dir / "rank1" / "rank1.done").is_file()
    assert (fanout.log_dir / "rank1.log").is_file()


def test_run_skips_when_rank_zero_reuses_a_complete_run(tmp_path, monkeypatch):
    """A rank zero that exits cleanly before the sentinel launches no other rank."""
    fanout = _stub_fanout(tmp_path, "skip", monkeypatch)
    fanout.run()

    assert not fanout.ddp_dir.exists()


def test_run_propagates_an_early_rank_zero_failure(tmp_path, monkeypatch):
    """A rank zero failing before the sentinel aborts the fan-out."""
    fanout = _stub_fanout(tmp_path, "fail", monkeypatch)

    with pytest.raises(SystemExit, match="rank 0 failed with exit code 2"):
        fanout.run()

    assert not fanout.ddp_dir.exists()


def test_run_terminates_the_group_when_a_secondary_rank_fails(tmp_path, monkeypatch):
    """A secondary rank exiting non-zero mid-run tears the still-training ranks down."""
    fanout = _stub_fanout(tmp_path, "rank_fail", monkeypatch)

    started = time.monotonic()
    with pytest.raises(SystemExit, match="rank 1 failed with exit code 2"):
        fanout.run()

    assert time.monotonic() - started < 60.0
    assert not (fanout.run_dir / "rank0.done").is_file()


def test_run_publishes_a_progress_unit_per_secondary_rank(tmp_path, monkeypatch):
    """The fan-out writes the per-unit progress registry the console's sub-run selector reads."""
    fanout = _stub_fanout(tmp_path, "train", monkeypatch)
    fanout.run()

    payload = json.loads((fanout.run_dir / "gpu_pool_progress.json").read_text(encoding="utf-8"))
    units   = payload["units"]

    assert [unit["name"] for unit in units] == ["rank1"]
    assert units[0]["status"]               == "done"
    assert units[0]["returncode"]           == 0
    assert units[0]["gpu"]                  == 1
    assert units[0]["log"]                  == str(fanout.log_dir / "rank1.log")
    assert payload["total"]                 == 1
    assert payload["done"]                  == 1


def test_progress_marks_a_failed_rank(tmp_path, monkeypatch):
    """A secondary rank exiting non-zero lands in the registry as failed with its exit code."""
    fanout = _stub_fanout(tmp_path, "rank_fail", monkeypatch)

    with pytest.raises(SystemExit, match="rank 1 failed"):
        fanout.run()

    payload = json.loads((fanout.run_dir / "gpu_pool_progress.json").read_text(encoding="utf-8"))

    assert payload["units"][0]["status"]     == "failed"
    assert payload["units"][0]["returncode"] == 2
    assert payload["failed"]                 == 1


def test_progress_follows_a_configured_pool_path(tmp_path, monkeypatch):
    """With gpus_file set, the registry lands beside that pool path, where the console looks."""
    entry = tmp_path / "entry.py"
    entry.write_text(_STUB_ENTRY)
    (tmp_path / "mode.txt").write_text("train")
    monkeypatch.setattr(DdpFanout, "POLL_S", 0.05)

    pool   = tmp_path / "pools" / "job42.json"
    fanout = DdpFanout.for_runner(_config(tmp_path, run_name="stub_run", gpus_file=str(pool)), {}, entry, _Runner)
    fanout.run()

    payload = json.loads((tmp_path / "pools" / "job42_progress.json").read_text(encoding="utf-8"))

    assert [unit["name"] for unit in payload["units"]] == ["rank1"]
    assert payload["units"][0]["status"]               == "done"


def test_skipped_run_writes_no_progress_file(tmp_path, monkeypatch):
    """A rank zero that reuses a complete run publishes no registry, so the console shows only the pipeline."""
    fanout = _stub_fanout(tmp_path, "skip", monkeypatch)
    fanout.run()

    assert not (fanout.run_dir / "gpu_pool_progress.json").exists()


def test_run_clears_a_completed_mirror_so_the_rank_rejoins(tmp_path, monkeypatch):
    """A mirror left complete would skip training and stall the rendezvous, so its marker goes."""
    fanout = _stub_fanout(tmp_path, "train", monkeypatch)
    mirror = fanout.ddp_dir / "rank1"
    mirror.mkdir(parents=True)
    CompletionMarker.stamp(mirror, {"status": "complete"})

    fanout.run()

    assert not CompletionMarker.is_complete(mirror)
    assert (mirror / "rank1.done").is_file()
