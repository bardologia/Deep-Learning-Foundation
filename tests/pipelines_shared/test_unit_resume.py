
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.backbone.training.launcher           import SingleTrainRunner
from pipelines.shared.training.unit_resume         import UnitResume
from tools.data.io                                 import FileIO
from tools.runtime.completion                      import CompletionMarker
from tools.runtime.config_cli                      import ConfigCli
from tools.runtime.git_state                       import GitState


@dataclass
class _TrainingStub:

    resume     : bool = False
    batch_size : int  = 8


@dataclass
class _EntryStub:

    run_name     : str           = "unit"
    resume       : bool          = True
    resume_from  : str           = ""
    logdir       : Path          = Path("")
    infer_after  : bool          = False
    infer_at_end : bool          = False
    training     : _TrainingStub = field(default_factory=_TrainingStub)


def _entry_config(tmp_path: Path, resume: bool = True, resume_from: str = "") -> _EntryStub:
    return _EntryStub(resume=resume, resume_from=resume_from, logdir=tmp_path)


def mark_complete(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "test"})


def mark_inference_complete(directory: Path, split: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "inference", "split": split})


def unit(tmp_path: Path, enabled: bool = True, trainer_resume: bool = False, config: _EntryStub | None = None) -> UnitResume:
    return UnitResume(tmp_path / "run", enabled=enabled, trainer_resume=trainer_resume, config=config or _entry_config(tmp_path))


def record(run_directory: Path, config: _EntryStub, commit: str | None) -> None:
    ConfigCli.save_resolved(config, ResolvedEntryConfigWriter.path_for(run_directory))

    if commit is not None:
        FileIO.save_json({"git_commit": commit}, ResolvedEntryConfigWriter.provenance_path_for(run_directory))


def resume_log(run_directory: Path) -> str:
    return (run_directory / "logs" / "unit_resume.log").read_text(encoding="utf-8")


def test_fresh_directory_trains(tmp_path):
    resume = unit(tmp_path)

    assert not resume.skip_training()


def test_completed_directory_skips_training(tmp_path):
    resume = unit(tmp_path)
    mark_complete(resume.run_directory)

    assert resume.skip_training()
    assert resume.run_directory.exists()


def test_unfinished_directory_is_deleted_and_retrained(tmp_path):
    resume = unit(tmp_path)
    resume.run_directory.mkdir(parents=True)
    (resume.run_directory / "best_model.pt").write_text("x")

    assert not resume.skip_training()
    assert not (resume.run_directory / "best_model.pt").exists()

    written = resume_log(resume.run_directory)

    assert "[WARNING]" in written
    assert "unfinished run deleted" in written
    assert "1 file(s) and no last.pt" in written


def test_deleting_a_trainer_state_names_its_size_and_save_time(tmp_path):
    resume = unit(tmp_path)
    resume.run_directory.mkdir(parents=True)
    (resume.run_directory / "last.pt").write_bytes(b"\0" * (3 * 1024 ** 2))
    (resume.run_directory / "logs").mkdir()
    (resume.run_directory / "logs" / "old.log").write_text("x")

    assert not resume.skip_training()

    written = resume_log(resume.run_directory)

    assert "2 file(s) including last.pt (3.0 MB, last saved 20" in written
    assert "pass training.resume" in written


def test_disabled_never_skips_or_deletes(tmp_path):
    resume = unit(tmp_path, enabled=False)
    mark_complete(resume.run_directory)
    (resume.run_directory / "best_model.pt").write_text("x")

    assert not resume.skip_training()
    assert (resume.run_directory / "best_model.pt").exists()


def test_trainer_resume_keeps_unfinished_directory(tmp_path):
    resume = unit(tmp_path, trainer_resume=True)
    resume.run_directory.mkdir(parents=True)
    (resume.run_directory / "last.pt").write_text("x")

    assert not resume.skip_training()
    assert (resume.run_directory / "last.pt").exists()


def test_trainer_resume_still_skips_completed_run(tmp_path):
    resume = unit(tmp_path, trainer_resume=True)
    mark_complete(resume.run_directory)

    assert resume.skip_training()


def test_reuse_is_a_warning_naming_the_changed_leaves_and_the_commit(tmp_path):
    resume = unit(tmp_path)
    mark_complete(resume.run_directory)
    record(resume.run_directory, _EntryStub(logdir=tmp_path, training=_TrainingStub(batch_size=16)), commit="deadbeef0000")

    assert resume.skip_training()

    written = resume_log(resume.run_directory)

    assert "[WARNING]" in written
    assert "completed training reused" in written
    assert "training.batch_size" in written and "16" in written
    assert "different configuration on 1 leaf(s)" in written
    assert "deadbeef0000" in written and GitState.commit() in written


def test_reuse_of_a_matching_run_reports_the_match(tmp_path):
    resume   = unit(tmp_path)
    recorded = _EntryStub(logdir=tmp_path / "elsewhere", run_name="other", resume_from="/pasted", resume=False)
    mark_complete(resume.run_directory)
    record(resume.run_directory, recorded, commit=GitState.commit())

    assert resume.skip_training()

    written = resume_log(resume.run_directory)

    assert "recorded the configuration of this launch" in written
    assert "Reused run vs this launch" not in written
    assert "was produced at commit" not in written


def test_reuse_without_a_recording_warns_that_nothing_can_be_compared(tmp_path):
    resume = unit(tmp_path)
    mark_complete(resume.run_directory)

    assert resume.skip_training()

    written = resume_log(resume.run_directory)

    assert "recorded no docs/resolved_entry_config.json" in written
    assert "unknown (no docs/provenance.json recorded)" in written


def test_disabled_resume_warns_that_a_completed_run_refuses(tmp_path):
    resume = unit(tmp_path, enabled=False)
    mark_complete(resume.run_directory)

    assert not resume.skip_training()

    written = resume_log(resume.run_directory)

    assert "[WARNING]" in written
    assert "holds a completed run, which is never overwritten in place; training refuses to start" in written


def test_disabled_resume_warns_that_an_unfinished_attempt_is_deleted(tmp_path):
    resume = unit(tmp_path, enabled=False)
    resume.run_directory.mkdir(parents=True)

    assert not resume.skip_training()

    written = resume_log(resume.run_directory)

    assert "[WARNING]" in written
    assert "the unfinished earlier attempt" in written
    assert "is deleted when training starts" in written


def test_no_inference_directory_runs_inference(tmp_path):
    resume = unit(tmp_path)
    resume.run_directory.mkdir(parents=True)

    assert not resume.skip_inference("val")


def test_complete_inference_is_reused(tmp_path):
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "val")

    assert resume.skip_inference("val")


def test_inference_for_another_split_is_not_reused(tmp_path):
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "test")

    assert not resume.skip_inference("val")
    assert (resume.run_directory / "inference" / "stamp0").exists()


def test_unfinished_inference_is_purged_and_rerun(tmp_path):
    resume = unit(tmp_path)

    unfinished = resume.run_directory / "inference" / "stamp0"
    unfinished.mkdir(parents=True)
    (unfinished / "metrics.json").write_text("{}")

    assert not resume.skip_inference("val")
    assert not unfinished.exists()


def test_mixed_inference_purges_unfinished_and_reuses_complete(tmp_path):
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "done", "val")

    unfinished = resume.run_directory / "inference" / "partial"
    unfinished.mkdir(parents=True)
    (unfinished / "metrics.json").write_text("{}")

    assert resume.skip_inference("val")
    assert not unfinished.exists()
    assert (resume.run_directory / "inference" / "done").exists()


def test_every_resume_message_survives_on_disk(tmp_path):
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "done", "val")
    (resume.run_directory / "inference" / "partial").mkdir(parents=True)

    assert resume.skip_inference("val")

    written = resume_log(resume.run_directory)

    assert "[WARNING]" in written
    assert "unfinished inference 'partial' deleted (0 file(s)" in written
    assert "existing 'val' inference reused"                   in written


def test_disabled_inference_check_is_inert(tmp_path):
    resume = unit(tmp_path, enabled=False)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "val")

    assert not resume.skip_inference("val")


class _StubPipeline:

    launched = []

    def __init__(self, config) -> None:
        self.config = config

    def run(self, resolved_entry_config=None):
        _StubPipeline.launched.append(self.config.run_name)
        return "trained"


class _StubRunner(SingleTrainRunner):

    label = "stub"

    def _pipeline(self, logdir) -> _StubPipeline:
        return _StubPipeline(self.config)


def test_runner_trains_when_unit_is_fresh(tmp_path):
    _StubPipeline.launched = []

    result = _StubRunner(_entry_config(tmp_path)).run()

    assert result == "trained"
    assert _StubPipeline.launched == ["stub_unit"]


def test_runner_skips_completed_unit(tmp_path):
    _StubPipeline.launched = []
    mark_complete(tmp_path / "stub_unit")

    result = _StubRunner(_entry_config(tmp_path)).run()

    assert result is None
    assert _StubPipeline.launched == []


def test_runner_deletes_unfinished_unit_before_training(tmp_path):
    _StubPipeline.launched = []

    unit_dir = tmp_path / "stub_unit"
    unit_dir.mkdir(parents=True)
    (unit_dir / "best_model.pt").write_text("x")

    _StubRunner(_entry_config(tmp_path)).run()

    assert _StubPipeline.launched == ["stub_unit"]
    assert not (unit_dir / "best_model.pt").exists()


def test_runner_resume_off_trains_in_place(tmp_path):
    _StubPipeline.launched = []
    mark_complete(tmp_path / "stub_unit")

    result = _StubRunner(_entry_config(tmp_path, resume=False)).run()

    assert result == "trained"
    assert CompletionMarker.is_complete(tmp_path / "stub_unit")


def test_resume_from_pins_the_pasted_directory(tmp_path):
    _StubPipeline.launched = []

    target = tmp_path / "elsewhere" / "resunet-set_pred-K_3_myrun"
    target.mkdir(parents=True)

    config = _entry_config(tmp_path, resume=False, resume_from=str(target))
    runner = _StubRunner(config)
    runner.run()

    assert runner.run_directory     == target
    assert config.run_name          == "resunet-set_pred-K_3_myrun"
    assert config.logdir            == target.parent
    assert config.resume            is True
    assert config.training.resume   is False
    assert _StubPipeline.launched   == ["resunet-set_pred-K_3_myrun"]


def test_resume_from_creates_an_absent_directory_fresh(tmp_path):
    _StubPipeline.launched = []

    target = tmp_path / "sweep_root" / "seed4"
    target.parent.mkdir(parents=True)

    config = _entry_config(tmp_path, resume_from=str(target))
    runner = _StubRunner(config)
    runner.run()

    assert runner.run_directory   == target
    assert config.logdir          == target.parent
    assert config.run_name        == "seed4"
    assert config.training.resume is False
    assert _StubPipeline.launched == ["seed4"]


def test_resume_from_never_deletes_the_unfinished_target(tmp_path):
    _StubPipeline.launched = []

    target = tmp_path / "run_stub_old"
    target.mkdir(parents=True)
    (target / "last.pt").write_text("x")

    config = _entry_config(tmp_path, resume_from=str(target))
    _StubRunner(config).run()

    assert (target / "last.pt").exists()
    assert config.training.resume is True
    assert _StubPipeline.launched == ["run_stub_old"]


def test_resume_from_skips_a_completed_target(tmp_path):
    _StubPipeline.launched = []

    target = tmp_path / "run_stub_done"
    mark_complete(target)

    result = _StubRunner(_entry_config(tmp_path, resume_from=str(target))).run()

    assert result is None
    assert _StubPipeline.launched == []
