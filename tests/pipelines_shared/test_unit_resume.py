"""Tests for per-unit resume: skipping completed runs, purging unfinished ones, and reusing inference."""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pipelines.shared.training.training_runner import EntryConfigTrainRunner
from pipelines.shared.training.unit_resume     import UnitResume
from tools.runtime.completion                  import CompletionMarker


def mark_complete(directory: Path) -> None:
    """Stamps a training completion marker on the directory, creating it if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "test"})


def mark_inference_complete(directory: Path, split: str) -> None:
    """Stamps an inference completion marker for the split on the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "inference", "split": split})


def unit(tmp_path: Path, enabled: bool = True, trainer_resume: bool = False) -> UnitResume:
    """Returns a UnitResume over tmp_path/run with the given enable and trainer-resume flags."""
    return UnitResume(tmp_path / "run", enabled=enabled, trainer_resume=trainer_resume)


def test_fresh_directory_trains(tmp_path):
    """A run directory that does not exist is trained."""
    resume = unit(tmp_path)

    assert not resume.skip_training()


def test_completed_directory_skips_training(tmp_path):
    """A run carrying a completion marker is skipped and left on disk."""
    resume = unit(tmp_path)
    mark_complete(resume.run_directory)

    assert resume.skip_training()
    assert resume.run_directory.exists()


def test_unfinished_directory_is_deleted_and_retrained(tmp_path):
    """An unfinished run directory is deleted before retraining."""
    resume = unit(tmp_path)
    resume.run_directory.mkdir(parents=True)
    (resume.run_directory / "best_model.pt").write_text("x")

    assert not resume.skip_training()
    assert not (resume.run_directory / "best_model.pt").exists()


def test_disabled_never_skips_or_deletes(tmp_path):
    """With resume disabled a completed run is retrained in place and its files survive."""
    resume = unit(tmp_path, enabled=False)
    mark_complete(resume.run_directory)
    (resume.run_directory / "best_model.pt").write_text("x")

    assert not resume.skip_training()
    assert (resume.run_directory / "best_model.pt").exists()


def test_trainer_resume_keeps_unfinished_directory(tmp_path):
    """Trainer-level resume retrains an unfinished run without deleting its checkpoint."""
    resume = unit(tmp_path, trainer_resume=True)
    resume.run_directory.mkdir(parents=True)
    (resume.run_directory / "last.pt").write_text("x")

    assert not resume.skip_training()
    assert (resume.run_directory / "last.pt").exists()


def test_trainer_resume_still_skips_completed_run(tmp_path):
    """Trainer-level resume still skips a completed run."""
    resume = unit(tmp_path, trainer_resume=True)
    mark_complete(resume.run_directory)

    assert resume.skip_training()


def test_no_inference_directory_runs_inference(tmp_path):
    """A run without an inference directory runs inference."""
    resume = unit(tmp_path)
    resume.run_directory.mkdir(parents=True)

    assert not resume.skip_inference("val")


def test_complete_inference_is_reused(tmp_path):
    """A completed inference for the split is reused."""
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "val")

    assert resume.skip_inference("val")


def test_inference_for_another_split_is_not_reused(tmp_path):
    """A completed inference for a different split is kept but not reused."""
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "test")

    assert not resume.skip_inference("val")
    assert (resume.run_directory / "inference" / "stamp0").exists()


def test_unfinished_inference_is_purged_and_rerun(tmp_path):
    """An unfinished inference directory is deleted and inference reruns."""
    resume = unit(tmp_path)

    unfinished = resume.run_directory / "inference" / "stamp0"
    unfinished.mkdir(parents=True)
    (unfinished / "metrics.json").write_text("{}")

    assert not resume.skip_inference("val")
    assert not unfinished.exists()


def test_mixed_inference_purges_unfinished_and_reuses_complete(tmp_path):
    """Unfinished inference stamps are purged while the completed one is reused."""
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "done", "val")

    unfinished = resume.run_directory / "inference" / "partial"
    unfinished.mkdir(parents=True)
    (unfinished / "metrics.json").write_text("{}")

    assert resume.skip_inference("val")
    assert not unfinished.exists()
    assert (resume.run_directory / "inference" / "done").exists()


def test_every_resume_message_survives_on_disk(tmp_path):
    """Both the purge and the reuse decisions are written to the unit resume log."""
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "done", "val")
    (resume.run_directory / "inference" / "partial").mkdir(parents=True)

    assert resume.skip_inference("val")

    written = (resume.run_directory / "logs" / "unit_resume.log").read_text(encoding="utf-8")

    assert "unfinished inference 'partial' deleted" in written
    assert "existing 'val' inference reused"        in written


def test_disabled_inference_check_is_inert(tmp_path):
    """With resume disabled a completed inference is not reused."""
    resume = unit(tmp_path, enabled=False)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "val")

    assert not resume.skip_inference("val")


def test_subdir_scoped_inference_is_reused(tmp_path):
    """Inference reuse is scoped to the requested subdirectory and split."""
    resume = unit(tmp_path)
    mark_inference_complete(resume.run_directory / "inference" / "profile_ae" / "stamp0", "val")

    assert resume.skip_inference("val", subdir="profile_ae")
    assert not resume.skip_inference("test", subdir="profile_ae")


def test_subdir_scoped_unfinished_inference_is_purged(tmp_path):
    """An unfinished inference inside a subdirectory is deleted and inference reruns."""
    resume     = unit(tmp_path)
    unfinished = resume.run_directory / "inference" / "profile_ae" / "partial"
    unfinished.mkdir(parents=True)

    assert not resume.skip_inference("val", subdir="profile_ae")
    assert not unfinished.exists()


class _StubPipeline:
    """Training pipeline stub recording the run names it was launched with.

    Attributes:
        run_label : Label the runner uses when composing directory names.
        launched  : Run names of every launch, shared across instances.
        config    : Entry config the pipeline was built from.
    """
    run_label = "stub"

    launched = []

    def __init__(self, config) -> None:
        """Stores the entry config."""
        self.config = config

    def run(self):
        """Records the run name and returns the training result sentinel."""
        _StubPipeline.launched.append(self.config.run_name)
        return "trained"


class _StubRunner(EntryConfigTrainRunner):
    """Entry-config train runner wired to the stub pipeline with a probing preflight stub.

    The stub preflight writes the batch size a real preflight would have probed,
    so a resolved config saved before it would be visibly stale.
    """
    pipeline_class = _StubPipeline

    def _pretrain_preflight(self) -> None:
        """Stands in for the preflight by writing its probed batch size into the config."""
        self.config.training.batch_size = 128


@dataclass
class _TrainingStub:
    """Training settings stub with resume disabled and a preflight-tuned batch size."""

    resume     : bool = False
    batch_size : int  = 8


@dataclass
class _EntryStub:
    """Entry config stub carrying the fields the train runner touches."""

    run_name    : str           = "unit"
    resume      : bool          = True
    resume_from : str           = ""
    logdir      : Path          = Path("")
    training    : _TrainingStub = field(default_factory=_TrainingStub)


def _entry_config(tmp_path: Path, resume: bool = True, resume_from: str = "") -> _EntryStub:
    """Returns an entry config stub named unit with the given resume settings."""
    return _EntryStub(resume=resume, resume_from=resume_from, logdir=tmp_path)


def test_runner_trains_when_unit_is_fresh(tmp_path):
    """A fresh unit is trained and returns the pipeline result."""
    _StubPipeline.launched = []

    result = _StubRunner(_entry_config(tmp_path)).run()

    assert result == "trained"
    assert _StubPipeline.launched == ["unit"]
    assert (tmp_path / "unit" / "docs" / "run_directory_settled.marker").is_file()
    assert (tmp_path / "unit" / "docs" / "resolved_entry_config.json").is_file()


def test_runner_skips_completed_unit(tmp_path):
    """A completed unit is skipped and the pipeline never runs."""
    _StubPipeline.launched = []
    mark_complete(tmp_path / "unit")

    result = _StubRunner(_entry_config(tmp_path)).run()

    assert result is None
    assert _StubPipeline.launched == []


def test_runner_deletes_unfinished_unit_before_training(tmp_path):
    """An unfinished unit directory is purged before the pipeline runs."""
    _StubPipeline.launched = []

    unit_dir = tmp_path / "unit"
    unit_dir.mkdir(parents=True)
    (unit_dir / "best_model.pt").write_text("x")

    _StubRunner(_entry_config(tmp_path)).run()

    assert _StubPipeline.launched == ["unit"]
    assert not (unit_dir / "best_model.pt").exists()


def test_runner_resume_off_trains_in_place(tmp_path):
    """With resume off a completed unit is retrained and keeps its marker."""
    _StubPipeline.launched = []
    mark_complete(tmp_path / "unit")

    result = _StubRunner(_entry_config(tmp_path, resume=False)).run()

    assert result == "trained"
    assert CompletionMarker.is_complete(tmp_path / "unit")


def test_resume_from_pins_the_pasted_directory(tmp_path):
    """A pasted resume_from directory becomes the run directory, name, and log parent, with the unit resume forced on."""
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
    """A resume_from path that does not exist yet is trained into fresh, at exactly that path."""
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
    """An unfinished resume_from target holding a trainer state keeps it and continues from it in place."""
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
    """A completed resume_from target is skipped and the pipeline never runs."""
    _StubPipeline.launched = []

    target = tmp_path / "run_stub_done"
    mark_complete(target)

    result = _StubRunner(_entry_config(tmp_path, resume_from=str(target))).run()

    assert result is None
    assert _StubPipeline.launched == []


def test_resume_from_rejects_a_path_without_a_parent_directory(tmp_path):
    """A resume_from path whose parent does not exist either raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="no parent directory"):
        _StubRunner(_entry_config(tmp_path, resume_from=str(tmp_path / "nope" / "unit"))).run()


def test_resolved_config_records_the_post_preflight_values(tmp_path):
    """The saved entry config carries what the preflight tuned, not the raw defaults.

    The preflight rewrites training.batch_size, num_workers, prefetch_factor and
    lr_multiplier, so a config written before it would describe a run that never
    happened and would relaunch under different settings when copied.
    """
    _StubPipeline.launched = []

    _StubRunner(_entry_config(tmp_path)).run()

    saved = json.loads((tmp_path / "unit" / "docs" / "resolved_entry_config.json").read_text(encoding="utf-8"))
    assert saved["training.batch_size"] == 128


def test_settling_marker_is_written_before_the_preflight_runs(tmp_path):
    """The DDP settling marker exists by the time the preflight starts.

    The other ranks are released on that marker, so it must be the first file in
    a freshly settled run directory — long before the resolved entry config,
    which only lands once the preflight has finished.
    """
    _StubPipeline.launched = []

    seen = {}

    class _OrderRunner(_StubRunner):
        """Runner recording which files exist at the moment the preflight runs."""

        def _pretrain_preflight(self) -> None:
            """Records the run directory contents, then applies the stub preflight."""
            seen["marker"] = (self.run_directory / "docs" / "run_directory_settled.marker").is_file()
            seen["config"] = (self.run_directory / "docs" / "resolved_entry_config.json").is_file()
            super()._pretrain_preflight()

    _OrderRunner(_entry_config(tmp_path)).run()

    assert seen == {"marker": True, "config": False}
