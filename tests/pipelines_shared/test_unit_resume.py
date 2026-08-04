from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

from pipelines.shared.training.training_runner import EntryConfigTrainRunner
from pipelines.shared.training.unit_resume     import UnitResume
from tools.runtime.completion                  import CompletionMarker


def mark_complete(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "test"})


def mark_inference_complete(directory: Path, split: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "inference", "split": split})


def unit(tmp_path: Path, enabled: bool = True, trainer_resume: bool = False) -> UnitResume:
    return UnitResume(tmp_path / "run", enabled=enabled, trainer_resume=trainer_resume)


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


def test_disabled_inference_check_is_inert(tmp_path):
    resume = unit(tmp_path, enabled=False)
    mark_inference_complete(resume.run_directory / "inference" / "stamp0", "val")

    assert not resume.skip_inference("val")


class _StubPipeline:
    run_label = "stub"

    launched = []

    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        _StubPipeline.launched.append(self.config.run_name)
        return "trained"


class _StubRunner(EntryConfigTrainRunner):
    pipeline_class = _StubPipeline

    def _pretrain_preflight(self) -> None:
        pass


def _entry_config(tmp_path: Path, resume: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        run_name = "unit",
        resume   = resume,
        logdir   = tmp_path,
        training = SimpleNamespace(resume=False),
    )


def test_runner_trains_when_unit_is_fresh(tmp_path):
    _StubPipeline.launched = []

    result = _StubRunner(_entry_config(tmp_path)).run()

    assert result == "trained"
    assert _StubPipeline.launched == ["unit"]


def test_runner_skips_completed_unit(tmp_path):
    _StubPipeline.launched = []
    mark_complete(tmp_path / "unit")

    result = _StubRunner(_entry_config(tmp_path)).run()

    assert result is None
    assert _StubPipeline.launched == []


def test_runner_deletes_unfinished_unit_before_training(tmp_path):
    _StubPipeline.launched = []

    unit_dir = tmp_path / "unit"
    unit_dir.mkdir(parents=True)
    (unit_dir / "best_model.pt").write_text("x")

    _StubRunner(_entry_config(tmp_path)).run()

    assert _StubPipeline.launched == ["unit"]
    assert not (unit_dir / "best_model.pt").exists()


def test_runner_resume_off_trains_in_place(tmp_path):
    _StubPipeline.launched = []
    mark_complete(tmp_path / "unit")

    result = _StubRunner(_entry_config(tmp_path, resume=False)).run()

    assert result == "trained"
    assert CompletionMarker.is_complete(tmp_path / "unit")
