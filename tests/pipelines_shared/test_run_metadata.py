"""Tests for the training run metadata that lays out a run directory, refuses an occupied one and writes its config files."""

from __future__ import annotations

import json

import pytest

from configuration.dataset.gaussian       import GaussianConfig
from configuration.training               import BackboneTrainerConfig
from pipelines.shared.config.run_metadata import RunDirectoryLeftovers, TrainingRunMetadata
from tools.monitoring.logger              import Logger
from tools.runtime.completion             import CompletionMarker


def _trainer_config() -> BackboneTrainerConfig:
    """Returns a backbone trainer config with five Gaussians over a -20 to 80 metre height axis."""
    gaussian = GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0, activity_threshold=1e-4)

    return BackboneTrainerConfig(gaussian=gaussian)


def _logger(tmp_path) -> Logger:
    """Returns a silent logger writing under tmp_path."""
    return Logger(log_dir=str(tmp_path / "logs"), name="rm_test", level="ERROR")


def test_directory_layout_created(tmp_path):
    """Constructing the metadata creates the run, tensorboard, docs, logs, and metadata directories."""
    meta = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_x", logger=_logger(tmp_path))

    try:
        assert meta.run_directory      == tmp_path / "run_x"
        assert meta.tensorboard_dir.is_dir()
        assert meta.docs_directory.is_dir()
        assert meta.logs_directory.is_dir()
        assert meta.metadata_directory.is_dir()
    finally:
        meta.close()


def test_io_logdir_wired_and_writer_kept_out_of_the_config(tmp_path):
    """The run directory is written back onto the config while the tensorboard writer stays off it."""
    tc   = _trainer_config()
    meta = TrainingRunMetadata(tc, "unet", tmp_path, run_name="run_io", logger=_logger(tmp_path))

    try:
        assert tc.io.logdir == str(meta.run_directory)
        assert meta.writer is not None
        assert not hasattr(tc.io, "writer")
    finally:
        meta.close()


def test_save_trainer_config_roundtrip(tmp_path):
    """The saved trainer config lands in docs and excludes the writer."""
    meta = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_cfg", logger=_logger(tmp_path))

    try:
        out_path = meta.save_trainer_config()
        payload  = json.loads(out_path.read_text())

        assert out_path == meta.docs_directory / "trainer_config.json"
        assert "writer" not in payload["io"]
        assert payload["gaussian"]["n_default_gaussians"] == 5
    finally:
        meta.close()


def test_save_run_summary_payload(tmp_path):
    """The run summary records the model name, channel counts, axis length, and run directory."""
    meta = TrainingRunMetadata(_trainer_config(), "swin", tmp_path, run_name="run_sum", logger=_logger(tmp_path))

    try:
        out_path = meta.save_run_summary("swin", in_channels=9, out_channels=15, x_axis_length=150)
        payload  = json.loads(out_path.read_text())

        assert payload["model_name"]    == "swin"
        assert payload["in_channels"]   == 9
        assert payload["out_channels"]  == 15
        assert payload["x_axis_length"] == 150
        assert payload["run_directory"] == str(meta.run_directory)
    finally:
        meta.close()


def test_context_manager_closes_writer(tmp_path):
    """Leaving the context closes the tensorboard writer but keeps the run directory."""
    with TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_ctx", logger=_logger(tmp_path)) as meta:
        assert meta.run_directory.is_dir()
        assert meta.writer.all_writers is not None

    assert meta.run_directory.is_dir()
    assert meta.writer.all_writers is None
    assert meta.writer.file_writer is None


def test_exit_with_an_exception_records_the_abort_in_the_owned_log(tmp_path):
    """An exception escaping the context writes an ABORTED banner with the exception into the run's own log."""
    with pytest.raises(RuntimeError):
        with TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_abort") as meta:
            raise RuntimeError("loss is non-finite at step 3")

    contents = (meta.logs_directory / "unet_metadata.log").read_text(encoding="utf-8")

    assert "[Aborted] RuntimeError: loss is non-finite at step 3" in contents
    assert ">>> ABORTED unet_metadata" in contents
    assert ">>> END unet_metadata" not in contents


def _earlier_run(tmp_path, name: str):
    """Writes the leftovers of an earlier run named `name`: a best checkpoint, an inference stamp directory and a TensorBoard event file."""
    run_directory = tmp_path / name

    (run_directory / "tensorboard").mkdir(parents=True)
    (run_directory / "inference").mkdir()
    (run_directory / "best_model.pt").write_bytes(b"")
    (run_directory / "tensorboard" / "events.out.tfevents.1.host").write_bytes(b"")

    return run_directory


def test_leftovers_are_listed_relative_to_the_run_directory(tmp_path):
    """The finder names the checkpoint, the inference directory and the event file, and sees nothing in an absent or clean directory."""
    run_directory = _earlier_run(tmp_path, "run_old")
    (run_directory / "docs").mkdir()

    assert RunDirectoryLeftovers.find(run_directory)       == ["best_model.pt", "inference", "tensorboard/events.out.tfevents.1.host"]
    assert RunDirectoryLeftovers.find(tmp_path / "absent") == []
    assert RunDirectoryLeftovers.find(tmp_path)            == []


def test_fresh_run_refuses_a_directory_holding_a_completed_run(tmp_path):
    """Without trainer resume the constructor raises on a completed run, naming the directory and every artifact, and leaves them in place."""
    run_directory = _earlier_run(tmp_path, "run_old")
    CompletionMarker.stamp(run_directory, {"stage": "training"})

    with pytest.raises(ValueError, match="already holds a completed run"):
        TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_old", logger=_logger(tmp_path))

    assert [path.name for path in (run_directory / "tensorboard").iterdir()] == ["events.out.tfevents.1.host"]
    assert (run_directory / "best_model.pt").is_file()


def test_fresh_run_deletes_an_unfinished_attempt_and_keeps_its_logs(tmp_path):
    """Without trainer resume an unfinished attempt's artifacts are deleted with a warning, its logs stay, and only the new event file remains."""
    run_directory = _earlier_run(tmp_path, "run_old")
    (run_directory / "logs").mkdir()
    (run_directory / "logs" / "worker.log").write_text("Traceback of the killed attempt\n")

    meta = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_old")
    try:
        events = [path.name for path in (run_directory / "tensorboard").iterdir()]
    finally:
        meta.close()

    assert not (run_directory / "best_model.pt").exists()
    assert not (run_directory / "inference").exists()
    assert "events.out.tfevents.1.host" not in events and len(events) == 1
    assert (run_directory / "logs" / "worker.log").read_text() == "Traceback of the killed attempt\n"
    assert "holds an unfinished earlier attempt; deleting its artifacts" in (run_directory / "logs" / "unet_metadata.log").read_text()


def test_relaunch_after_a_killed_attempt_that_left_only_its_event_file(tmp_path):
    """An attempt killed right after opening its writer leaves only an event file; the relaunch clears it instead of failing."""
    first = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_killed", logger=_logger(tmp_path))
    first.writer.flush()
    first.writer.close()

    second = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_killed", logger=_logger(tmp_path))
    try:
        events = list((second.run_directory / "tensorboard").glob("events.out.tfevents.*"))
    finally:
        second.close()

    assert len(events) == 1


def test_resumed_run_keeps_the_earlier_artifacts(tmp_path):
    """With trainer resume on the same directory is accepted and its artifacts stay in place."""
    run_directory = _earlier_run(tmp_path, "run_old")

    config                 = _trainer_config()
    config.training.resume = True

    meta = TrainingRunMetadata(config, "unet", tmp_path, run_name="run_old", logger=_logger(tmp_path))
    try:
        assert meta.run_directory == run_directory
        assert (run_directory / "best_model.pt").is_file()
    finally:
        meta.close()


def test_run_summary_records_the_patch_geometry(tmp_path):
    """The run summary carries the configured and effective patch geometry handed to it, and null when a run records none."""
    meta     = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_patch", logger=_logger(tmp_path))
    geometry = {"configured_size": [64, 32], "configured_stride": [32, 16], "size": [64, 64], "stride": [32, 32], "squared": True}

    try:
        with_patch = json.loads(meta.save_run_summary("unet", 2, 15, 170, patch_geometry=geometry).read_text())
        without    = json.loads(meta.save_run_summary("unet", 2, 15, 170).read_text())
    finally:
        meta.close()

    assert with_patch["patch_geometry"] == geometry
    assert without["patch_geometry"]    is None


def test_run_summary_records_provenance(tmp_path):
    """The run summary carries the provenance block and the dirty-diff slot beside the commit."""
    meta = TrainingRunMetadata(_trainer_config(), "unet", tmp_path, run_name="run_prov", logger=_logger(tmp_path))
    try:
        path    = meta.save_run_summary("unet", 2, 15, 170, n_gaussians=5, seed=1)
        payload = json.loads(path.read_text())
    finally:
        meta.close()

    assert payload["provenance"]["git_commit"] == meta.git_commit
    assert payload["provenance"]["torch"]
    assert bool(payload["cudnn"]) == (payload["n_devices"] > 0)
    assert payload["git_diff"] == (str(meta.docs_directory / "git_diff.patch") if meta.git_diff is not None else None)
