"""Tests covering the artifact bundles of datasets, parameter runs and training runs.

Covers discovering the three kinds of source directory, resolving the required,
optional and analysis path lists against one of them, gathering a training run
whole and dropping from it what the exclusion rules leave out, packing the
gathered files into size-capped zip volumes that keep the source tree, and the
batch loop that mails one message per volume.
"""

from __future__ import annotations

import base64
import json
import zipfile

import pytest

from configuration.diagnostics        import ReportMailEntryConfig
from tools.reporting.artifact_bundles import ArtifactBundle, DatasetSelector, ExcludedContent, ParamRunSelector, RunContentBundle, RunContentSelector
from tools.reporting.report_mailer    import ReportMailBatch

from tests.conftest import RecordingLogger


PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _write(path, payload):
    """Writes one file with its parents, taking text or bytes.

    Args:
        path    : Destination file path.
        payload : Text written as UTF-8, or raw bytes.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")

    return path


def _write_param_run(run_dir, analysed=True):
    """Builds one parameter-extraction run, optionally with its analysis outputs.

    Args:
        run_dir  : Directory the parameter run is written into.
        analysed : Whether the outputs of analyze_param_extraction are written too.

    Returns:
        Path of the parameter run directory.
    """
    _write(run_dir / "param_extraction_meta.json", json.dumps({"k_max": 2}))
    _write(run_dir / "parameters.npy", b"\x00" * 64)
    _write(run_dir / "fit_diagnostics.npz", b"\x00" * 32)
    _write(run_dir / "logs" / "param_extraction.log", "fitted\n")

    if analysed:
        _write(run_dir / "fit_metrics_summary.json", json.dumps({"global_summary": {}}))
        _write(run_dir / "images" / "metrics" / "mse.png", PNG_BYTES)

    return run_dir


def _write_dataset(root, name, param_runs=("params_k2",)):
    """Builds one generated dataset with its layout, metadata, logs and parameter runs.

    Args:
        root       : Dataset root the folder is created in.
        name       : Name of the dataset folder.
        param_runs : Names of the parameter runs written under params/.

    Returns:
        Path of the dataset directory.
    """
    dataset_dir = root / name

    _write(dataset_dir / "data" / "dataset.json", json.dumps({"dataset_type": "synthetic", "channel_labels": ["CH00", "CH01"]}))
    _write(dataset_dir / "data" / "cube_full.npy", b"\x00" * 4096)
    _write(dataset_dir / "data" / "primary.npy", b"\x00" * 1024)
    _write(dataset_dir / "meta" / "config_state.json", json.dumps({"crop": {"row_start": 0, "row_end": 64, "col_start": 0, "col_end": 64}}))
    _write(dataset_dir / "meta" / "environment.json", json.dumps({"python": "3.11", "torch": "2.11", "numpy": "2.4"}))
    _write(dataset_dir / "logs" / "generate_dataset.log", "scene synthesised, arrays written, layout saved\n")

    for run_name in param_runs:
        _write_param_run(dataset_dir / "params" / run_name)

    return dataset_dir


def _write_run(run_dir, inferred=True):
    """Builds one training run with its metadata, logs, checkpoints and, by default, an inference stamp.

    Args:
        run_dir  : Directory the run is written into.
        inferred : Whether the inference stamp with its report, figures, cubes and animation is written.

    Returns:
        Path of the run directory.
    """
    _write(run_dir / "meta" / "model_config.json", json.dumps({"backbone_name": "unet"}))
    _write(run_dir / "meta" / "run_summary.json", json.dumps({"model_name": "unet"}))
    _write(run_dir / "docs" / "model_doc.md", "# Model\n")
    _write(run_dir / "logs" / "training.log", "epoch 1\n")
    _write(run_dir / "tensorboard" / "events.out.tfevents.1757000000.host", b"\x00" * 128)
    _write(run_dir / "best_model.pt", b"\x00" * 4096)
    _write(run_dir / "checkpoints" / "last.pt", b"\x00" * 4096)

    if inferred:
        stamp = run_dir / "inference" / "20260101_000000"
        _write(stamp / "report.md", "# Report\n")
        _write(stamp / "metrics.json", json.dumps({"curve_rmse_gt": 0.5}))
        _write(stamp / "figures" / "map.png", PNG_BYTES)
        _write(stamp / "cubes" / "pred_curves.npy", b"\x00" * 2048)
        _write(stamp / "animations" / "walk_z.gif", b"\x00" * 6000)

    return run_dir


@pytest.fixture
def runs_dir(tmp_path):
    """Returns a runs tree holding one scored backbone run and one that was never inferred."""
    root = tmp_path / "runs"
    _write_run(root / "backbone" / "unet_a")
    _write_run(root / "backbone" / "unet_b", inferred=False)
    return root


@pytest.fixture
def datasets_dir(tmp_path):
    """Returns a dataset root holding one dataset with a parameter run and one without."""
    root = tmp_path / "datasets"
    _write_dataset(root, "ds_alpha")
    _write_dataset(root, "ds_beta", param_runs=())
    return root


@pytest.fixture
def entry_config(datasets_dir, runs_dir, tmp_path):
    """Returns an entry config pointing every artifact target at the dataset root and the runs tree."""
    return ReportMailEntryConfig(
        target       = "datasets",
        datasets_dir = datasets_dir,
        params_dir   = datasets_dir,
        runs_dir     = runs_dir,
        staging_dir  = tmp_path / "staging",
        recipient    = "victor@example.com",
        sender       = "victor@example.com",
        password_env = "TEST_REPORT_MAIL_PASSWORD",
    )


def _archived_names(archive_path):
    """Returns the sorted member names of one written archive."""
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(archive.namelist())


def test_dataset_selector_discovers_only_folders_with_a_layout(datasets_dir):
    """Verifies discovery keeps children carrying data/dataset.json and skips the rest."""
    (datasets_dir / "scratch").mkdir()

    datasets = DatasetSelector(datasets_dir, RecordingLogger()).all()

    assert [dataset.name for dataset in datasets] == ["ds_alpha", "ds_beta"]


def test_dataset_selector_raises_without_any_dataset(tmp_path):
    """Verifies a root with no generated dataset raises and names generate_dataset."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="generate_dataset"):
        DatasetSelector(tmp_path / "empty", RecordingLogger()).all()


def test_param_run_selector_skips_runs_without_the_parameter_stack(datasets_dir):
    """Verifies a run carrying only the extraction metadata is not offered."""
    _write(datasets_dir / "ds_beta" / "params" / "half_written" / "param_extraction_meta.json", "{}")

    runs = ParamRunSelector(datasets_dir, RecordingLogger()).all()

    assert [run.name for run in runs] == ["params_k2"]


def test_dataset_bundle_packs_the_layout_metadata_and_logs(entry_config, datasets_dir):
    """Verifies the archive carries the layout, metadata and logs under the label."""
    config   = entry_config.to_dataset_config(datasets_dir / "ds_alpha")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    assert [path.name for path in archives] == ["datasets__ds_alpha.zip"]
    assert archives[0].parent == entry_config.staging_dir / "datasets" / "ds_alpha"
    assert _archived_names(archives[0]) == [
        "datasets/ds_alpha/data/dataset.json",
        "datasets/ds_alpha/logs/generate_dataset.log",
        "datasets/ds_alpha/meta/config_state.json",
        "datasets/ds_alpha/meta/environment.json",
    ]


def test_dataset_bundle_leaves_the_arrays_out(entry_config, datasets_dir):
    """Verifies the arrays under data/ are not packed by the default paths."""
    config   = entry_config.to_dataset_config(datasets_dir / "ds_alpha")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    assert not any(name.endswith(".npy") for name in _archived_names(archives[0]))


def test_dataset_bundle_warns_for_a_missing_optional_path(entry_config, datasets_dir):
    """Verifies a dataset without its docs directory still packs and warns once naming the path."""
    logger   = RecordingLogger()
    config   = entry_config.to_dataset_config(datasets_dir / "ds_beta")
    archives = ArtifactBundle(config, logger).stage()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert [message for message in warnings if "'docs'" in message]
    assert "datasets/ds_beta/data/dataset.json" in _archived_names(archives[0])


def test_dataset_bundle_aborts_on_a_missing_required_path(entry_config, datasets_dir):
    """Verifies a dataset without its layout file aborts naming the missing path."""
    (datasets_dir / "ds_alpha" / "data" / "dataset.json").unlink()

    config = entry_config.to_dataset_config(datasets_dir / "ds_alpha")

    with pytest.raises(FileNotFoundError, match="data/dataset.json"):
        ArtifactBundle(config, RecordingLogger()).stage()


def test_param_bundle_packs_extraction_and_analysis_artifacts(entry_config, datasets_dir):
    """Verifies the parameter archive carries the metadata, logs, metrics and figures."""
    config   = entry_config.to_param_config(datasets_dir / "ds_alpha" / "params" / "params_k2")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    assert [path.name for path in archives] == ["ds_alpha__params_k2.zip"]
    assert _archived_names(archives[0]) == [
        "ds_alpha/params_k2/fit_metrics_summary.json",
        "ds_alpha/params_k2/images/metrics/mse.png",
        "ds_alpha/params_k2/logs/param_extraction.log",
        "ds_alpha/params_k2/param_extraction_meta.json",
    ]


def test_param_bundle_warns_and_names_the_analysis_entry(entry_config, tmp_path):
    """Verifies an unanalysed parameter run still packs and the warning names analyze_param_extraction."""
    run_dir  = _write_param_run(tmp_path / "ds_gamma" / "params" / "params_k2", analysed=False)
    logger   = RecordingLogger()
    archives = ArtifactBundle(entry_config.to_param_config(run_dir), logger).stage()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert any("analyze_param_extraction" in message for message in warnings)
    assert not any("/images/" in name for name in _archived_names(archives[0]))


def test_bundle_splits_into_numbered_volumes_over_the_cap(entry_config, datasets_dir):
    """Verifies a bundle above the cap is split and every volume names its position."""
    entry_config.max_attachment_mb = 100 / (1024 * 1024)

    config   = entry_config.to_dataset_config(datasets_dir / "ds_alpha")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    packed = [name for archive in archives for name in _archived_names(archive)]

    assert len(archives) > 1
    assert [path.name for path in archives] == [f"datasets__ds_alpha_{index:02d}of{len(archives):02d}.zip" for index in range(1, len(archives) + 1)]
    assert len(packed) == len(set(packed))
    assert sorted(packed) == [
        "datasets/ds_alpha/data/dataset.json",
        "datasets/ds_alpha/logs/generate_dataset.log",
        "datasets/ds_alpha/meta/config_state.json",
        "datasets/ds_alpha/meta/environment.json",
    ]


def test_bundle_aborts_when_one_file_alone_exceeds_the_cap(entry_config, datasets_dir):
    """Verifies an oversized single file aborts and points at max_attachment_mb."""
    entry_config.max_attachment_mb = 32 / (1024 * 1024)

    config = entry_config.to_dataset_config(datasets_dir / "ds_alpha")

    with pytest.raises(ValueError, match="max_attachment_mb"):
        ArtifactBundle(config, RecordingLogger()).stage()


def test_repacking_removes_the_volumes_of_the_previous_run(entry_config, datasets_dir):
    """Verifies a bundle that shrinks to one volume leaves no numbered part behind."""
    entry_config.max_attachment_mb = 100 / (1024 * 1024)
    ArtifactBundle(entry_config.to_dataset_config(datasets_dir / "ds_alpha"), RecordingLogger()).stage()

    entry_config.max_attachment_mb = 15.0
    archives = ArtifactBundle(entry_config.to_dataset_config(datasets_dir / "ds_alpha"), RecordingLogger()).stage()

    assert sorted(path.name for path in archives[0].parent.glob("*.zip")) == ["datasets__ds_alpha.zip"]


def test_batch_mails_one_message_per_dataset(entry_config, monkeypatch, fake_smtp):
    """Verifies the dataset target sends one message per dataset with the archive attached."""
    monkeypatch.setenv(entry_config.password_env, "app-password")

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    message     = fake_smtp.instances[0].messages[0]
    attachments = [part.get_filename() for part in message.iter_attachments()]

    assert [result["label"] for result in results] == ["datasets/ds_alpha", "datasets/ds_beta"]
    assert message["Subject"] == f"{entry_config.subject_prefix}: datasets/ds_alpha"
    assert attachments == ["datasets__ds_alpha.zip"]
    assert "Artifact bundle of dataset" in message.get_body(preferencelist=("plain",)).get_content()


def test_batch_numbers_the_subject_of_a_split_bundle(entry_config, monkeypatch, fake_smtp):
    """Verifies a split bundle is mailed as one message per volume, each numbered."""
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.max_attachment_mb = 100 / (1024 * 1024)
    entry_config.dataset_filter    = ["ds_alpha"]

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    total    = results[0]["n_messages"]
    subjects = [message["Subject"] for instance in fake_smtp.instances for message in instance.messages]

    assert total > 1
    assert subjects == [f"{entry_config.subject_prefix}: datasets/ds_alpha [{index}/{total}]" for index in range(1, total + 1)]


def test_batch_mails_the_selected_parameter_run(entry_config, monkeypatch, fake_smtp):
    """Verifies the parameter target selects by the run path relative to the root."""
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.target           = "param_extraction"
    entry_config.param_run_filter = ["ds_alpha/params/params_k2"]

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    assert [result["label"] for result in results] == ["ds_alpha/params_k2"]
    assert results[0]["subjects"] == [f"{entry_config.subject_prefix}: ds_alpha/params_k2"]


def test_batch_rejects_an_unknown_target(entry_config):
    """Verifies an unknown target aborts and lists the accepted values."""
    entry_config.target = "checkpoints"

    with pytest.raises(ValueError, match="param_extraction"):
        ReportMailBatch(entry_config, RecordingLogger())


def test_run_selector_lists_every_run_below_the_root(runs_dir):
    """Verifies the runs are found by the meta directory they hold, wherever they sit."""
    runs = RunContentSelector(runs_dir, RecordingLogger()).all()

    assert [str(run.relative_to(runs_dir)) for run in runs] == ["backbone/unet_a", "backbone/unet_b"]


def test_run_selector_announces_the_size_scan_before_the_table(runs_dir):
    """Verifies the per-run file scan is announced with the run count before the run table is logged."""
    logger = RecordingLogger()

    RunContentSelector(runs_dir, logger).all()

    assert logger.messages[0] == ("info", f"Scanning 2 run(s) under {runs_dir} for their file counts and sizes")
    assert logger.messages[1] == ("metrics_table", f"Runs at or below {runs_dir}")


def test_run_selector_takes_a_root_that_is_itself_a_run(runs_dir):
    """Verifies pointing the selector at one run offers that run instead of searching below it."""
    run_dir = runs_dir / "backbone" / "unet_a"

    runs = RunContentSelector(run_dir, RecordingLogger()).all()

    assert runs == [run_dir]


def test_run_selector_raises_without_any_run(tmp_path):
    """Verifies a root holding no run aborts naming the marker it looked for."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="meta"):
        RunContentSelector(tmp_path / "empty", RecordingLogger()).all()


def test_run_bundle_packs_everything_the_run_holds(entry_config, runs_dir):
    """Verifies the archive carries the metadata, docs, logs, events, report and figures of the run."""
    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, RecordingLogger()).stage()

    assert [path.name for path in archives] == ["backbone__unet_a.zip"]
    assert archives[0].parent == entry_config.staging_dir / "backbone" / "unet_a"
    assert _archived_names(archives[0]) == [
        "backbone/unet_a/docs/model_doc.md",
        "backbone/unet_a/inference/20260101_000000/animations/walk_z.gif",
        "backbone/unet_a/inference/20260101_000000/figures/map.png",
        "backbone/unet_a/inference/20260101_000000/metrics.json",
        "backbone/unet_a/inference/20260101_000000/report.md",
        "backbone/unet_a/logs/training.log",
        "backbone/unet_a/meta/model_config.json",
        "backbone/unet_a/meta/run_summary.json",
        "backbone/unet_a/tensorboard/events.out.tfevents.1757000000.host",
    ]


def test_run_bundle_leaves_the_checkpoints_and_cubes_out(entry_config, runs_dir):
    """Verifies the default rules drop the checkpoint, the checkpoints directory and the cube."""
    logger   = RecordingLogger()
    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, logger).stage()

    packed = _archived_names(archives[0])
    titles = [message for level, message in logger.messages if level == "metrics_table"]

    assert not [name for name in packed if name.endswith((".pt", ".npy"))]
    assert not [name for name in packed if "/checkpoints/" in name or "/cubes/" in name]
    assert any(title.startswith("Left out of") for title in titles)


def test_run_bundle_drops_a_file_above_the_per_file_cap(entry_config, runs_dir):
    """Verifies the size rule leaves the animation behind and keeps the figure of the same stamp."""
    entry_config.run_max_file_mb = 5000 / (1024 * 1024)

    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, RecordingLogger()).stage()

    packed = _archived_names(archives[0])

    assert not [name for name in packed if name.endswith(".gif")]
    assert "backbone/unet_a/inference/20260101_000000/figures/map.png" in packed


def test_run_bundle_mails_a_run_that_was_never_inferred(entry_config, runs_dir):
    """Verifies a run without an inference stamp still travels with what it recorded."""
    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_b")
    archives = RunContentBundle(config, RecordingLogger()).stage()

    assert _archived_names(archives[0]) == [
        "backbone/unet_b/docs/model_doc.md",
        "backbone/unet_b/logs/training.log",
        "backbone/unet_b/meta/model_config.json",
        "backbone/unet_b/meta/run_summary.json",
        "backbone/unet_b/tensorboard/events.out.tfevents.1757000000.host",
    ]


def test_excluded_content_names_the_rule_that_dropped_each_file(runs_dir):
    """Verifies each dropped file is recorded under the first rule that matched it."""
    run_dir  = runs_dir / "backbone" / "unet_a"
    paths    = sorted(path for path in run_dir.rglob("*") if path.is_file())
    excluded = ExcludedContent(run_dir, ["cubes"], [".pt"], 5000, RecordingLogger())

    kept  = excluded.filter(paths)
    rules = {str(relative): rule for relative, _size, rule in excluded.dropped}

    assert rules == {
        "best_model.pt"                                   : "suffix .pt",
        "checkpoints/last.pt"                             : "suffix .pt",
        "inference/20260101_000000/cubes/pred_curves.npy" : "path cubes",
        "inference/20260101_000000/animations/walk_z.gif" : "over 0.0 MB",
    }
    assert len(kept) == len(paths) - len(rules)


def test_run_bundle_aborts_on_a_missing_run_directory(entry_config, tmp_path):
    """Verifies a run path that does not exist aborts naming the directory."""
    config = entry_config.to_run_config(tmp_path / "runs" / "backbone" / "gone")

    with pytest.raises(FileNotFoundError, match="gone"):
        RunContentBundle(config, RecordingLogger()).stage()


def test_batch_mails_one_message_per_run(entry_config, monkeypatch, fake_smtp):
    """Verifies the run target mails each selected run under its path below the runs root."""
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.target     = "runs"
    entry_config.run_filter = ["backbone/unet_a"]

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    message     = fake_smtp.instances[0].messages[0]
    attachments = [part.get_filename() for part in message.iter_attachments()]

    assert [result["label"] for result in results] == ["backbone/unet_a"]
    assert message["Subject"] == f"{entry_config.subject_prefix}: backbone/unet_a"
    assert attachments == ["backbone__unet_a.zip"]
    assert "Content bundle of run" in message.get_body(preferencelist=("plain",)).get_content()
