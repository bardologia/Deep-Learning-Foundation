
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
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")

    return path


def _write_param_run(run_dir, analysed=True):
    _write(run_dir / "param_extraction_meta.json", json.dumps({"k_max": 2}))
    _write(run_dir / "parameters.npy", b"\x00" * 64)
    _write(run_dir / "fit_diagnostics.npz", b"\x00" * 32)
    _write(run_dir / "logs" / "param_extraction.log", "fitted\n")

    if analysed:
        _write(run_dir / "fit_metrics_summary.json", json.dumps({"global_summary": {}}))
        _write(run_dir / "images" / "metrics" / "mse.png", PNG_BYTES)

    return run_dir


def _write_dataset(root, name, param_runs=("params_k2",)):
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
    root = tmp_path / "runs"
    _write_run(root / "backbone" / "unet_a")
    _write_run(root / "backbone" / "unet_b", inferred=False)
    return root


@pytest.fixture
def datasets_dir(tmp_path):
    root = tmp_path / "datasets"
    _write_dataset(root, "ds_alpha")
    _write_dataset(root, "ds_beta", param_runs=())
    return root


@pytest.fixture
def entry_config(datasets_dir, runs_dir, tmp_path):
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
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(archive.namelist())


def test_dataset_selector_discovers_only_folders_with_a_layout(datasets_dir):
    (datasets_dir / "scratch").mkdir()

    datasets = DatasetSelector(datasets_dir, RecordingLogger()).all()

    assert [dataset.name for dataset in datasets] == ["ds_alpha", "ds_beta"]


def test_dataset_selector_raises_without_any_dataset(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="generate_dataset"):
        DatasetSelector(tmp_path / "empty", RecordingLogger()).all()


def test_param_run_selector_skips_runs_without_the_parameter_stack(datasets_dir):
    _write(datasets_dir / "ds_beta" / "params" / "half_written" / "param_extraction_meta.json", "{}")

    runs = ParamRunSelector(datasets_dir, RecordingLogger()).all()

    assert [run.name for run in runs] == ["params_k2"]


def test_dataset_bundle_packs_the_layout_metadata_and_logs(entry_config, datasets_dir):
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
    config   = entry_config.to_dataset_config(datasets_dir / "ds_alpha")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    assert not any(name.endswith(".npy") for name in _archived_names(archives[0]))


def test_dataset_bundle_warns_for_a_missing_optional_path(entry_config, datasets_dir):
    logger   = RecordingLogger()
    config   = entry_config.to_dataset_config(datasets_dir / "ds_beta")
    archives = ArtifactBundle(config, logger).stage()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert [message for message in warnings if "'docs'" in message]
    assert "datasets/ds_beta/data/dataset.json" in _archived_names(archives[0])


def test_dataset_bundle_aborts_on_a_missing_required_path(entry_config, datasets_dir):
    (datasets_dir / "ds_alpha" / "data" / "dataset.json").unlink()

    config = entry_config.to_dataset_config(datasets_dir / "ds_alpha")

    with pytest.raises(FileNotFoundError, match="data/dataset.json"):
        ArtifactBundle(config, RecordingLogger()).stage()


def test_param_bundle_packs_extraction_and_analysis_artifacts(entry_config, datasets_dir):
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
    run_dir  = _write_param_run(tmp_path / "ds_gamma" / "params" / "params_k2", analysed=False)
    logger   = RecordingLogger()
    archives = ArtifactBundle(entry_config.to_param_config(run_dir), logger).stage()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert any("analyze_param_extraction" in message for message in warnings)
    assert not any("/images/" in name for name in _archived_names(archives[0]))


def test_bundle_splits_into_numbered_volumes_over_the_cap(entry_config, datasets_dir):
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
    entry_config.max_attachment_mb = 32 / (1024 * 1024)

    config = entry_config.to_dataset_config(datasets_dir / "ds_alpha")

    with pytest.raises(ValueError, match="max_attachment_mb"):
        ArtifactBundle(config, RecordingLogger()).stage()


def test_repacking_removes_the_volumes_of_the_previous_run(entry_config, datasets_dir):
    entry_config.max_attachment_mb = 100 / (1024 * 1024)
    ArtifactBundle(entry_config.to_dataset_config(datasets_dir / "ds_alpha"), RecordingLogger()).stage()

    entry_config.max_attachment_mb = 15.0
    archives = ArtifactBundle(entry_config.to_dataset_config(datasets_dir / "ds_alpha"), RecordingLogger()).stage()

    assert sorted(path.name for path in archives[0].parent.glob("*.zip")) == ["datasets__ds_alpha.zip"]


def test_batch_mails_one_message_per_dataset(entry_config, monkeypatch, fake_smtp):
    monkeypatch.setenv(entry_config.password_env, "app-password")

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    message     = fake_smtp.instances[0].messages[0]
    attachments = [part.get_filename() for part in message.iter_attachments()]

    assert [result["label"] for result in results] == ["datasets/ds_alpha", "datasets/ds_beta"]
    assert message["Subject"] == f"{entry_config.subject_prefix}: datasets/ds_alpha"
    assert attachments == ["datasets__ds_alpha.zip"]
    assert "Artifact bundle of dataset" in message.get_body(preferencelist=("plain",)).get_content()


def test_batch_numbers_the_subject_of_a_split_bundle(entry_config, monkeypatch, fake_smtp):
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.max_attachment_mb = 100 / (1024 * 1024)
    entry_config.dataset_filter    = ["ds_alpha"]

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    total    = results[0]["n_messages"]
    subjects = [message["Subject"] for instance in fake_smtp.instances for message in instance.messages]

    assert total > 1
    assert subjects == [f"{entry_config.subject_prefix}: datasets/ds_alpha [{index}/{total}]" for index in range(1, total + 1)]


def test_batch_mails_the_selected_parameter_run(entry_config, monkeypatch, fake_smtp):
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.target           = "param_extraction"
    entry_config.param_run_filter = ["ds_alpha/params/params_k2"]

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    assert [result["label"] for result in results] == ["ds_alpha/params_k2"]
    assert results[0]["subjects"] == [f"{entry_config.subject_prefix}: ds_alpha/params_k2"]


def test_batch_rejects_an_unknown_target(entry_config):
    entry_config.target = "checkpoints"

    with pytest.raises(ValueError, match="param_extraction"):
        ReportMailBatch(entry_config, RecordingLogger())


def test_run_selector_lists_every_run_below_the_root(runs_dir):
    runs = RunContentSelector(runs_dir, RecordingLogger()).all()

    assert [str(run.relative_to(runs_dir)) for run in runs] == ["backbone/unet_a", "backbone/unet_b"]


def test_run_selector_announces_the_size_scan_before_the_table(runs_dir):
    logger = RecordingLogger()

    RunContentSelector(runs_dir, logger).all()

    assert logger.messages[0] == ("info", f"Scanning 2 run(s) under {runs_dir} for their file counts and sizes")
    assert logger.messages[1] == ("metrics_table", f"Runs at or below {runs_dir}")


def test_run_selector_takes_a_root_that_is_itself_a_run(runs_dir):
    run_dir = runs_dir / "backbone" / "unet_a"

    runs = RunContentSelector(run_dir, RecordingLogger()).all()

    assert runs == [run_dir]


def test_run_selector_raises_without_any_run(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="meta"):
        RunContentSelector(tmp_path / "empty", RecordingLogger()).all()


def test_run_bundle_packs_everything_the_run_holds(entry_config, runs_dir):
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
    logger   = RecordingLogger()
    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, logger).stage()

    packed = _archived_names(archives[0])
    titles = [message for level, message in logger.messages if level == "metrics_table"]

    assert not [name for name in packed if name.endswith((".pt", ".npy"))]
    assert not [name for name in packed if "/checkpoints/" in name or "/cubes/" in name]
    assert any(title.startswith("Left out of") for title in titles)


def test_run_bundle_drops_a_file_above_the_per_file_cap(entry_config, runs_dir):
    entry_config.run_max_file_mb = 5000 / (1024 * 1024)

    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, RecordingLogger()).stage()

    packed = _archived_names(archives[0])

    assert not [name for name in packed if name.endswith(".gif")]
    assert "backbone/unet_a/inference/20260101_000000/figures/map.png" in packed


def test_run_bundle_mails_a_run_that_was_never_inferred(entry_config, runs_dir):
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


def test_excluded_content_names_the_files_the_size_cap_dropped(runs_dir):
    run_dir  = runs_dir / "backbone" / "unet_a"
    paths    = sorted(path for path in run_dir.rglob("*") if path.is_file())
    logger   = RecordingLogger()
    excluded = ExcludedContent(run_dir, ["cubes"], [".pt"], 5000, logger)

    excluded.filter(paths)

    warnings = [message for level, message in logger.messages if level == "warning"]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"1 file(s) of {run_dir} left out for being over 0.0 MB, largest first: inference/20260101_000000/animations/walk_z.gif (")
    assert "best_model.pt" not in warnings[0]


def test_excluded_content_names_the_largest_oversized_files_and_counts_the_rest(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    for index in range(ExcludedContent.NAMED_LIMIT + 2):
        (source / f"file_{index:02d}.bin").write_bytes(b"x" * (2000 + index))

    logger = RecordingLogger()
    ExcludedContent(source, [], [], 1000, logger).filter(sorted(source.iterdir()))

    warning = [message for level, message in logger.messages if level == "warning"][0]
    assert warning.startswith(f"{ExcludedContent.NAMED_LIMIT + 2} file(s) of {source} left out for being over 0.0 MB, largest first: file_21.bin")
    assert warning.endswith(" and 2 more")
    assert "file_00.bin" not in warning


def test_run_bundle_aborts_on_a_missing_run_directory(entry_config, tmp_path):
    config = entry_config.to_run_config(tmp_path / "runs" / "backbone" / "gone")

    with pytest.raises(FileNotFoundError, match="gone"):
        RunContentBundle(config, RecordingLogger()).stage()


def test_batch_mails_one_message_per_run(entry_config, monkeypatch, fake_smtp):
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
