
from __future__ import annotations

import base64
import json
import zipfile

import pytest

from configuration.diagnostics        import ReportMailEntryConfig
from tools.reporting.artifact_bundles import ArtifactBundle, DatasetSelector, ExcludedContent, RunContentBundle, RunContentSelector
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


def _write_dataset(root, name):
    dataset_dir = root / name
    layout      = {"name": name, "input_channels": ["x0", "x1", "x2", "x3"], "target_channels": ["pointwise", "neighbourhood"], "rows": 64, "cols": 64, "splits": {"train": [[0, 40, 0, 64]], "val": [[40, 52, 0, 64]], "test": [[52, 64, 0, 64]]}, "generator": {"seed": 0}}

    _write(dataset_dir / "dataset.json", json.dumps(layout, indent=4))
    _write(dataset_dir / "inputs.npy", b"\x00" * 4096)
    _write(dataset_dir / "targets.npy", b"\x00" * 1024)
    _write(dataset_dir / "logs" / "generate_dataset.log", "fields drawn, targets computed, dataset written, layout saved last\n")

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
        _write(stamp / "metrics.json", json.dumps({"rmse_mean": 0.5}))
        _write(stamp / "figures" / "map.png", PNG_BYTES)
        _write(stamp / "maps" / "predictions.npy", b"\x00" * 2048)
        _write(stamp / "figures" / "pointwise" / "error.gif", b"\x00" * 6000)

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
    _write_dataset(root, "ds_beta")
    return root


@pytest.fixture
def entry_config(datasets_dir, runs_dir, tmp_path):
    return ReportMailEntryConfig(
        target       = "datasets",
        datasets_dir = datasets_dir,
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


def test_dataset_bundle_packs_the_layout_metadata_and_logs(entry_config, datasets_dir):
    config   = entry_config.to_dataset_config(datasets_dir / "ds_alpha")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    assert [path.name for path in archives] == ["datasets__ds_alpha.zip"]
    assert archives[0].parent == entry_config.staging_dir / "datasets" / "ds_alpha"
    assert _archived_names(archives[0]) == [
        "datasets/ds_alpha/dataset.json",
        "datasets/ds_alpha/logs/generate_dataset.log",
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
    assert "datasets/ds_beta/dataset.json" in _archived_names(archives[0])


def test_bundle_splits_into_numbered_volumes_over_the_cap(entry_config, datasets_dir):
    entry_config.max_attachment_mb = 100 / (1024 * 1024)

    config   = entry_config.to_dataset_config(datasets_dir / "ds_alpha")
    archives = ArtifactBundle(config, RecordingLogger()).stage()

    packed = [name for archive in archives for name in _archived_names(archive)]

    assert len(archives) > 1
    assert [path.name for path in archives] == [f"datasets__ds_alpha_{index:02d}of{len(archives):02d}.zip" for index in range(1, len(archives) + 1)]
    assert len(packed) == len(set(packed))
    assert sorted(packed) == [
        "datasets/ds_alpha/dataset.json",
        "datasets/ds_alpha/logs/generate_dataset.log",
    ]


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


def test_run_bundle_packs_everything_the_run_holds(entry_config, runs_dir):
    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, RecordingLogger()).stage()

    assert [path.name for path in archives] == ["backbone__unet_a.zip"]
    assert archives[0].parent == entry_config.staging_dir / "backbone" / "unet_a"
    assert _archived_names(archives[0]) == [
        "backbone/unet_a/docs/model_doc.md",
        "backbone/unet_a/inference/20260101_000000/figures/map.png",
        "backbone/unet_a/inference/20260101_000000/figures/pointwise/error.gif",
        "backbone/unet_a/inference/20260101_000000/metrics.json",
        "backbone/unet_a/inference/20260101_000000/report.md",
        "backbone/unet_a/logs/training.log",
        "backbone/unet_a/meta/model_config.json",
        "backbone/unet_a/meta/run_summary.json",
        "backbone/unet_a/tensorboard/events.out.tfevents.1757000000.host",
    ]


def test_run_bundle_leaves_the_checkpoints_and_maps_out(entry_config, runs_dir):
    logger   = RecordingLogger()
    config   = entry_config.to_run_config(runs_dir / "backbone" / "unet_a")
    archives = RunContentBundle(config, logger).stage()

    packed = _archived_names(archives[0])
    titles = [message for level, message in logger.messages if level == "metrics_table"]

    assert not [name for name in packed if name.endswith((".pt", ".npy"))]
    assert not [name for name in packed if "/checkpoints/" in name or "/maps/" in name]
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
    excluded = ExcludedContent(run_dir, ["maps"], [".pt"], 5000, RecordingLogger())

    kept  = excluded.filter(paths)
    rules = {str(relative): rule for relative, _size, rule in excluded.dropped}

    assert rules == {
        "best_model.pt"                                   : "suffix .pt",
        "checkpoints/last.pt"                             : "suffix .pt",
        "inference/20260101_000000/maps/predictions.npy"  : "path maps",
        "inference/20260101_000000/figures/pointwise/error.gif" : "over 0.0 MB",
    }
    assert len(kept) == len(paths) - len(rules)


def test_excluded_content_names_the_files_the_size_cap_dropped(runs_dir):
    run_dir  = runs_dir / "backbone" / "unet_a"
    paths    = sorted(path for path in run_dir.rglob("*") if path.is_file())
    logger   = RecordingLogger()
    excluded = ExcludedContent(run_dir, ["maps"], [".pt"], 5000, logger)

    excluded.filter(paths)

    warnings = [message for level, message in logger.messages if level == "warning"]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"1 file(s) of {run_dir} left out for being over 0.0 MB, largest first: inference/20260101_000000/figures/pointwise/error.gif (")
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
