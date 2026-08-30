"""Tests covering the mailing of experiment report bundles.

Covers discovering experiment folders, gathering trial seed-aggregate reports,
comparison reports and configuration files into a staging directory, assembling
the email with attachments, and the batch loop over every experiment.
"""

from __future__ import annotations

import base64
import json

import pytest

from configuration.diagnostics     import ReportMailEntryConfig
from tools.reporting.report_mailer import ExperimentReportBundle, ExperimentReportMailBatch, ExperimentSelector, ReportMailer

from tests.conftest import RecordingLogger


PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _write_report(run_dir, tag):
    """Writes a report.md plus one PNG figure under the run's inference tag directory.

    Args:
        run_dir : Run directory that receives the inference/<tag> subtree.
        tag     : Inference stamp directory name.

    Returns:
        Path of the written report.md.
    """
    output_dir  = run_dir / "inference" / tag
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)

    (figures_dir / "map.png").write_bytes(PNG_BYTES)
    (output_dir / "report.md").write_text("# Report\n\n![map](figures/map.png)\n", encoding="utf-8")
    return output_dir / "report.md"


def _write_experiment(root, name):
    """Builds one complete experiment folder with two trials, a comparison and configs.

    Args:
        root : Directory the experiment folder is created in.
        name : Name of the experiment folder.

    Returns:
        Path of the experiment directory.
    """
    experiment_dir = root / name

    seeded = experiment_dir / "trial_a"
    _write_report(seeded / "seed0", "20260101_000000")
    _write_report(seeded / "seed1", "20260101_000000")
    _write_report(seeded, "seed_comparison")

    resolved = seeded / "seed0" / "docs"
    resolved.mkdir(parents=True)
    (resolved / "resolved_entry_config.json").write_text(json.dumps({"backbone_name": "resunet"}), encoding="utf-8")

    _write_report(experiment_dir / "trial_b", "20260102_000000")

    comparison = experiment_dir / "_comparison" / "20260103_000000"
    comparison.mkdir(parents=True)
    (comparison / "overview.md").write_text("# Overview\n", encoding="utf-8")
    (comparison / "metrics_comparison.md").write_text("# Metrics\n", encoding="utf-8")

    (experiment_dir / "train_scheduler_results.json").write_text(json.dumps({"units": 2}), encoding="utf-8")
    return experiment_dir


@pytest.fixture
def experiments_dir(tmp_path):
    """Returns an experiments root holding two complete experiment folders."""
    root = tmp_path / "experiments"
    _write_experiment(root, "exp_alpha")
    _write_experiment(root, "exp_beta")
    return root


@pytest.fixture
def entry_config(experiments_dir, tmp_path):
    """Returns an entry config pointing at the experiments root and a staging directory."""
    return ReportMailEntryConfig(
        experiments_dir = experiments_dir,
        staging_dir     = tmp_path / "staging",
        recipient       = "victor@example.com",
        sender          = "victor@example.com",
        password_env    = "TEST_REPORT_MAIL_PASSWORD",
    )


class FakeSmtp:
    """Stand-in for smtplib.SMTP_SSL recording logins and sent messages.

    Attributes:
        instances : Every constructed fake, in creation order.
    """

    instances = []

    def __init__(self, host, port):
        """Records the endpoint and registers the instance."""
        self.host     = host
        self.port     = port
        self.logins   = []
        self.messages = []
        FakeSmtp.instances.append(self)

    def __enter__(self):
        """Returns the fake as its own context."""
        return self

    def __exit__(self, *exc_info):
        """Leaves the context without suppressing exceptions."""
        return False

    def login(self, user, password):
        """Records one login attempt."""
        self.logins.append((user, password))

    def send_message(self, message):
        """Records one submitted message."""
        self.messages.append(message)


@pytest.fixture
def fake_smtp(monkeypatch):
    """Replaces SMTP_SSL in the mailer module with the recording fake."""
    FakeSmtp.instances = []
    monkeypatch.setattr("tools.reporting.report_mailer.smtplib.SMTP_SSL", FakeSmtp)
    return FakeSmtp


def test_selector_discovers_only_experiments_with_a_comparison(experiments_dir, tmp_path):
    """Verifies discovery keeps children with a comparison directory and skips the rest."""
    (experiments_dir / "logs").mkdir()
    (experiments_dir / "stray_run").mkdir()

    selector    = ExperimentSelector(experiments_dir, "_comparison", RecordingLogger())
    experiments = selector.all()

    assert [experiment.name for experiment in experiments] == ["exp_alpha", "exp_beta"]


def test_selector_raises_without_any_experiment(tmp_path):
    """Verifies an empty experiments root raises FileNotFoundError."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="_comparison"):
        ExperimentSelector(tmp_path / "empty", "_comparison", RecordingLogger()).all()


def test_bundle_stages_trial_comparison_and_config_files(entry_config, experiments_dir):
    """Verifies the bundle stages one report per trial, the comparison reports and both configs."""
    config = entry_config.to_config(experiments_dir / "exp_alpha")
    staged = ExperimentReportBundle(config, RecordingLogger()).stage()

    assert [path.name for path in staged] == [
        "trial_a.md",
        "trial_b.md",
        "comparison_overview.md",
        "comparison_metrics_comparison.md",
        "train_scheduler_results.json",
        "resolved_entry_config.json",
    ]
    assert all(path.is_file() for path in staged)
    assert staged[0].parent == entry_config.staging_dir / "experiments" / "exp_alpha"


def test_bundle_rewrites_image_links_to_absolute_paths(entry_config, experiments_dir):
    """Verifies a staged trial report points at the original figure by absolute path."""
    config = entry_config.to_config(experiments_dir / "exp_alpha")
    staged = ExperimentReportBundle(config, RecordingLogger()).stage()

    original = (experiments_dir / "exp_alpha" / "trial_b" / "inference" / "20260102_000000" / "figures" / "map.png").resolve()

    assert original.as_posix() in staged[1].read_text(encoding="utf-8")


def test_bundle_requires_the_seed_aggregate_report(entry_config, experiments_dir):
    """Verifies a seeded trial without its aggregated report raises and names compare_seeds."""
    aggregate = experiments_dir / "exp_alpha" / "trial_a" / "inference"
    for path in sorted(aggregate.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    aggregate.rmdir()

    config = entry_config.to_config(experiments_dir / "exp_alpha")

    with pytest.raises(FileNotFoundError, match="compare_seeds"):
        ExperimentReportBundle(config, RecordingLogger()).stage()


def test_bundle_requires_every_comparison_report(entry_config, experiments_dir):
    """Verifies a missing configured comparison report raises FileNotFoundError."""
    (experiments_dir / "exp_alpha" / "_comparison" / "20260103_000000" / "metrics_comparison.md").unlink()

    config = entry_config.to_config(experiments_dir / "exp_alpha")

    with pytest.raises(FileNotFoundError, match="metrics_comparison.md"):
        ExperimentReportBundle(config, RecordingLogger()).stage()


def test_bundle_warns_when_config_files_are_absent(entry_config, experiments_dir):
    """Verifies missing configuration files only warn and the reports still stage."""
    (experiments_dir / "exp_alpha" / "train_scheduler_results.json").unlink()
    (experiments_dir / "exp_alpha" / "trial_a" / "seed0" / "docs" / "resolved_entry_config.json").unlink()

    logger = RecordingLogger()
    config = entry_config.to_config(experiments_dir / "exp_alpha")
    staged = ExperimentReportBundle(config, logger).stage()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert [path.name for path in staged] == ["trial_a.md", "trial_b.md", "comparison_overview.md", "comparison_metrics_comparison.md"]
    assert len(warnings) == 2


def test_mailer_raises_without_the_password_variable(entry_config, monkeypatch, tmp_path):
    """Verifies sending without the password environment variable raises EnvironmentError."""
    monkeypatch.delenv(entry_config.password_env, raising=False)

    attachment = tmp_path / "report.md"
    attachment.write_text("# Report\n", encoding="utf-8")

    with pytest.raises(EnvironmentError, match=entry_config.password_env):
        ReportMailer(entry_config, RecordingLogger()).send("subject", "body", [attachment])


def test_mailer_dry_run_needs_no_password_and_sends_nothing(entry_config, monkeypatch, fake_smtp, tmp_path):
    """Verifies a dry run neither reads the password nor opens an SMTP connection."""
    monkeypatch.delenv(entry_config.password_env, raising=False)
    entry_config.dry_run = True

    attachment = tmp_path / "report.md"
    attachment.write_text("# Report\n", encoding="utf-8")

    ReportMailer(entry_config, RecordingLogger()).send("subject", "body", [attachment])

    assert fake_smtp.instances == []


def test_batch_sends_one_message_per_experiment(entry_config, monkeypatch, fake_smtp):
    """Verifies the batch mails every experiment with its staged files attached."""
    monkeypatch.setenv(entry_config.password_env, "app-password")

    results = ExperimentReportMailBatch(entry_config, RecordingLogger()).run()

    assert len(results) == 2
    assert len(fake_smtp.instances) == 2

    message     = fake_smtp.instances[0].messages[0]
    attachments = [part.get_filename() for part in message.iter_attachments()]

    assert message["To"] == "victor@example.com"
    assert message["Subject"] == f"{entry_config.subject_prefix}: experiments/exp_alpha"
    assert attachments == ["trial_a.md", "trial_b.md", "comparison_overview.md", "comparison_metrics_comparison.md", "train_scheduler_results.json", "resolved_entry_config.json"]
    assert fake_smtp.instances[0].logins == [("victor@example.com", "app-password")]


def test_batch_respects_the_experiment_filter(entry_config, monkeypatch, fake_smtp):
    """Verifies the filter restricts the batch to the named experiment."""
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.experiment_filter = ["exp_beta"]

    results = ExperimentReportMailBatch(entry_config, RecordingLogger()).run()

    assert len(results) == 1
    assert results[0]["subject"].endswith("exp_beta")


def test_same_named_sweeps_under_different_parents_do_not_collide(entry_config, tmp_path):
    """Two 'folds' sweeps of different cross-validation blocks get distinct labels and staging trees."""
    first  = entry_config.to_config(tmp_path / "ws" / "cv1" / "folds")
    second = entry_config.to_config(tmp_path / "ws" / "cv2" / "folds")

    assert entry_config.experiment_label(tmp_path / "ws" / "cv1" / "folds") == "cv1/folds"
    assert first.staging_dir  == entry_config.staging_dir / "cv1" / "folds"
    assert second.staging_dir == entry_config.staging_dir / "cv2" / "folds"
