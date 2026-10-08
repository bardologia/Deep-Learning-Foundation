"""Tests for the markdown inference reports of the image and profile autoencoder families.

Covers the shared report base, the per-family title and metric sections, the scored
checkpoint and one-based epochs in the summary table, and that the figures section appears
only when figures were produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipelines.shared.inference.report              import InferenceReportBase
from pipelines.image_autoencoder.inference.report   import ImageAeReport
from pipelines.profile_autoencoder.inference.report import ProfileAeReport


def _region():
    """Returns a split region stub reporting a (start, end) tuple."""
    return SimpleNamespace(as_tuple=lambda: (0, 64))


def _metrics():
    """Returns a minimal metric mapping with reconstruction errors and the mean embedding norm."""
    return {
        "mse_mean"            : 0.5,
        "mae_mean"            : 0.25,
        "embedding_norm_mean" : 1.5,
    }


def _config(checkpoint_name="best_model.pt"):
    """Returns an inference-config stub naming the scored checkpoint and the profile curve sampling."""
    return SimpleNamespace(checkpoint_name=checkpoint_name, pixel_subsample=0.5, keep_empty_frac=0.05, seed=7)


def _image_run():
    """Returns an image autoencoder run stub with the architecture, split and checkpoint fields the report reads."""
    return SimpleNamespace(
        ae_name                     = "conv_ae",
        embedding_dim               = 32,
        in_channels                 = 4,
        patch_size                  = 64,
        split_name                  = "test",
        split_region                = _region(),
        checkpoint_meta             = {"epoch": 5, "best_epoch": 5, "best_val_loss": 0.1},
        preprocessing_run_directory = "/runs/pp",
    )


def _profile_run():
    """Returns a profile autoencoder run stub, its z axis spanning 40 bins."""
    return SimpleNamespace(
        ae_name                     = "mlp_ae",
        embedding_dim               = 16,
        z_axis                      = np.linspace(0.0, 1.0, 40, dtype=np.float32),
        split_name                  = "test",
        split_regions               = [_region()],
        checkpoint_meta             = {"epoch": 9, "best_epoch": 3, "best_val_loss": 0.2},
        activity_threshold          = 1e-3,
        preprocessing_run_directory = "/runs/pp",
    )


def test_image_and_profile_reports_share_base():
    """Both autoencoder reports derive from InferenceReportBase."""
    assert issubclass(ImageAeReport, InferenceReportBase)
    assert issubclass(ProfileAeReport, InferenceReportBase)


def test_image_report_assembles_with_title_and_metrics(tmp_path):
    """The image report writes its title and metric table, and omits the figures section when there are none."""
    report = ImageAeReport(tmp_path, _image_run(), config=_config(), metrics=_metrics(), figures={}, report_path=tmp_path / "report.md")

    out  = report.assemble()
    text = out.read_text()

    assert out.exists()
    assert "Image Autoencoder Inference Report" in text
    assert "mse_mean" in text
    assert "Figures" not in text


def test_summaries_name_the_scored_checkpoint_and_one_based_epochs(tmp_path):
    """Both summaries carry the checkpoint file name, its one-based epoch and the one-based best epoch, so a final_model.pt inference reads differently from a best_model.pt one."""
    image   = ImageAeReport(tmp_path, _image_run(),     config=_config("final_model.pt"), metrics=_metrics(), figures={}, report_path=tmp_path / "image.md")._summary()
    profile = ProfileAeReport(tmp_path, _profile_run(), config=_config("final_model.pt"), metrics=_metrics(), figures={}, report_path=tmp_path / "profile.md")._summary()

    assert image["checkpoint"]         == "final_model.pt"
    assert image["checkpoint_epoch"]   == 6
    assert image["best_epoch"]         == 6
    assert profile["checkpoint"]       == "final_model.pt"
    assert profile["checkpoint_epoch"] == 10
    assert profile["best_epoch"]       == 4


def test_profile_summary_records_the_threshold_and_curve_sampling(tmp_path):
    """The profile summary names the activity threshold and the sampling settings that selected the scored curves."""
    summary = ProfileAeReport(tmp_path, _profile_run(), config=_config(), metrics=_metrics(), figures={}, report_path=tmp_path / "profile.md")._summary()

    assert summary["activity_threshold"] == 1e-3
    assert summary["pixel_subsample"]    == 0.5
    assert summary["keep_empty_frac"]    == 0.05
    assert summary["selection_seed"]     == 7


def test_profile_report_assembles_with_title_and_metrics(tmp_path):
    """The profile report writes its title, the profile length derived from the z axis, and the metric table."""
    report = ProfileAeReport(tmp_path, _profile_run(), config=_config(), metrics=_metrics(), figures={}, report_path=tmp_path / "report.md")

    out  = report.assemble()
    text = out.read_text()

    assert out.exists()
    assert "Profile Autoencoder Inference Report" in text
    assert "profile_length" in text
    assert "mse_mean" in text


def test_report_writes_figure_section_when_present(tmp_path):
    """A supplied figure adds the figures section with its captioned entry."""
    figures = {"error_histogram": [tmp_path / "error_histogram.png"]}
    report  = ImageAeReport(tmp_path, _image_run(), config=_config(), metrics=_metrics(), figures=figures, report_path=tmp_path / "report.md")

    text = report.assemble().read_text()

    assert "Figures" in text
    assert "Error distribution" in text
