from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipelines.shared.inference.report              import InferenceReportBase
from pipelines.image_autoencoder.inference.report   import ImageAeReport
from pipelines.profile_autoencoder.inference.report import ProfileAeReport


def _region():
    return SimpleNamespace(as_tuple=lambda: (0, 64))


def _metrics():
    return {
        "mse_mean"            : 0.5,
        "mae_mean"            : 0.25,
        "embedding_norm_mean" : 1.5,
    }


def _image_run():
    return SimpleNamespace(
        ae_name                     = "conv_ae",
        embedding_dim               = 32,
        in_channels                 = 4,
        patch_size                  = 64,
        split_name                  = "test",
        split_region                = _region(),
        checkpoint_meta             = {"best_epoch": 5, "best_val_loss": 0.1},
        preprocessing_run_directory = "/runs/pp",
    )


def _profile_run():
    return SimpleNamespace(
        ae_name                     = "mlp_ae",
        embedding_dim               = 16,
        x_axis                      = np.linspace(0.0, 1.0, 40, dtype=np.float32),
        split_name                  = "test",
        split_regions               = [_region()],
        checkpoint_meta             = {"best_epoch": 3, "best_val_loss": 0.2},
        preprocessing_run_directory = "/runs/pp",
    )


def test_image_and_profile_reports_share_base():
    assert issubclass(ImageAeReport, InferenceReportBase)
    assert issubclass(ProfileAeReport, InferenceReportBase)


def test_image_report_assembles_with_title_and_metrics(tmp_path):
    report = ImageAeReport(tmp_path, _image_run(), config=None, metrics=_metrics(), figures={}, report_path=tmp_path / "report.md")

    out  = report.assemble()
    text = out.read_text()

    assert out.exists()
    assert "Image Autoencoder Inference Report" in text
    assert "mse_mean" in text
    assert "Figures" not in text


def test_profile_report_assembles_with_title_and_metrics(tmp_path):
    report = ProfileAeReport(tmp_path, _profile_run(), config=None, metrics=_metrics(), figures={}, report_path=tmp_path / "report.md")

    out  = report.assemble()
    text = out.read_text()

    assert out.exists()
    assert "Profile Autoencoder Inference Report" in text
    assert "profile_length" in text
    assert "mse_mean" in text


def test_report_writes_figure_section_when_present(tmp_path):
    figures = {"error_histogram": [tmp_path / "error_histogram.png"]}
    report  = ImageAeReport(tmp_path, _image_run(), config=None, metrics=_metrics(), figures=figures, report_path=tmp_path / "report.md")

    text = report.assemble().read_text()

    assert "Figures" in text
    assert "Error distribution" in text
