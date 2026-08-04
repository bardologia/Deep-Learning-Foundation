from __future__ import annotations

from types import SimpleNamespace

from pipelines.shared.inference.metadata              import InferenceMetadata
from pipelines.image_autoencoder.inference.pipeline   import ImageAeInferenceMetadata
from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferenceMetadata


def _config(tmp_path, output_subdir=""):
    paths = SimpleNamespace(figures_subdir="figures", logs_subdir="logs", metrics_filename="metrics.json", report_filename="report.md")
    return SimpleNamespace(paths=paths, run_directory=tmp_path, output_subdir=output_subdir)


def test_metadata_share_base():
    assert issubclass(ImageAeInferenceMetadata, InferenceMetadata)
    assert issubclass(ProfileAeInferenceMetadata, InferenceMetadata)


def test_subdir_distinguishes_image_and_profile(tmp_path):
    img  = ImageAeInferenceMetadata(_config(tmp_path, "out"))
    prof = ProfileAeInferenceMetadata(_config(tmp_path, "out"))

    assert img.output_dir   == tmp_path / "inference" / "image_ae" / "out"
    assert prof.output_dir  == tmp_path / "inference" / "profile_ae" / "out"
    assert img.figures_dir  == img.output_dir / "figures"
    assert img.metrics_path == img.output_dir / "metrics.json"


def test_create_dirs(tmp_path):
    m = ImageAeInferenceMetadata(_config(tmp_path, "out"))
    m.create_dirs()

    assert m.output_dir.is_dir() and m.figures_dir.is_dir() and m.logs_dir.is_dir()
