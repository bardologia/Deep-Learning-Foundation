"""Tests that the image and profile autoencoder inference stacks share one base implementation.

Covers the metadata directory layout each family derives, the checkpoint z-axis
guard of the profile run loader, and that a failing pipeline stage still closes its logger.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.autoencoder_common.inference.pipeline  import AeInferencePipeline
from pipelines.image_autoencoder.inference.loader     import ImageAeRunLoader
from pipelines.profile_autoencoder.inference.loader   import ProfileAeRunLoader
from pipelines.shared.inference.metadata              import InferenceMetadata
from pipelines.shared.inference.run_artifacts         import RunArtifactLoader
from pipelines.image_autoencoder.inference.pipeline   import ImageAeInferenceMetadata, ImageAeInferencePipeline
from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferenceMetadata, ProfileAeInferencePipeline


def _config(tmp_path, output_subdir=""):
    """Returns a stub inference config carrying the path fields the metadata classes read."""
    paths = SimpleNamespace(figures_subdir="figures", logs_subdir="logs", metrics_filename="metrics.json", report_filename="report.md")
    return SimpleNamespace(paths=paths, run_directory=tmp_path, output_subdir=output_subdir)


def test_metadata_share_base():
    """Both autoencoder metadata classes derive from the shared InferenceMetadata base."""
    assert issubclass(ImageAeInferenceMetadata, InferenceMetadata)
    assert issubclass(ProfileAeInferenceMetadata, InferenceMetadata)


def test_run_loaders_share_the_run_artifact_base():
    """Both autoencoder run loaders inherit the construction and checkpoint loading of RunArtifactLoader."""
    assert issubclass(ImageAeRunLoader, RunArtifactLoader)
    assert issubclass(ProfileAeRunLoader, RunArtifactLoader)

    assert ProfileAeRunLoader.__init__          is RunArtifactLoader.__init__
    assert ProfileAeRunLoader._load_checkpoint  is RunArtifactLoader._load_checkpoint


def test_subdir_distinguishes_image_and_profile(tmp_path):
    """Image and profile metadata place their outputs under distinct family subdirectories."""
    img  = ImageAeInferenceMetadata(_config(tmp_path, "out"))
    prof = ProfileAeInferenceMetadata(_config(tmp_path, "out"))

    assert img.output_dir   == tmp_path / "inference" / "image_ae" / "out"
    assert prof.output_dir  == tmp_path / "inference" / "profile_ae" / "out"
    assert img.figures_dir  == img.output_dir / "figures"
    assert img.metrics_path == img.output_dir / "metrics.json"


def test_create_dirs(tmp_path):
    """create_dirs materializes the output, figures and logs directories."""
    m = ImageAeInferenceMetadata(_config(tmp_path, "out"))
    m.create_dirs()

    assert m.output_dir.is_dir() and m.figures_dir.is_dir() and m.logs_dir.is_dir()


def test_x_axis_validation_rejects_a_shifted_axis_of_equal_length(tmp_path):
    """The z axis guard rejects an axis shifted in metres even at equal bin count."""
    axis    = np.linspace(0.0, 30.0, 16).astype(np.float32)
    shifted = axis + 5.0

    ProfileAeRunLoader._validate_x_axis(axis, axis.copy(), tmp_path / "best_model.pt")

    with pytest.raises(ValueError, match="z axis"):
        ProfileAeRunLoader._validate_x_axis(axis, shifted, tmp_path / "best_model.pt")


def test_x_axis_validation_rejects_a_different_bin_count(tmp_path):
    """The z axis guard rejects an axis with a different number of bins."""
    with pytest.raises(ValueError, match="z axis"):
        ProfileAeRunLoader._validate_x_axis(
            np.linspace(0.0, 30.0, 16).astype(np.float32),
            np.linspace(0.0, 30.0, 32).astype(np.float32),
            tmp_path / "best_model.pt",
        )


def test_inference_pipelines_share_the_ae_base():
    """Neither autoencoder pipeline overrides any stage of the shared AeInferencePipeline."""
    assert issubclass(ImageAeInferencePipeline, AeInferencePipeline)
    assert issubclass(ProfileAeInferencePipeline, AeInferencePipeline)

    for attribute in ("_setup", "_load_run", "_predict", "_evaluate_metrics", "_compose_figures", "_build_report", "run"):
        assert getattr(ImageAeInferencePipeline, attribute)   is getattr(AeInferencePipeline, attribute)
        assert getattr(ProfileAeInferencePipeline, attribute) is getattr(AeInferencePipeline, attribute)


class _FailingPipeline(AeInferencePipeline):
    """Autoencoder inference pipeline whose run-loading stage always raises."""

    FAMILY      = "Test AE"
    LOGGER_NAME = "test_ae_inference"

    def _setup(self):
        """Returns a minimal metadata stub and the logger carried by the config."""
        return SimpleNamespace(output_dir=self.config.run_directory), self.config.logger

    def _load_run(self, logger):
        """Raises to simulate a missing checkpoint."""
        raise RuntimeError("checkpoint is missing")


def test_inference_logger_is_closed_when_a_stage_fails(tmp_path):
    """A stage raising mid-run still logs the error and closes the logger before propagating."""
    logged = []

    logger = SimpleNamespace(error=lambda message: logged.append(message), close=lambda: logged.append("closed"))
    config = SimpleNamespace(run_directory=tmp_path, logger=logger)

    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        _FailingPipeline(config).run()

    assert logged[-1] == "closed"
    assert "checkpoint is missing" in logged[0]
