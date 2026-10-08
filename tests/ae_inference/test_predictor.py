"""Tests for the autoencoder inference predictors of the image and profile families.

Covers the shared predictor and result base, the assembly of multi-batch reconstructions in
loader order, the denormalization of the stored arrays, the capture of reconstruction errors
in normalized units, the progress bar and footprint line of the split loop, and the guards
against a loader that does not cover its split.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pipelines.autoencoder_common.inference.predictor  import AeReconstructionPredictor, AeResult
from pipelines.image_autoencoder.inference.predictor   import ImageAePredictor, ImageAeResult
from pipelines.profile_autoencoder.inference.predictor import ProfileAePredictor, ProfileAeResult
from tests.conftest                                    import SilentLogger


class _Scaled:
    """Normalizer stub scaling by a constant factor in both directions."""

    def denormalize_input(self, x):
        """Returns the input scaled by four."""
        return x * 4.0

    def denormalize(self, x):
        """Returns the reconstruction scaled by four."""
        return x * 4.0


class _ImageModel:
    """Image autoencoder stub returning a scaled reconstruction and a zero embedding."""

    def reconstruct(self, x):
        """Returns the patch batch scaled by 0.9 and a zero embedding of shape (batch, 8)."""
        return x * 0.9, torch.zeros(x.shape[0], 8)


class _ProfileModel:
    """Profile autoencoder stub returning a scaled reconstruction and a zero embedding of shape (batch, 8, 1, 1)."""

    def reconstruct(self, x):
        """Returns the curve batch scaled by 0.9 and a zero embedding of shape (batch, 8, 1, 1)."""
        return x * 0.9, torch.zeros(x.shape[0], 8, 1, 1)


class _RecordingProgress:
    """Progress stub recording the task total and every advance.

    Attributes:
        total    : Total handed to add_task.
        advanced : Sum of the advances applied to the task.
    """

    def __init__(self):
        """Starts with no task."""
        self.total    = None
        self.advanced = 0

    def add_task(self, description, total=None, **kwargs):
        """Records the task total and returns a task id."""
        self.total = total
        return 0

    def update(self, task, advance=0, **kwargs):
        """Accumulates the advance."""
        self.advanced += advance


class _RecordingLogger(SilentLogger):
    """Silent logger exposing the progress stub and the info lines the predictor writes.

    Attributes:
        progress : Progress stub yielded by track.
        infos    : Info messages, in call order.
    """

    def __init__(self):
        """Creates the progress stub and an empty info list."""
        self.progress = _RecordingProgress()
        self.infos    = []

    def info(self, message, *args, **kwargs):
        """Records the info message."""
        self.infos.append(message)

    def track(self, transient=False):
        """Yields the recording progress stub."""
        return _ProgressContext(self.progress)


class _ProgressContext:
    """Context manager handing out a fixed progress stub.

    Attributes:
        progress : Progress stub returned on entry.
    """

    def __init__(self, progress):
        """Stores the stub to yield."""
        self.progress = progress

    def __enter__(self):
        """Returns the stub."""
        return self.progress

    def __exit__(self, exc_type, exc_value, traceback):
        """Propagates any exception."""
        return False


def _logger():
    """Returns a logger stub swallowing the section, progress, info and table calls the predictors make."""
    return SilentLogger()


class _Loader(list):
    """Data loader stub yielding fixed batches and reporting a dataset of ``n_samples`` entries."""

    def __init__(self, batches, n_samples):
        """Stores the batches to yield and the sample count the predictor checks coverage against."""
        super().__init__(batches)
        self.dataset = range(n_samples)


def test_predictors_share_base():
    """Both predictors and both result containers derive from the shared autoencoder bases."""
    assert issubclass(ImageAePredictor, AeReconstructionPredictor)
    assert issubclass(ProfileAePredictor, AeReconstructionPredictor)
    assert issubclass(ImageAeResult, AeResult)
    assert issubclass(ProfileAeResult, AeResult)


def test_image_predictor_run_inference():
    """Image inference returns patches of shape (patches, channels, height, width) and one embedding per patch."""
    x   = torch.rand(2, 1, 4, 4)
    run = SimpleNamespace(model=_ImageModel(), normalizer=_Scaled(), loader=_Loader([(x,)], 2))

    res = ImageAePredictor(run, "cpu", _logger()).run_inference()

    assert res.gt.shape == (2, 1, 4, 4)
    assert res.embeddings.shape == (2, 8)
    assert isinstance(res, ImageAeResult)


def test_profile_predictor_run_inference():
    """Profile inference returns curves of shape (curves, bins) and one embedding per curve."""
    x   = torch.rand(2, 20)
    run = SimpleNamespace(model=_ProfileModel(), normalizer=_Scaled(), loader=_Loader([(x, x)], 2))

    res = ProfileAePredictor(run, "cpu", _logger()).run_inference()

    assert res.gt.shape == (2, 20)
    assert res.embeddings.shape == (2, 8)
    assert isinstance(res, ProfileAeResult)


def test_normalized_errors_are_measured_before_denormalization():
    """The reported normalized MSE and MAE are measured on the network-space tensors, not the denormalized ones."""
    x   = torch.rand(2, 20)
    run = SimpleNamespace(model=_ProfileModel(), normalizer=_Scaled(), loader=_Loader([(x, x)], 2))

    res = ProfileAePredictor(run, "cpu", _logger()).run_inference()

    expected_mse = float(((x * 0.9 - x) ** 2).double().mean())
    expected_mae = float((x * 0.9 - x).abs().double().mean())

    assert res.normalized_errors["mse_mean_normalized"] == pytest.approx(expected_mse, rel=1e-6)
    assert res.normalized_errors["mae_mean_normalized"] == pytest.approx(expected_mae, rel=1e-6)


def test_multi_batch_reconstruction_is_assembled_in_loader_order():
    """Batches of unequal size are concatenated in loader order and denormalized."""
    first  = torch.rand(2, 20)
    second = torch.rand(3, 20)
    run    = SimpleNamespace(model=_ProfileModel(), normalizer=_Scaled(), loader=_Loader([(first, first), (second, second)], 5))

    res = ProfileAePredictor(run, "cpu", _logger()).run_inference()

    assert res.gt.shape         == (5, 20)
    assert res.pred.shape       == (5, 20)
    assert res.embeddings.shape == (5, 8)

    torch.testing.assert_close(torch.from_numpy(res.gt[:2]), first * 4.0)
    torch.testing.assert_close(torch.from_numpy(res.gt[2:]), second * 4.0)


def test_split_loop_tracks_one_step_per_batch_and_logs_the_array_footprint():
    """The progress task spans the loader length, advances once per batch, and the footprint line names the samples, shapes and gigabytes held."""
    first  = torch.rand(2, 20)
    second = torch.rand(3, 20)
    run    = SimpleNamespace(model=_ProfileModel(), normalizer=_Scaled(), loader=_Loader([(first, first), (second, second)], 5))
    logger = _RecordingLogger()

    ProfileAePredictor(run, "cpu", logger).run_inference()

    assert logger.progress.total    == 2
    assert logger.progress.advanced == 2

    footprint = [message for message in logger.infos if message.startswith("Result arrays")]
    assert len(footprint) == 1
    assert "5 samples x (20,)" in footprint[0]
    assert "embeddings (8,)"   in footprint[0]
    assert "GB held in RAM"    in footprint[0]


def test_uncovered_split_raises():
    """A loader yielding fewer samples than its dataset holds raises rather than reporting a partial split."""
    x   = torch.rand(2, 20)
    run = SimpleNamespace(model=_ProfileModel(), normalizer=_Scaled(), loader=_Loader([(x, x)], 5))

    with pytest.raises(ValueError, match="must not drop the last batch"):
        ProfileAePredictor(run, "cpu", _logger()).run_inference()


def test_empty_loader_raises():
    """A loader yielding no batch at all raises."""
    run = SimpleNamespace(model=_ProfileModel(), normalizer=_Scaled(), loader=_Loader([], 5))

    with pytest.raises(ValueError, match="yielded no batches"):
        ProfileAePredictor(run, "cpu", _logger()).run_inference()
