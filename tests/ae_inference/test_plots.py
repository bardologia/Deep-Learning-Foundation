from __future__ import annotations

import numpy as np

from pipelines.autoencoder_common.inference.plots  import AePlotsBase
from pipelines.image_autoencoder.inference.plots   import ImageAePlots
from pipelines.profile_autoencoder.inference.plots import ProfileAePlots


def test_plotters_share_base():
    assert issubclass(ImageAePlots, AePlotsBase)
    assert issubclass(ProfileAePlots, AePlotsBase)


def test_bins_degenerate_and_normal():
    p = ImageAePlots()

    assert p._bins(np.array([1.0])) == 1
    assert p._bins(np.full(10, 3.0)) == 1
    assert p._bins(np.linspace(0.0, 1.0, 100)) == 60


def test_error_histogram_xlabel_differs_per_domain():
    assert ImageAePlots.ERROR_XLABEL != ProfileAePlots.ERROR_XLABEL
    assert "patch" in ImageAePlots.ERROR_XLABEL
    assert "curve" in ProfileAePlots.ERROR_XLABEL


def test_error_histogram_and_embedding_norm_save_files(tmp_path):
    p   = ProfileAePlots()
    mse = np.abs(np.random.default_rng(0).normal(size=200)) + 1e-3
    emb = np.random.default_rng(1).normal(size=(200, 8))

    hist = p._error_histogram(mse, tmp_path)
    norm = p._embedding_norm(emb, tmp_path)

    assert hist.exists() and hist.name == "error_histogram.png"
    assert norm.exists() and norm.name == "embedding_norm.png"
