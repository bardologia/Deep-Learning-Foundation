"""Tests for the autoencoder inference plotters shared between the image and profile families.

Covers histogram binning, the per-domain error labels, the latent RGB encoding of the three
highest-variance embedding components, and the mapping of flat pixel indices back onto the
azimuth-range grid of each split region.
"""

from __future__ import annotations

import numpy  as np
import pytest

from pipelines.autoencoder_common.inference.plots  import AePlotsBase
from pipelines.image_autoencoder.inference.plots   import ImageAePlots
from pipelines.profile_autoencoder.inference.plots import ProfileAePlots
from tools.data.regions                            import CropRegion


def test_plotters_share_base():
    """Both autoencoder plotters derive from AePlotsBase."""
    assert issubclass(ImageAePlots, AePlotsBase)
    assert issubclass(ProfileAePlots, AePlotsBase)


def test_bins_degenerate_and_normal():
    """Histogram binning collapses to a single bin for constant data and caps at 60 otherwise."""
    p = ImageAePlots()

    assert p._bins(np.array([1.0])) == 1
    assert p._bins(np.full(10, 3.0)) == 1
    assert p._bins(np.linspace(0.0, 1.0, 100)) == 60


def test_error_histogram_xlabel_differs_per_domain():
    """The error histogram axis label names patches for images and curves for profiles."""
    assert ImageAePlots.ERROR_XLABEL != ProfileAePlots.ERROR_XLABEL
    assert "patch" in ImageAePlots.ERROR_XLABEL
    assert "curve" in ProfileAePlots.ERROR_XLABEL


def test_error_histogram_and_embedding_norm_save_files(tmp_path):
    """The error histogram and embedding norm plots are written under their expected file names."""
    p   = ProfileAePlots()
    mse = np.abs(np.random.default_rng(0).normal(size=200)) + 1e-3
    emb = np.random.default_rng(1).normal(size=(200, 8))

    hist = p._error_histogram(mse, tmp_path)
    norm = p._embedding_norm(emb, tmp_path)

    assert hist.exists() and hist.name == "error_histogram.png"
    assert norm.exists() and norm.name == "embedding_norm.png"


def test_top_variance_components_are_ranked_by_spread():
    """The three selected embedding components are the highest-variance ones, in descending variance."""
    p   = ProfileAePlots()
    emb = np.random.default_rng(2).normal(size=(500, 6)) * np.array([0.1, 5.0, 0.2, 3.0, 0.05, 1.0])

    ranked, variances = p._top_variance_components(emb)

    assert list(ranked) == [1, 3, 5]
    assert np.all(np.diff(variances) <= 0.0)


def test_top_variance_components_reject_a_narrow_embedding():
    """Selecting the top components raises when the embedding holds fewer than three dimensions."""
    p = ProfileAePlots()

    with pytest.raises(ValueError):
        p._top_variance_components(np.zeros((10, 2)))


def test_rgb_channels_span_the_displayable_range():
    """Each RGB channel is rescaled to span the full brightness floor to one range."""
    p       = ProfileAePlots()
    columns = np.random.default_rng(3).normal(size=(1000, 3))

    rgb = p._rgb_channels(columns)

    assert rgb.shape == columns.shape
    assert rgb.min() == ProfileAePlots.RGB_BRIGHTNESS and rgb.max() == 1.0
    assert np.all((rgb >= ProfileAePlots.RGB_BRIGHTNESS) & (rgb <= 1.0))


def test_rgb_channels_survive_a_constant_component():
    """A constant embedding component maps to the brightness floor instead of producing NaNs."""
    p       = ProfileAePlots()
    columns = np.column_stack([np.linspace(0.0, 1.0, 50), np.full(50, 2.0), np.linspace(-1.0, 1.0, 50)])

    rgb = p._rgb_channels(columns)

    assert np.all(rgb[:, 1] == ProfileAePlots.RGB_BRIGHTNESS)
    assert np.isfinite(rgb).all()


def test_latent_rgb_scatter_saves_a_file(tmp_path):
    """The latent RGB scatter is written under its expected file name."""
    p   = ProfileAePlots()
    emb = np.random.default_rng(4).normal(size=(300, 8)) * np.linspace(0.1, 2.0, 8)

    path = p._latent_rgb_scatter(emb, n_points=100, seed=0, figures_dir=tmp_path)

    assert path.exists() and path.name == "latent_rgb_scatter.png"


def test_region_pixels_invert_the_row_major_stacking():
    """Flat pixel indices are split by region and inverted into per-region azimuth and range coordinates."""
    p       = ProfileAePlots()
    regions = [CropRegion(0, 4, 0, 5), CropRegion(10, 12, 0, 5)]
    bounds  = p._region_bounds(regions)

    assert list(bounds) == [0, 20, 30]

    pixel_index           = np.array([0, 7, 19, 20, 26], dtype=np.int64)
    rows, azimuth, ranges = p._region_pixels(pixel_index, regions[0], int(bounds[0]), int(bounds[1]))

    assert list(rows)    == [0, 1, 2]
    assert list(azimuth) == [0, 1, 3]
    assert list(ranges)  == [0, 2, 4]

    rows, azimuth, ranges = p._region_pixels(pixel_index, regions[1], int(bounds[1]), int(bounds[2]))

    assert list(rows)    == [3, 4]
    assert list(azimuth) == [0, 1]
    assert list(ranges)  == [0, 1]


def test_latent_rgb_maps_write_one_image_per_region(tmp_path):
    """One latent RGB map is written per split region, named by its azimuth and range bounds."""
    p           = ProfileAePlots()
    regions     = [CropRegion(0, 4, 0, 5), CropRegion(10, 12, 0, 5)]
    pixel_index = np.arange(30, dtype=np.int64)
    emb         = np.random.default_rng(5).normal(size=(30, 6)) * np.linspace(0.1, 2.0, 6)

    paths = p._latent_rgb_maps(emb, pixel_index, regions, tmp_path)

    assert [path.name for path in paths] == ["az0-4_rg0-5.png", "az10-12_rg0-5.png"]
    assert all(path.exists() for path in paths)
    assert all(path.parent.name == "latent_rgb_maps" for path in paths)


def test_latent_rgb_maps_reject_an_index_outside_the_regions(tmp_path):
    """A pixel index beyond the stacked region extent raises instead of silently wrapping."""
    p       = ProfileAePlots()
    regions = [CropRegion(0, 4, 0, 5)]
    emb     = np.random.default_rng(6).normal(size=(3, 6))

    with pytest.raises(ValueError):
        p._latent_rgb_maps(emb, np.array([0, 1, 20], dtype=np.int64), regions, tmp_path)


def test_latent_rgb_maps_reject_a_mismatched_index_length(tmp_path):
    """A pixel index array shorter than the embedding array raises."""
    p       = ProfileAePlots()
    regions = [CropRegion(0, 4, 0, 5)]
    emb     = np.random.default_rng(7).normal(size=(4, 6))

    with pytest.raises(ValueError):
        p._latent_rgb_maps(emb, np.array([0, 1, 2], dtype=np.int64), regions, tmp_path)
