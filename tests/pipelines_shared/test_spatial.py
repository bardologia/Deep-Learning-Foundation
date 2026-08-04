from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.dataset.spatial import GridInfo, Patcher


def test_grid_count_no_padding_exact_tiling():
    patcher = Patcher.build(spatial_size=(64, 64), patch_size=(32, 32), stride=(32, 32))

    assert patcher.grid.number_of_patches == 4
    assert patcher.grid.padding_vertical   == 0
    assert patcher.grid.padding_horizontal == 0


def test_grid_count_with_overlap_and_padding():
    patcher = Patcher.build(spatial_size=(50, 50), patch_size=(32, 32), stride=(16, 16))
    grid    = patcher.grid

    assert grid.n_v == 3
    assert grid.n_h == 3
    assert grid.padded_size[0] >= 50
    assert grid.padded_size[1] >= 50


def test_single_patch_when_image_smaller_than_patch():
    patcher = Patcher.build(spatial_size=(10, 10), patch_size=(32, 32), stride=(16, 16))

    assert patcher.grid.number_of_patches == 1


def test_extract_preserves_patch_shape():
    array   = np.arange(3 * 40 * 40, dtype=np.float32).reshape(3, 40, 40)
    patcher = Patcher.build(spatial_size=(40, 40), patch_size=(16, 16), stride=(16, 16))

    for idx in range(patcher.grid.number_of_patches):
        patch = patcher.extract(array, idx)
        assert patch.shape == (3, 16, 16)


def test_extract_interior_patch_matches_source_slice():
    array   = np.arange(50 * 50, dtype=np.float32).reshape(50, 50)
    patcher = Patcher.build(spatial_size=(50, 50), patch_size=(32, 32), stride=(16, 16))

    v0c, v1c, h0c, h1c, pw_spec = patcher._patch_coords[0]
    patch                       = patcher.extract(array, 0)

    if pw_spec is None:
        np.testing.assert_array_equal(patch, array[v0c:v1c, h0c:h1c])


def test_extract_covers_every_pixel_when_unpadded():
    H = W   = 64
    array   = np.zeros((H, W), dtype=np.int64)
    patcher = Patcher.build(spatial_size=(H, W), patch_size=(32, 32), stride=(32, 32))

    for idx in range(patcher.grid.number_of_patches):
        v0c, v1c, h0c, h1c, _ = patcher._patch_coords[idx]
        array[v0c:v1c, h0c:h1c] += 1

    assert np.all(array >= 1)


def test_reflective_padding_mode_recorded():
    patcher = Patcher.build(spatial_size=(50, 50), patch_size=(32, 32), stride=(16, 16), use_symmetric_padding=True)
    modes   = {coord[4][4] for coord in patcher._patch_coords if coord[4] is not None}

    assert modes == {"symmetric"}


def test_constant_padding_mode_recorded():
    patcher = Patcher.build(spatial_size=(50, 50), patch_size=(32, 32), stride=(16, 16), use_symmetric_padding=False)
    modes   = {coord[4][4] for coord in patcher._patch_coords if coord[4] is not None}

    assert modes == {"constant"}


def test_gridinfo_as_dict_roundtrip_fields():
    grid = GridInfo(
        n_v=2, n_h=3, pad_top=1, pad_bot=1, pad_left=0, pad_right=2,
        patch_size=(16, 16), stride=(8, 8), spatial_size=(30, 40),
    )
    payload = grid.as_dict()

    assert payload["number_of_patches"] == 6
    assert payload["patch_size"]        == [16, 16]
    assert payload["stride"]            == [8, 8]
    assert payload["spatial_size"]      == [30, 40]
    assert grid.padding_vertical        == 2
    assert grid.padding_horizontal      == 2


def test_rectangular_patch_exact_tiling():
    patcher = Patcher.build(spatial_size=(64, 96), patch_size=(32, 48), stride=(32, 48))
    grid    = patcher.grid

    assert grid.n_v == 2
    assert grid.n_h == 2
    assert grid.padding_vertical   == 0
    assert grid.padding_horizontal == 0


def test_rectangular_patch_extract_preserves_shape():
    array   = np.arange(3 * 40 * 60, dtype=np.float32).reshape(3, 40, 60)
    patcher = Patcher.build(spatial_size=(40, 60), patch_size=(16, 24), stride=(8, 12))

    for idx in range(patcher.grid.number_of_patches):
        patch = patcher.extract(array, idx)
        assert patch.shape == (3, 16, 24)


def test_rectangular_patch_covers_every_pixel():
    H, W    = 50, 70
    array   = np.zeros((H, W), dtype=np.int64)
    patcher = Patcher.build(spatial_size=(H, W), patch_size=(16, 32), stride=(8, 16))

    for idx in range(patcher.grid.number_of_patches):
        v0c, v1c, h0c, h1c, _ = patcher._patch_coords[idx]
        array[v0c:v1c, h0c:h1c] += 1

    assert np.all(array >= 1)


def test_anisotropic_stride_on_square_patch():
    patcher = Patcher.build(spatial_size=(64, 64), patch_size=(32, 32), stride=(16, 32))
    grid    = patcher.grid

    assert grid.n_v == 3
    assert grid.n_h == 2
    assert grid.stride == (16, 32)


def test_rectangular_patch_interior_matches_source_slice():
    array   = np.arange(48 * 80, dtype=np.float32).reshape(48, 80)
    patcher = Patcher.build(spatial_size=(48, 80), patch_size=(16, 40), stride=(16, 40))

    for idx in range(patcher.grid.number_of_patches):
        v0c, v1c, h0c, h1c, pw_spec = patcher._patch_coords[idx]

        assert pw_spec is None
        np.testing.assert_array_equal(patcher.extract(array, idx), array[v0c:v1c, h0c:h1c])
