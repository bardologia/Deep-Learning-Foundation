
from __future__ import annotations

from dataclasses import asdict

import pytest

from tools.data.regions import CropRegion, SplitRegions


def test_crop_region_sizes():
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.row_size == 1000
    assert region.col_size   == 500


def test_crop_region_as_tuple():
    region = CropRegion(10, 20, 30, 40)

    assert region.as_tuple() == (10, 20, 30, 40)


def test_crop_region_identifier_string():
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.as_identifier_string() == "1000a2000a500a1000"


def test_crop_region_labeled_string():
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.as_labeled_string() == "row1000-2000_col500-1000"


def test_local_slices_relative_to_global():
    glob  = CropRegion(1000, 2000, 500, 1000)
    local = CropRegion(1100, 1300, 600, 700)

    row_slice, col_slice = local.local_slices(glob)

    assert row_slice == slice(100, 300)
    assert col_slice == slice(100, 200)


def test_local_slices_identity_when_equal():
    glob = CropRegion(0, 50, 0, 50)
    row, col = glob.local_slices(glob)

    assert row == slice(0, 50)
    assert col == slice(0, 50)


def test_subdivide_by_row_covers_full_range():
    region = CropRegion(0, 1000, 0, 100)
    parts  = region.subdivide_by_rows(300)

    assert len(parts) == 4
    assert parts[0].as_tuple()  == (0, 300, 0, 100)
    assert parts[-1].as_tuple() == (900, 1000, 0, 100)
    assert sum(p.row_size for p in parts) == region.row_size


def test_subdivide_exact_multiple():
    region = CropRegion(0, 900, 0, 50)
    parts  = region.subdivide_by_rows(300)

    assert len(parts) == 3
    assert all(p.row_size == 300 for p in parts)


def test_subdivide_wider_than_region():
    region = CropRegion(0, 200, 0, 50)
    parts  = region.subdivide_by_rows(1000)

    assert len(parts) == 1
    assert parts[0].as_tuple() == (0, 200, 0, 50)




def _band_splits() -> SplitRegions:
    return SplitRegions(
        train = CropRegion(0,   700,  0, 500),
        val   = CropRegion(700, 850,  0, 500),
        test  = CropRegion(850, 1000, 0, 500),
    )


def test_as_list_wraps_single():
    region = CropRegion(0, 10, 0, 10)

    assert SplitRegions.as_list(region) == [region]


def test_as_list_passes_through_list():
    a = CropRegion(0, 10, 0, 10)
    b = CropRegion(10, 20, 0, 10)

    assert SplitRegions.as_list([a, b]) == [a, b]


def test_regions_lookup_by_name():
    splits = _band_splits()

    assert splits.regions("train") == [splits.train]
    assert splits.regions("test")  == [splits.test]


def test_region_rows_single_region_labels():
    splits = _band_splits()
    rows   = splits.region_rows()

    labels = [r["Split"] for r in rows]
    assert labels == ["train", "val", "test"]


def test_region_rows_list_region_indexes():
    a      = CropRegion(0, 100, 0, 50)
    b      = CropRegion(100, 200, 0, 50)
    splits = SplitRegions(train=[a, b], val=a, test=b)
    rows   = splits.region_rows()

    train_labels = [r["Split"] for r in rows if r["Split"].startswith("train")]
    assert train_labels == ["train[0]", "train[1]"]


@pytest.mark.dataset
def test_crop_region_matches_scene_config(config_state_json, scene_config):
    crop   = config_state_json["crop"]
    region = CropRegion(crop["row_start"], crop["row_end"], crop["col_start"], crop["col_end"])

    assert region.row_size == scene_config.row_size
    assert region.col_size   == scene_config.col_size


@pytest.mark.dataset
def test_scene_crop_local_slices_index_data(config_state_json, cube_full):
    crop   = config_state_json["crop"]
    glob   = CropRegion(crop["row_start"], crop["row_end"], crop["col_start"], crop["col_end"])
    sub    = CropRegion(crop["row_start"], crop["row_start"] + 32, crop["col_start"], crop["col_start"] + 16)

    row, col = sub.local_slices(glob)
    window = cube_full[:, row, col]

    assert window.shape == (cube_full.shape[0], 32, 16)


def test_overlaps_detects_intersection():
    a = CropRegion(0, 100, 0, 50)
    b = CropRegion(50, 150, 20, 80)
    c = CropRegion(100, 200, 0, 50)
    d = CropRegion(0, 100, 50, 100)

    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not a.overlaps(d)


def test_contains_true_for_inner_and_equal_regions():
    outer = CropRegion(100, 200, 50, 150)

    assert outer.contains(CropRegion(120, 180, 60, 140))
    assert outer.contains(outer)


def test_contains_false_when_any_edge_escapes():
    outer = CropRegion(100, 200, 50, 150)

    assert not outer.contains(CropRegion(90, 180, 60, 140))
    assert not outer.contains(CropRegion(120, 210, 60, 140))
    assert not outer.contains(CropRegion(120, 180, 40, 140))
    assert not outer.contains(CropRegion(120, 180, 60, 160))


def test_split_regions_round_trip_through_their_persisted_form():
    regions = SplitRegions(
        train = [CropRegion(1000, 1162, 500, 1000), CropRegion(1348, 1512, 500, 1000)],
        val   = CropRegion(1178, 1332, 500, 1000),
        test  = CropRegion(1000, 1162, 500, 1000),
    )

    assert SplitRegions.from_payload(asdict(regions)) == regions
