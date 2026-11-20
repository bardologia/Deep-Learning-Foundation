
from __future__ import annotations

import json


from tools.data.regions import CropRegion, SplitRegions


def test_crop_region_sizes():
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.row_size == 1000
    assert region.col_size   == 500


def test_crop_region_as_tuple():
    region = CropRegion(10, 20, 30, 40)

    assert region.as_tuple() == (10, 20, 30, 40)


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
        train = [CropRegion(0, 162, 0, 500), CropRegion(348, 512, 0, 500)],
        val   = [CropRegion(178, 332, 0, 500)],
        test  = [CropRegion(0, 162, 0, 500)],
    )

    assert regions.bounds()["train"] == [[0, 162, 0, 500], [348, 512, 0, 500]]
    assert SplitRegions.from_bounds(json.loads(json.dumps(regions.bounds()))) == regions
