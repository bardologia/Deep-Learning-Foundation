"""Tests for the crop region container and the train/val/test split regions."""

from __future__ import annotations

import pytest

from tools.data.regions import CropRegion, SplitRegions


def test_crop_region_sizes():
    """Verifies the row and column sizes are the extents of the half-open bounds."""
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.row_size == 1000
    assert region.col_size   == 500


def test_crop_region_as_tuple():
    """Verifies the tuple form is the row and column bounds in order."""
    region = CropRegion(10, 20, 30, 40)

    assert region.as_tuple() == (10, 20, 30, 40)


def test_crop_region_identifier_string():
    """Verifies the identifier string joins the four bounds with the 'a' separator."""
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.as_identifier_string() == "1000a2000a500a1000"


def test_crop_region_labeled_string():
    """Verifies the labelled string spells out the row and column bounds."""
    region = CropRegion(1000, 2000, 500, 1000)

    assert region.as_labeled_string() == "row1000-2000_col500-1000"


def test_crop_region_rejects_bad_rows():
    """Verifies a row end before the start is rejected."""
    with pytest.raises(ValueError):
        CropRegion(2000, 1000, 0, 10)


def test_crop_region_rejects_equal_rows():
    """Verifies an empty row extent is rejected."""
    with pytest.raises(ValueError):
        CropRegion(5, 5, 0, 10)


def test_crop_region_rejects_bad_range():
    """Verifies a range end before the start is rejected."""
    with pytest.raises(ValueError):
        CropRegion(0, 10, 100, 50)


def test_local_slices_relative_to_global():
    """Verifies local slices are offset by the global crop origin."""
    glob  = CropRegion(1000, 2000, 500, 1000)
    local = CropRegion(1100, 1300, 600, 700)

    row_slice, col_slice = local.local_slices(glob)

    assert row_slice == slice(100, 300)
    assert col_slice == slice(100, 200)


def test_local_slices_identity_when_equal():
    """Verifies a region taken against itself yields slices from zero."""
    glob = CropRegion(0, 50, 0, 50)
    row, col = glob.local_slices(glob)

    assert row == slice(0, 50)
    assert col == slice(0, 50)


def test_subdivide_by_row_covers_full_range():
    """Verifies row subdivision tiles the region with a short final part."""
    region = CropRegion(0, 1000, 0, 100)
    parts  = region.subdivide_by_rows(300)

    assert len(parts) == 4
    assert parts[0].as_tuple()  == (0, 300, 0, 100)
    assert parts[-1].as_tuple() == (900, 1000, 0, 100)
    assert sum(p.row_size for p in parts) == region.row_size


def test_subdivide_exact_multiple():
    """Verifies an exact multiple subdivides into equal parts."""
    region = CropRegion(0, 900, 0, 50)
    parts  = region.subdivide_by_rows(300)

    assert len(parts) == 3
    assert all(p.row_size == 300 for p in parts)


def test_subdivide_wider_than_region():
    """Verifies a subdivision width beyond the region yields the region itself."""
    region = CropRegion(0, 200, 0, 50)
    parts  = region.subdivide_by_rows(1000)

    assert len(parts) == 1
    assert parts[0].as_tuple() == (0, 200, 0, 50)




def _band_splits() -> SplitRegions:
    """Returns three contiguous row bands partitioning a 1000 by 500 crop."""
    return SplitRegions(
        train = CropRegion(0,   700,  0, 500),
        val   = CropRegion(700, 850,  0, 500),
        test  = CropRegion(850, 1000, 0, 500),
    )


def test_as_list_wraps_single():
    """Verifies a single region is wrapped in a one-element list."""
    region = CropRegion(0, 10, 0, 10)

    assert SplitRegions.as_list(region) == [region]


def test_as_list_passes_through_list():
    """Verifies a list of regions passes through unchanged."""
    a = CropRegion(0, 10, 0, 10)
    b = CropRegion(10, 20, 0, 10)

    assert SplitRegions.as_list([a, b]) == [a, b]


def test_regions_lookup_by_name():
    """Verifies regions are looked up by split name."""
    splits = _band_splits()

    assert splits.regions("train") == [splits.train]
    assert splits.regions("test")  == [splits.test]


def test_region_rows_single_region_labels():
    """Verifies the report rows are labelled by split name in order."""
    splits = _band_splits()
    rows   = splits.region_rows()

    labels = [r["Split"] for r in rows]
    assert labels == ["train", "val", "test"]


def test_region_rows_list_region_indexes():
    """Verifies a split holding several regions labels its rows by index."""
    a      = CropRegion(0, 100, 0, 50)
    b      = CropRegion(100, 200, 0, 50)
    splits = SplitRegions(train=[a, b], val=a, test=b)
    rows   = splits.region_rows()

    train_labels = [r["Split"] for r in rows if r["Split"].startswith("train")]
    assert train_labels == ["train[0]", "train[1]"]


@pytest.mark.dataset
def test_crop_region_matches_scene_config(config_state_json, scene_config):
    """Verifies the scene preprocessing crop spans 1000 rows by 500 columns."""
    crop   = config_state_json["crop"]
    region = CropRegion(crop["row_start"], crop["row_end"], crop["col_start"], crop["col_end"])

    assert region.row_size == scene_config.row_size
    assert region.col_size   == scene_config.col_size


@pytest.mark.dataset
def test_scene_crop_local_slices_index_data(config_state_json, cube_full):
    """Verifies local slices of the scene crop index the cube window they describe."""
    crop   = config_state_json["crop"]
    glob   = CropRegion(crop["row_start"], crop["row_end"], crop["col_start"], crop["col_end"])
    sub    = CropRegion(crop["row_start"], crop["row_start"] + 32, crop["col_start"], crop["col_start"] + 16)

    row, col = sub.local_slices(glob)
    window = cube_full[:, row, col]

    assert window.shape == (cube_full.shape[0], 32, 16)


def test_overlaps_detects_intersection():
    """Verifies overlap detection is symmetric and requires both axes to intersect."""
    a = CropRegion(0, 100, 0, 50)
    b = CropRegion(50, 150, 20, 80)
    c = CropRegion(100, 200, 0, 50)
    d = CropRegion(0, 100, 50, 100)

    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not a.overlaps(d)


def test_validate_disjoint_accepts_disjoint_splits():
    """Verifies disjoint splits pass validation."""
    splits = _band_splits()

    splits.validate_disjoint()


def test_validate_disjoint_rejects_overlapping_splits():
    """Verifies overlapping splits are rejected."""
    splits = SplitRegions(
        train = CropRegion(0, 120, 0, 50),
        val   = CropRegion(100, 200, 0, 50),
        test  = CropRegion(200, 300, 0, 50),
    )

    with pytest.raises(ValueError, match="overlap"):
        splits.validate_disjoint()


def test_validate_disjoint_rejects_overlap_within_split():
    """Verifies overlap between two regions of the same split names the offending index."""
    a = CropRegion(0, 100, 0, 50)
    splits = SplitRegions(
        train = [a, CropRegion(50, 150, 0, 50)],
        val   = CropRegion(150, 200, 0, 50),
        test  = CropRegion(200, 300, 0, 50),
    )

    with pytest.raises(ValueError, match="train\\[0\\]"):
        splits.validate_disjoint()

def test_contains_true_for_inner_and_equal_regions():
    """Verifies containment holds for inner and identical regions."""
    outer = CropRegion(100, 200, 50, 150)

    assert outer.contains(CropRegion(120, 180, 60, 140))
    assert outer.contains(outer)


def test_contains_false_when_any_edge_escapes():
    """Verifies containment fails when any edge leaves the outer region."""
    outer = CropRegion(100, 200, 50, 150)

    assert not outer.contains(CropRegion(90, 180, 60, 140))
    assert not outer.contains(CropRegion(120, 210, 60, 140))
    assert not outer.contains(CropRegion(120, 180, 40, 140))
    assert not outer.contains(CropRegion(120, 180, 60, 160))


def test_validate_within_accepts_contained_splits():
    """Verifies splits inside the global crop pass validation."""
    glob   = CropRegion(0, 1000, 0, 500)
    splits = _band_splits()

    splits.validate_within(glob)


def test_validate_within_rejects_region_outside_crop():
    """Verifies a split reaching outside the global crop is rejected by name."""
    glob   = CropRegion(100, 1000, 0, 500)
    stale  = CropRegion(70, 90, 0, 500)
    splits = SplitRegions(train=stale, val=CropRegion(100, 200, 0, 500), test=CropRegion(200, 300, 0, 500))

    with pytest.raises(ValueError, match="'train'.*not contained"):
        splits.validate_within(glob)


def test_validate_within_names_indexed_region_in_lists():
    """Verifies the offending region of a multi-region split is named by index."""
    glob   = CropRegion(0, 1000, 0, 500)
    good   = CropRegion(0, 100, 0, 500)
    bad    = CropRegion(900, 1100, 0, 500)
    splits = SplitRegions(train=[good, bad], val=CropRegion(100, 200, 0, 500), test=CropRegion(200, 300, 0, 500))

    with pytest.raises(ValueError, match=r"'train\[1\]'"):
        splits.validate_within(glob)
