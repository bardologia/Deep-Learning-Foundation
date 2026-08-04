from __future__ import annotations

import pytest

from tools.reporting.ranges import RangeFormatter


def test_single_value():
    assert RangeFormatter.compact([5]) == "5"


def test_two_contiguous():
    assert RangeFormatter.compact([3, 4]) == "3-4"


def test_full_contiguous_run():
    assert RangeFormatter.compact([0, 1, 2, 3]) == "0-3"


def test_single_gap_splits_runs():
    assert RangeFormatter.compact([0, 1, 3, 4]) == "0-1, 3-4"


def test_isolated_values():
    assert RangeFormatter.compact([0, 2, 4]) == "0, 2, 4"


def test_mixed_runs_and_singletons():
    assert RangeFormatter.compact([1, 2, 3, 7, 10, 11]) == "1-3, 7, 10-11"


def test_trailing_singleton():
    assert RangeFormatter.compact([1, 2, 5]) == "1-2, 5"


def test_leading_singleton():
    assert RangeFormatter.compact([0, 3, 4, 5]) == "0, 3-5"


def test_max_items_truncates_with_ellipsis():
    values = [0, 2, 4, 6, 8, 10, 12, 14]
    out    = RangeFormatter.compact(values, max_items=3)
    assert out == "0, 2, 4, ..."


def test_max_items_exactly_at_boundary_no_ellipsis():
    values = [0, 2, 4]
    assert RangeFormatter.compact(values, max_items=3) == "0, 2, 4"


def test_max_items_default_six():
    values = list(range(0, 20, 2))
    out    = RangeFormatter.compact(values)
    assert out.endswith("...")
    assert out.count(",") == 6


def test_negative_values():
    assert RangeFormatter.compact([-3, -2, -1]) == "-3--1"


def test_non_increasing_treated_as_break():
    assert RangeFormatter.compact([5, 4]) == "5, 4"


def test_duplicate_breaks_run():
    assert RangeFormatter.compact([1, 1, 2]) == "1, 1-2"


@pytest.mark.real_data
def test_compact_on_contiguous_track_index(track_profiles):
    keys = sorted(track_profiles.keys())
    assert keys

    indices = list(range(len(keys)))
    out     = RangeFormatter.compact(indices, max_items=len(indices))
    assert out == f"0-{len(indices) - 1}"
