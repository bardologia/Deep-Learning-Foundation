"""Tests how completed inference stamps are discovered when a split was scored region by region.

Covers the region pass recorded in a completion marker, the rule that a region
group counts as a completed inference only when every one of its regions is
stamped, the listing of groups still missing regions, and single-pass stamps
that keep counting on their own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.shared.inference.metadata import CompletedInference, RegionPass
from tools.runtime.completion            import CompletionMarker


def _stamp(base: Path, name: str, split: str = "train", region: RegionPass | None = None) -> Path:
    """Writes a completed stamp under the base, recording the region pass when given."""
    stamp = base / name
    stamp.mkdir(parents=True)

    payload = {"stage": "inference", "split": split}
    if region is not None:
        payload[RegionPass.KEY] = region.payload()

    CompletionMarker.stamp(stamp, payload)

    return stamp


def test_region_pass_round_trips_through_the_marker_payload():
    """A region pass written into a marker payload is read back as the same pass, and a single-pass payload reads as none."""
    region = RegionPass(index=1, count=3, group="20260101_000000")

    assert RegionPass.from_payload({"split": "train", RegionPass.KEY: region.payload()}) == region
    assert RegionPass.from_payload({"split": "train"}) is None


def test_a_region_group_missing_a_region_does_not_count_as_completed(tmp_path):
    """Stamps of a group whose other region was never scored are left out of the completed stamps and reported as incomplete."""
    base = tmp_path / "inference"
    _stamp(base, "20260101_000000_region0", region=RegionPass(index=0, count=2, group="20260101_000000"))

    assert CompletedInference.stamps(base, "train") == []
    assert CompletedInference.latest(base, "train") is None
    assert CompletedInference.incomplete_groups(base, "train") == {"20260101_000000": ([0], 2)}


def test_a_fully_stamped_region_group_counts_as_completed(tmp_path):
    """Once every region of the group is stamped, all of its stamps count and the group is no longer incomplete."""
    base   = tmp_path / "inference"
    first  = _stamp(base, "20260101_000000_region0", region=RegionPass(index=0, count=2, group="20260101_000000"))
    second = _stamp(base, "20260101_000000_region1", region=RegionPass(index=1, count=2, group="20260101_000000"))

    assert CompletedInference.stamps(base, "train")            == [first, second]
    assert CompletedInference.latest(base, "train")            == second
    assert CompletedInference.incomplete_groups(base, "train") == {}


def test_single_pass_stamps_count_beside_an_incomplete_group(tmp_path):
    """A single-pass stamp on the split still counts while an incomplete group next to it does not, and other splits are untouched."""
    base   = tmp_path / "inference"
    single = _stamp(base, "20251231_000000")
    _stamp(base, "20260101_000000_region1", region=RegionPass(index=1, count=2, group="20260101_000000"))
    _stamp(base, "20260102_000000", split="test")

    assert CompletedInference.stamps(base, "train")            == [single]
    assert CompletedInference.incomplete_groups(base, "train") == {"20260101_000000": ([1], 2)}
    assert CompletedInference.incomplete_groups(base, "test")  == {}


def test_a_marker_without_a_split_is_refused(tmp_path):
    """A marker predating split recording raises with the backfill hint instead of being guessed."""
    base  = tmp_path / "inference"
    stamp = base / "20260101_000000"
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"stage": "inference"})

    with pytest.raises(KeyError, match="backfill_completion_markers"):
        CompletedInference.stamps(base, "train")
