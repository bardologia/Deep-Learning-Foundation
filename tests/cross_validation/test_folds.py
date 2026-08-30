"""Tests for the azimuth-blocked fold planner, its guards and the fold naming helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.cross_validation   import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds import FoldConfigFactory, FoldFamily, FoldNaming, FoldPlan, FoldPlanner
from tools.data.regions               import CropRegion, SplitRegions
from tools.runtime.completion         import CompletionMarker


def make_config(n_folds: int, row_start: int = 1000, row_end: int = 2000, guard: int = 0) -> CrossValidationConfig:
    """Returns a cross-validation config with the given fold layout over an row window."""
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=n_folds, row_start=row_start, row_end=row_end, guard=guard)
    return config


def make_planner(n_folds: int, row_start: int = 1000, row_end: int = 2000, col_start: int = 500, col_end: int = 1000, guard: int = 0) -> FoldPlanner:
    """Returns a fold planner over the given row window and col extent."""
    config = make_config(n_folds, row_start, row_end, guard=guard)
    return FoldPlanner(config, col_start=col_start, col_end=col_end)


def all_blocks(plan: FoldPlan) -> list[int]:
    """Returns the sorted block indices used by a fold plan."""
    return sorted([plan.test_block, plan.val_block, *plan.train_blocks])


def row_intervals(regions: list[CropRegion]) -> list[tuple[int, int]]:
    """Returns the sorted (start, end) row intervals of the given crop regions."""
    return sorted((region.row_start, region.row_end) for region in regions)


def test_block_count_equals_n_folds():
    """Verifies the planner produces one azimuth block per fold."""
    planner = make_planner(7)
    assert len(planner.blocks) == 7


def test_blocks_partition_full_extent_contiguously():
    """Verifies consecutive blocks touch and span the full azimuth extent."""
    planner = make_planner(5)
    bounds  = [planner.blocks[0][0]] + [end for _, end in planner.blocks]

    assert bounds[0]  == 1000
    assert bounds[-1] == 2000

    for index in range(len(planner.blocks)):
        assert planner.blocks[index][0] == bounds[index]
        assert planner.blocks[index][1] == bounds[index + 1]


def test_blocks_are_disjoint_and_cover_every_line():
    """Verifies the blocks cover every azimuth line exactly once."""
    planner = make_planner(8)
    covered = []

    for start, end in planner.blocks:
        covered.extend(range(start, end))

    assert covered           == list(range(1000, 2000))
    assert len(set(covered)) == len(covered)


def test_uneven_extent_last_block_absorbs_remainder():
    """Verifies an extent not divisible by the fold count is spread over the blocks with a remainder of at most three lines."""
    planner = make_planner(3, row_start=1000, row_end=2000)

    assert planner.blocks == [(1000, 1333), (1333, 1666), (1666, 2000)]
    sizes = [end - start for start, end in planner.blocks]

    assert max(sizes) - min(sizes) <= 3
    assert sum(sizes)              == 1000


def test_block_sizes_balanced_when_even():
    """Verifies an evenly divisible extent yields equally sized blocks."""
    planner = make_planner(4, row_start=1000, row_end=2000)
    sizes   = [end - start for start, end in planner.blocks]

    assert sizes == [250, 250, 250, 250]


def test_partition_deterministic_for_same_inputs():
    """Verifies the block partition is reproducible for identical inputs."""
    first  = make_planner(6).blocks
    second = make_planner(6).blocks

    assert first == second


def test_plan_train_val_test_disjoint_and_cover_all_blocks():
    """Verifies train, validation and test blocks are disjoint and together cover all blocks."""
    planner = make_planner(5)

    for fold_index in range(5):
        plan   = planner.plan(fold_index)
        blocks = all_blocks(plan)

        assert blocks                  == list(range(5))
        assert plan.test_block         not in plan.train_blocks
        assert plan.val_block          not in plan.train_blocks
        assert plan.test_block         != plan.val_block


def test_plan_test_block_equals_fold_index():
    """Verifies the test block index equals the fold index."""
    planner = make_planner(5)

    for fold_index in range(5):
        assert planner.plan(fold_index).test_block == fold_index


def test_plan_val_block_is_next_cyclic():
    """Verifies the validation block is the next block, wrapping at the last fold."""
    planner = make_planner(5)

    assert planner.plan(0).val_block == 1
    assert planner.plan(4).val_block == 0


def test_plan_train_is_single_region_when_contiguous():
    """Verifies a contiguous training remainder is a single crop region."""
    planner = make_planner(5)
    plan    = planner.plan(0)

    assert isinstance(plan.split_regions.train, CropRegion)
    assert row_intervals(plan.split_regions.regions("train")) == [(1400, 2000)]


def test_plan_train_is_list_of_disjoint_regions_when_split():
    """Verifies a training remainder split by the held-out blocks becomes two disjoint regions."""
    planner = make_planner(5)
    plan    = planner.plan(2)

    assert isinstance(plan.split_regions.train, list)
    assert row_intervals(plan.split_regions.regions("train")) == [(1000, 1400), (1800, 2000)]


def test_plan_regions_carry_full_range_extent():
    """Verifies every split region inherits the full col extent."""
    planner = make_planner(5, col_start=500, col_end=1000)
    plan    = planner.plan(0)

    for name in ("train", "val", "test"):
        for region in plan.split_regions.regions(name):
            assert region.col_start == 500
            assert region.col_end   == 1000


def test_plan_splits_cover_full_rows_without_overlap():
    """Verifies the split regions of each fold cover the row extent without overlap."""
    planner = make_planner(6)

    for fold_index in range(6):
        plan      = planner.plan(fold_index)
        intervals = []

        for name in ("train", "val", "test"):
            intervals.extend(row_intervals(plan.split_regions.regions(name)))

        covered = []
        for start, end in intervals:
            covered.extend(range(start, end))

        assert sorted(covered)   == list(range(1000, 2000))
        assert len(set(covered)) == len(covered)


def test_plans_returns_one_plan_per_fold_in_order():
    """Verifies plans() returns one plan per fold in fold-index order."""
    planner = make_planner(5)
    plans   = planner.plans()

    assert len(plans)                                  == 5
    assert [plan.fold_index for plan in plans]         == list(range(5))
    assert [plan.test_block for plan in plans]         == list(range(5))


def test_plans_deterministic_across_calls():
    """Verifies repeated plans() calls yield identical test regions."""
    planner = make_planner(7)

    first  = [row_intervals(plan.split_regions.regions("test")) for plan in planner.plans()]
    second = [row_intervals(plan.split_regions.regions("test")) for plan in planner.plans()]

    assert first == second


def test_test_blocks_across_folds_tile_the_extent():
    """Verifies the test blocks of all folds tile the azimuth extent."""
    planner   = make_planner(5)
    intervals = []

    for plan in planner.plans():
        intervals.extend(row_intervals(plan.split_regions.regions("test")))

    covered = []
    for start, end in sorted(intervals):
        covered.extend(range(start, end))

    assert covered == list(range(1000, 2000))


def test_guard_trims_internal_boundaries_only():
    """Verifies the guard trims only the boundaries internal to the fold window."""
    planner = make_planner(5, guard=64)
    plan    = planner.plan(0)

    assert row_intervals(plan.split_regions.regions("test"))  == [(1000, 1168)]
    assert row_intervals(plan.split_regions.regions("val"))   == [(1232, 1368)]
    assert row_intervals(plan.split_regions.regions("train")) == [(1432, 2000)]


def test_guard_gap_separates_every_split_pair():
    """Verifies neighbouring split regions are separated by exactly the guard width."""
    planner = make_planner(5, guard=64)

    for plan in planner.plans():
        intervals = []
        for name in ("train", "val", "test"):
            intervals.extend(row_intervals(plan.split_regions.regions(name)))

        for (_, left_end), (right_start, _) in zip(sorted(intervals), sorted(intervals)[1:]):
            assert right_start - left_end == 64


def test_guard_keeps_split_train_runs_trimmed():
    """Verifies both runs of a split training region are trimmed by the guard."""
    planner = make_planner(5, guard=64)
    plan    = planner.plan(2)

    assert row_intervals(plan.split_regions.regions("train")) == [(1000, 1368), (1832, 2000)]


def test_zero_guard_reproduces_block_aligned_regions():
    """Verifies a zero guard leaves the test region aligned with its block."""
    planner = make_planner(5, guard=0)
    plan    = planner.plan(0)

    test_region = plan.split_regions.regions("test")[0]
    assert (test_region.row_start, test_region.row_end) == planner.blocks[0]


def test_default_guard_matches_the_scene_fold_blocks():
    """Verifies the default guard width matches the scene fold blocks."""
    assert FoldConfig().guard == 16


def test_inference_regions_cover_untrimmed_blocks():
    """Verifies inference regions span the untrimmed blocks and contain the guarded split regions."""
    planner = make_planner(5, guard=64)

    for fold_index in range(5):
        plan        = planner.plan(fold_index)
        test_region = planner.inference_region(fold_index, "test")
        val_region  = planner.inference_region(fold_index, "val")

        assert (test_region.row_start, test_region.row_end) == planner.blocks[plan.test_block]
        assert (val_region.row_start,  val_region.row_end)  == planner.blocks[plan.val_block]
        assert test_region.contains(plan.split_regions.regions("test")[0])
        assert val_region.contains(plan.split_regions.regions("val")[0])
        assert (test_region.col_start, test_region.col_end) == (500, 1000)


def test_inference_regions_tile_the_fold_window_without_gaps():
    """Verifies the per-fold test inference regions tile the fold window without gaps."""
    planner = make_planner(5, guard=64)

    spans = sorted(planner.inference_region(fold_index, "test").as_tuple()[:2] for fold_index in range(5))

    assert spans[0][0]  == 1000
    assert spans[-1][1] == 2000
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_inference_region_rejects_unknown_split():
    """Verifies requesting an inference region for the training split is rejected."""
    planner = make_planner(5)

    with pytest.raises(ValueError, match="inference regions exist"):
        planner.inference_region(0, "train")


def test_rejects_odd_or_negative_guard():
    """Verifies odd and negative guard widths are rejected."""
    with pytest.raises(ValueError, match="non-negative even"):
        make_planner(5, guard=63)

    with pytest.raises(ValueError, match="non-negative even"):
        make_planner(5, guard=-2)


def test_rejects_guard_consuming_smallest_block():
    """Verifies a guard wider than the smallest block is rejected."""
    with pytest.raises(ValueError, match="consumes the smallest fold block"):
        make_planner(5, row_start=1000, row_end=1300, guard=64)


def test_rejects_fewer_than_three_folds():
    """Verifies fewer than three folds is rejected."""
    with pytest.raises(ValueError, match="n_folds must be >= 3"):
        make_planner(2)


def test_rejects_extent_smaller_than_n_folds():
    """Verifies an azimuth extent shorter than the fold count is rejected."""
    with pytest.raises(ValueError, match="smaller than n_folds"):
        FoldPlanner(make_config(5, row_start=1000, row_end=1003), col_start=0, col_end=10)


def test_plan_rejects_out_of_range_fold_index():
    """Verifies a fold index outside the valid range is rejected."""
    planner = make_planner(5)

    with pytest.raises(ValueError, match="fold_index must be in"):
        planner.plan(5)

    with pytest.raises(ValueError, match="fold_index must be in"):
        planner.plan(-1)


def test_fold_naming_roundtrip():
    """Verifies fold names and indices convert back and forth."""
    for index in range(12):
        assert FoldNaming.index(FoldNaming.name(index)) == index

    assert FoldNaming.name(3)        == "fold_3"
    assert FoldNaming.index("fold_7") == 7


def test_split_regions_helpers_consistent_with_plan():
    """Verifies the plan's split regions match the planner blocks and use the SplitRegions container."""
    planner = make_planner(5)
    plan    = planner.plan(0)

    test_region = plan.split_regions.regions("test")[0]
    val_region  = plan.split_regions.regions("val")[0]

    assert (test_region.row_start, test_region.row_end) == planner.blocks[0]
    assert (val_region.row_start, val_region.row_end)   == planner.blocks[1]
    assert isinstance(plan.split_regions, SplitRegions)


def factory_config(test_data_dir: Path, n_folds: int = 5, row_start: int = 1000, row_end: int = 1512) -> CrossValidationConfig:
    """Returns a cross-validation config pointing at the synthetic test dataset with the given fold window."""
    config                    = CrossValidationConfig()
    config.paths.dataset_path = test_data_dir
    config.folds              = FoldConfig(n_folds=n_folds, row_start=row_start, row_end=row_end)
    return config


@pytest.mark.dataset
def test_factory_planner_reads_global_crop_range(test_data_dir):
    """Verifies the factory planner inherits the range extent of the dataset global crop."""
    factory = FoldConfigFactory(factory_config(test_data_dir))
    planner = factory.planner()

    crop = factory.global_crop()
    assert planner.col_start == crop.col_start
    assert planner.col_end   == crop.col_end


@pytest.mark.dataset
def test_factory_planner_is_cached(test_data_dir):
    """Verifies the factory caches the planner it builds."""
    factory = FoldConfigFactory(factory_config(test_data_dir))

    assert factory.planner() is factory.planner()


@pytest.mark.dataset
def test_factory_planner_rejects_window_outside_global_crop(test_data_dir):
    """Verifies a fold window reaching outside the dataset global crop is rejected."""
    factory = FoldConfigFactory(factory_config(test_data_dir, row_start=500))

    with pytest.raises(ValueError, match="must lie within the dataset global crop"):
        factory.planner()


@pytest.mark.dataset
def test_factory_fold_inference_config_sets_split_and_region(test_data_dir):
    """Verifies the fold inference config carries the split, run directory and untrimmed region override."""
    factory   = FoldConfigFactory(factory_config(test_data_dir))
    run_dir   = Path("/tmp/cv_run/folds/fold_0")

    inference = factory.fold_inference_config(run_dir, "test", fold_index=0)

    crop = factory.global_crop()
    assert inference.split           == "test"
    assert inference.output_subdir   is None
    assert inference.run_directory   == run_dir
    assert inference.region_override == [1000, 1102, crop.col_start, crop.col_end]


def stamp_inference(base: Path, name: str, split: str) -> Path:
    """Creates a completion-stamped inference directory for the given split and returns it."""
    stamp = base / name
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"stage": "inference", "split": split})
    return stamp


def test_family_inference_base_nests_autoencoders(tmp_path):
    """Verifies autoencoder families nest their inference directories under a family subfolder."""
    run_dir = tmp_path / "fold_0"

    assert FoldFamily.inference_base(run_dir, "backbone")            == run_dir / "inference"
    assert FoldFamily.inference_base(run_dir, "jepa")                == run_dir / "inference"
    assert FoldFamily.inference_base(run_dir, "profile_autoencoder") == run_dir / "inference" / "profile_ae"
    assert FoldFamily.inference_base(run_dir, "image_autoencoder")   == run_dir / "inference" / "image_ae"


def test_family_split_inference_dir_matches_payload_not_name(tmp_path):
    """Verifies the split inference directory is chosen by the stamped payload, not the directory name."""
    run_dir = tmp_path / "fold_0"
    stamp   = stamp_inference(run_dir / "inference", "20260101_000000", "test")
    stamp_inference(run_dir / "inference", "20260102_000000", "val")

    assert FoldFamily.split_inference_dir(run_dir, "backbone", "test")  == stamp
    assert FoldFamily.split_inference_dir(run_dir, "backbone", "train") is None


def test_family_split_inference_dir_picks_latest_and_skips_incomplete(tmp_path):
    """Verifies the newest complete stamp wins and unstamped directories are ignored."""
    run_dir = tmp_path / "fold_0"
    stamp_inference(run_dir / "inference", "20260101_000000", "test")
    latest = stamp_inference(run_dir / "inference", "20260102_000000", "test")
    (run_dir / "inference" / "20260103_000000").mkdir()

    assert FoldFamily.split_inference_dir(run_dir, "backbone", "test") == latest


def test_family_split_inference_dir_missing_base_is_none(tmp_path):
    """Verifies a missing inference base yields no split directory."""
    assert FoldFamily.split_inference_dir(tmp_path / "fold_0", "profile_autoencoder", "test") is None
