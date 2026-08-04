from __future__ import annotations

from pathlib import Path

import pytest

from configuration.cross_validation   import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds import FoldConfigFactory, FoldNaming, FoldPlan, FoldPlanner
from tools.data.regions               import CropRegion, SplitRegions


def make_config(n_folds: int, azimuth_start: int = 1000, azimuth_end: int = 2000, guard: int = 0) -> CrossValidationConfig:
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=n_folds, azimuth_start=azimuth_start, azimuth_end=azimuth_end, guard=guard)
    return config


def make_planner(n_folds: int, azimuth_start: int = 1000, azimuth_end: int = 2000, range_start: int = 500, range_end: int = 1000, guard: int = 0) -> FoldPlanner:
    config = make_config(n_folds, azimuth_start, azimuth_end, guard=guard)
    return FoldPlanner(config, range_start=range_start, range_end=range_end)


def all_blocks(plan: FoldPlan) -> list[int]:
    return sorted([plan.test_block, plan.val_block, *plan.train_blocks])


def az_intervals(regions: list[CropRegion]) -> list[tuple[int, int]]:
    return sorted((region.azimuth_start, region.azimuth_end) for region in regions)


def test_block_count_equals_n_folds():
    planner = make_planner(7)
    assert len(planner.blocks) == 7


def test_blocks_partition_full_extent_contiguously():
    planner = make_planner(5)
    bounds  = [planner.blocks[0][0]] + [end for _, end in planner.blocks]

    assert bounds[0]  == 1000
    assert bounds[-1] == 2000

    for index in range(len(planner.blocks)):
        assert planner.blocks[index][0] == bounds[index]
        assert planner.blocks[index][1] == bounds[index + 1]


def test_blocks_are_disjoint_and_cover_every_line():
    planner = make_planner(8)
    covered = []

    for start, end in planner.blocks:
        covered.extend(range(start, end))

    assert covered           == list(range(1000, 2000))
    assert len(set(covered)) == len(covered)


def test_uneven_extent_last_block_absorbs_remainder():
    planner = make_planner(3, azimuth_start=1000, azimuth_end=2000)

    assert planner.blocks == [(1000, 1333), (1333, 1666), (1666, 2000)]
    sizes = [end - start for start, end in planner.blocks]

    assert max(sizes) - min(sizes) <= 3
    assert sum(sizes)              == 1000


def test_block_sizes_balanced_when_even():
    planner = make_planner(4, azimuth_start=1000, azimuth_end=2000)
    sizes   = [end - start for start, end in planner.blocks]

    assert sizes == [250, 250, 250, 250]


def test_partition_deterministic_for_same_inputs():
    first  = make_planner(6).blocks
    second = make_planner(6).blocks

    assert first == second


def test_plan_train_val_test_disjoint_and_cover_all_blocks():
    planner = make_planner(5)

    for fold_index in range(5):
        plan   = planner.plan(fold_index)
        blocks = all_blocks(plan)

        assert blocks                  == list(range(5))
        assert plan.test_block         not in plan.train_blocks
        assert plan.val_block          not in plan.train_blocks
        assert plan.test_block         != plan.val_block


def test_plan_test_block_equals_fold_index():
    planner = make_planner(5)

    for fold_index in range(5):
        assert planner.plan(fold_index).test_block == fold_index


def test_plan_val_block_is_next_cyclic():
    planner = make_planner(5)

    assert planner.plan(0).val_block == 1
    assert planner.plan(4).val_block == 0


def test_plan_train_is_single_region_when_contiguous():
    planner = make_planner(5)
    plan    = planner.plan(0)

    assert isinstance(plan.split_regions.train, CropRegion)
    assert az_intervals(plan.split_regions.regions("train")) == [(1400, 2000)]


def test_plan_train_is_list_of_disjoint_regions_when_split():
    planner = make_planner(5)
    plan    = planner.plan(2)

    assert isinstance(plan.split_regions.train, list)
    assert az_intervals(plan.split_regions.regions("train")) == [(1000, 1400), (1800, 2000)]


def test_plan_regions_carry_full_range_extent():
    planner = make_planner(5, range_start=500, range_end=1000)
    plan    = planner.plan(0)

    for name in ("train", "val", "test"):
        for region in plan.split_regions.regions(name):
            assert region.range_start == 500
            assert region.range_end   == 1000


def test_plan_splits_cover_full_azimuth_without_overlap():
    planner = make_planner(6)

    for fold_index in range(6):
        plan      = planner.plan(fold_index)
        intervals = []

        for name in ("train", "val", "test"):
            intervals.extend(az_intervals(plan.split_regions.regions(name)))

        covered = []
        for start, end in intervals:
            covered.extend(range(start, end))

        assert sorted(covered)   == list(range(1000, 2000))
        assert len(set(covered)) == len(covered)


def test_plans_returns_one_plan_per_fold_in_order():
    planner = make_planner(5)
    plans   = planner.plans()

    assert len(plans)                                  == 5
    assert [plan.fold_index for plan in plans]         == list(range(5))
    assert [plan.test_block for plan in plans]         == list(range(5))


def test_plans_deterministic_across_calls():
    planner = make_planner(7)

    first  = [az_intervals(plan.split_regions.regions("test")) for plan in planner.plans()]
    second = [az_intervals(plan.split_regions.regions("test")) for plan in planner.plans()]

    assert first == second


def test_test_blocks_across_folds_tile_the_extent():
    planner   = make_planner(5)
    intervals = []

    for plan in planner.plans():
        intervals.extend(az_intervals(plan.split_regions.regions("test")))

    covered = []
    for start, end in sorted(intervals):
        covered.extend(range(start, end))

    assert covered == list(range(1000, 2000))


def test_guard_trims_internal_boundaries_only():
    planner = make_planner(5, guard=64)
    plan    = planner.plan(0)

    assert az_intervals(plan.split_regions.regions("test"))  == [(1000, 1168)]
    assert az_intervals(plan.split_regions.regions("val"))   == [(1232, 1368)]
    assert az_intervals(plan.split_regions.regions("train")) == [(1432, 2000)]


def test_guard_gap_separates_every_split_pair():
    planner = make_planner(5, guard=64)

    for plan in planner.plans():
        intervals = []
        for name in ("train", "val", "test"):
            intervals.extend(az_intervals(plan.split_regions.regions(name)))

        for (_, left_end), (right_start, _) in zip(sorted(intervals), sorted(intervals)[1:]):
            assert right_start - left_end == 64


def test_guard_keeps_split_train_runs_trimmed():
    planner = make_planner(5, guard=64)
    plan    = planner.plan(2)

    assert az_intervals(plan.split_regions.regions("train")) == [(1000, 1368), (1832, 2000)]


def test_zero_guard_reproduces_block_aligned_regions():
    planner = make_planner(5, guard=0)
    plan    = planner.plan(0)

    test_region = plan.split_regions.regions("test")[0]
    assert (test_region.azimuth_start, test_region.azimuth_end) == planner.blocks[0]


def test_default_guard_is_64():
    assert FoldConfig().guard == 64


def test_rejects_odd_or_negative_guard():
    with pytest.raises(ValueError, match="non-negative even"):
        make_planner(5, guard=63)

    with pytest.raises(ValueError, match="non-negative even"):
        make_planner(5, guard=-2)


def test_rejects_guard_consuming_smallest_block():
    with pytest.raises(ValueError, match="consumes the smallest fold block"):
        make_planner(5, azimuth_start=1000, azimuth_end=1300, guard=64)


def test_rejects_fewer_than_three_folds():
    with pytest.raises(ValueError, match="n_folds must be >= 3"):
        make_planner(2)


def test_rejects_extent_smaller_than_n_folds():
    with pytest.raises(ValueError, match="smaller than n_folds"):
        FoldPlanner(make_config(5, azimuth_start=1000, azimuth_end=1003), range_start=0, range_end=10)


def test_plan_rejects_out_of_range_fold_index():
    planner = make_planner(5)

    with pytest.raises(ValueError, match="fold_index must be in"):
        planner.plan(5)

    with pytest.raises(ValueError, match="fold_index must be in"):
        planner.plan(-1)


def test_fold_naming_roundtrip():
    for index in range(12):
        assert FoldNaming.index(FoldNaming.name(index)) == index

    assert FoldNaming.name(3)        == "fold_3"
    assert FoldNaming.index("fold_7") == 7


def test_split_regions_helpers_consistent_with_plan():
    planner = make_planner(5)
    plan    = planner.plan(0)

    test_region = plan.split_regions.regions("test")[0]
    val_region  = plan.split_regions.regions("val")[0]

    assert (test_region.azimuth_start, test_region.azimuth_end) == planner.blocks[0]
    assert (val_region.azimuth_start, val_region.azimuth_end)   == planner.blocks[1]
    assert isinstance(plan.split_regions, SplitRegions)


def factory_config(test_data_dir: Path, n_folds: int = 5, azimuth_start: int = 1000, azimuth_end: int = 2000) -> CrossValidationConfig:
    config                    = CrossValidationConfig()
    config.paths.dataset_path = test_data_dir
    config.folds              = FoldConfig(n_folds=n_folds, azimuth_start=azimuth_start, azimuth_end=azimuth_end)
    return config


@pytest.mark.real_data
def test_factory_planner_reads_global_crop_range(test_data_dir):
    factory = FoldConfigFactory(factory_config(test_data_dir))
    planner = factory.planner()

    crop = factory.global_crop()
    assert planner.range_start == crop.range_start
    assert planner.range_end   == crop.range_end


@pytest.mark.real_data
def test_factory_planner_is_cached(test_data_dir):
    factory = FoldConfigFactory(factory_config(test_data_dir))

    assert factory.planner() is factory.planner()


@pytest.mark.real_data
def test_factory_planner_rejects_window_outside_global_crop(test_data_dir):
    factory = FoldConfigFactory(factory_config(test_data_dir, azimuth_start=500))

    with pytest.raises(ValueError, match="must lie within the dataset global crop"):
        factory.planner()


@pytest.mark.real_data
def test_factory_fold_inference_config_sets_split_and_subdir(test_data_dir):
    factory   = FoldConfigFactory(factory_config(test_data_dir))
    run_dir   = Path("/tmp/cv_run/folds/fold_0")

    inference = factory.fold_inference_config(run_dir, "test")

    assert inference.split         == "test"
    assert inference.output_subdir == "test"
    assert inference.run_directory == run_dir
