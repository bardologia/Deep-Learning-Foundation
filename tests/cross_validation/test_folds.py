
from __future__ import annotations

from pathlib import Path

import pytest

from configuration.cross_validation      import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds    import FoldConfigFactory, FoldNaming, FoldPlan, FoldPlanner
from pipelines.shared.inference.metadata import CompletedInference
from tools.data.regions                  import CropRegion, SplitRegions
from tools.runtime.completion            import CompletionMarker


def make_planner(n_folds: int, rows: int = 1000, cols: int = 500, guard: int = 0) -> FoldPlanner:
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=n_folds, guard=guard)
    return FoldPlanner(config, rows=rows, cols=cols)


def all_blocks(plan: FoldPlan) -> list[int]:
    return sorted([plan.test_block, plan.val_block, *plan.train_blocks])


def row_intervals(regions: list[CropRegion]) -> list[tuple[int, int]]:
    return sorted((region.row_start, region.row_end) for region in regions)


def test_block_count_equals_n_folds():
    planner = make_planner(7)
    assert len(planner.blocks) == 7


def test_blocks_partition_full_extent_contiguously():
    planner = make_planner(5)
    bounds  = [planner.blocks[0][0]] + [end for _, end in planner.blocks]

    assert bounds[0]  == 0
    assert bounds[-1] == 1000

    for index in range(len(planner.blocks)):
        assert planner.blocks[index][0] == bounds[index]
        assert planner.blocks[index][1] == bounds[index + 1]


def test_blocks_are_disjoint_and_cover_every_line():
    planner = make_planner(8)
    covered = []

    for start, end in planner.blocks:
        covered.extend(range(start, end))

    assert covered           == list(range(1000))
    assert len(set(covered)) == len(covered)


def test_uneven_extent_last_block_absorbs_remainder():
    planner = make_planner(3)

    assert planner.blocks == [(0, 333), (333, 666), (666, 1000)]
    sizes = [end - start for start, end in planner.blocks]

    assert max(sizes) - min(sizes) <= 3
    assert sum(sizes)              == 1000


def test_block_sizes_balanced_when_even():
    planner = make_planner(4)
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
    assert row_intervals(plan.split_regions.regions("train")) == [(400, 1000)]


def test_plan_train_is_list_of_disjoint_regions_when_split():
    planner = make_planner(5)
    plan    = planner.plan(2)

    assert isinstance(plan.split_regions.train, list)
    assert row_intervals(plan.split_regions.regions("train")) == [(0, 400), (800, 1000)]


def test_plan_regions_carry_every_column():
    planner = make_planner(5, cols=500)
    plan    = planner.plan(0)

    for name in ("train", "val", "test"):
        for region in plan.split_regions.regions(name):
            assert region.col_start == 0
            assert region.col_end   == 500


def test_plan_splits_cover_full_rows_without_overlap():
    planner = make_planner(6)

    for fold_index in range(6):
        plan      = planner.plan(fold_index)
        intervals = []

        for name in ("train", "val", "test"):
            intervals.extend(row_intervals(plan.split_regions.regions(name)))

        covered = []
        for start, end in intervals:
            covered.extend(range(start, end))

        assert sorted(covered)   == list(range(1000))
        assert len(set(covered)) == len(covered)


def test_plans_returns_one_plan_per_fold_in_order():
    planner = make_planner(5)
    plans   = planner.plans()

    assert len(plans)                                  == 5
    assert [plan.fold_index for plan in plans]         == list(range(5))
    assert [plan.test_block for plan in plans]         == list(range(5))


def test_plans_deterministic_across_calls():
    planner = make_planner(7)

    first  = [row_intervals(plan.split_regions.regions("test")) for plan in planner.plans()]
    second = [row_intervals(plan.split_regions.regions("test")) for plan in planner.plans()]

    assert first == second


def test_test_blocks_across_folds_tile_the_extent():
    planner   = make_planner(5)
    intervals = []

    for plan in planner.plans():
        intervals.extend(row_intervals(plan.split_regions.regions("test")))

    covered = []
    for start, end in sorted(intervals):
        covered.extend(range(start, end))

    assert covered == list(range(1000))


def test_guard_trims_internal_boundaries_only():
    planner = make_planner(5, guard=64)
    plan    = planner.plan(0)

    assert row_intervals(plan.split_regions.regions("test"))  == [(0, 168)]
    assert row_intervals(plan.split_regions.regions("val"))   == [(232, 368)]
    assert row_intervals(plan.split_regions.regions("train")) == [(432, 1000)]


def test_guard_gap_separates_every_split_pair():
    planner = make_planner(5, guard=64)

    for plan in planner.plans():
        intervals = []
        for name in ("train", "val", "test"):
            intervals.extend(row_intervals(plan.split_regions.regions(name)))

        for (_, left_end), (right_start, _) in zip(sorted(intervals), sorted(intervals)[1:]):
            assert right_start - left_end == 64


def test_guard_keeps_split_train_runs_trimmed():
    planner = make_planner(5, guard=64)
    plan    = planner.plan(2)

    assert row_intervals(plan.split_regions.regions("train")) == [(0, 368), (832, 1000)]


def test_zero_guard_reproduces_block_aligned_regions():
    planner = make_planner(5, guard=0)
    plan    = planner.plan(0)

    test_region = plan.split_regions.regions("test")[0]
    assert (test_region.row_start, test_region.row_end) == planner.blocks[0]


def test_default_guard_is_sixteen_rows():
    assert FoldConfig().guard == 16


def test_inference_regions_cover_untrimmed_blocks():
    planner = make_planner(5, guard=64)

    for fold_index in range(5):
        plan        = planner.plan(fold_index)
        test_region = planner.inference_region(fold_index, "test")
        val_region  = planner.inference_region(fold_index, "val")

        assert (test_region.row_start, test_region.row_end) == planner.blocks[plan.test_block]
        assert (val_region.row_start,  val_region.row_end)  == planner.blocks[plan.val_block]
        assert test_region.contains(plan.split_regions.regions("test")[0])
        assert val_region.contains(plan.split_regions.regions("val")[0])
        assert (test_region.col_start, test_region.col_end) == (0, 500)


def test_inference_regions_tile_the_fold_window_without_gaps():
    planner = make_planner(5, guard=64)

    spans = sorted(planner.inference_region(fold_index, "test").as_tuple()[:2] for fold_index in range(5))

    assert spans[0][0]  == 0
    assert spans[-1][1] == 1000
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


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

    assert (test_region.row_start, test_region.row_end) == planner.blocks[0]
    assert (val_region.row_start, val_region.row_end)   == planner.blocks[1]
    assert isinstance(plan.split_regions, SplitRegions)


def factory_config(test_data_dir: Path, n_folds: int = 5) -> CrossValidationConfig:
    config                    = CrossValidationConfig()
    config.paths.dataset_path = test_data_dir
    config.folds              = FoldConfig(n_folds=n_folds)
    return config


@pytest.mark.dataset
def test_factory_planner_spans_the_whole_dataset_grid(test_data_dir, dataset_json):
    planner = FoldConfigFactory(factory_config(test_data_dir)).planner()

    assert planner.rows         == dataset_json["rows"]
    assert planner.cols         == dataset_json["cols"]
    assert planner.blocks[-1][1] == dataset_json["rows"]


@pytest.mark.dataset
def test_factory_planner_is_cached(test_data_dir):
    factory = FoldConfigFactory(factory_config(test_data_dir))

    assert factory.planner() is factory.planner()


@pytest.mark.dataset
def test_factory_fold_inference_config_sets_split_and_region(test_data_dir, dataset_json):
    factory   = FoldConfigFactory(factory_config(test_data_dir))
    run_dir   = Path("/tmp/cv_run/folds/fold_0")

    inference = factory.fold_inference_config(run_dir, "test", fold_index=0)

    assert inference.split           == "test"
    assert inference.output_subdir   is None
    assert inference.run_directory   == run_dir
    assert inference.region_override == [0, dataset_json["rows"] // 5, 0, dataset_json["cols"]]


def stamp_inference(base: Path, name: str, split: str) -> Path:
    stamp = base / name
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"stage": "inference", "split": split})
    return stamp


def test_split_inference_dir_matches_payload_not_name(tmp_path):
    run_dir = tmp_path / "fold_0"
    stamp   = stamp_inference(run_dir / "inference", "20260101_000000", "test")
    stamp_inference(run_dir / "inference", "20260102_000000", "val")

    assert CompletedInference.latest(run_dir / "inference", "test")  == stamp
    assert CompletedInference.latest(run_dir / "inference", "train") is None


def test_split_inference_dir_picks_latest_and_skips_incomplete(tmp_path):
    run_dir = tmp_path / "fold_0"
    stamp_inference(run_dir / "inference", "20260101_000000", "test")
    latest = stamp_inference(run_dir / "inference", "20260102_000000", "test")
    (run_dir / "inference" / "20260103_000000").mkdir()

    assert CompletedInference.latest(run_dir / "inference", "test") == latest


def test_fold_naming_recognises_tagged_and_bare_fold_names():
    assert FoldNaming.is_fold("resunet-mse-hv_fold_0")
    assert FoldNaming.is_fold("fold_12")
    assert not FoldNaming.is_fold("fold_0_seed1")
    assert not FoldNaming.is_fold("_comparison")
    assert not FoldNaming.is_fold("training_results.json")
