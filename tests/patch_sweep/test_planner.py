from __future__ import annotations

from pathlib import Path

import pytest

from configuration.patch_sweep     import PatchSweepConfig
from pipelines.patch_sweep.planner import ArchitecturePatchStep, PatchSweepPlanner


def make_base(tmp_path: Path, names: list[str]) -> Path:
    base = tmp_path / "datasets"
    base.mkdir(exist_ok=True)
    for name in names:
        (base / name / "data").mkdir(parents=True)
    return base


def make_planner(tmp_path: Path, datasets: list[str], selection: list[str] | None = None, **patch_overrides) -> PatchSweepPlanner:
    config                   = PatchSweepConfig()
    config.dataset_base_path = make_base(tmp_path, datasets)
    config.dataset_filter    = selection or []
    for key, value in patch_overrides.items():
        setattr(config.patch, key, value)
    return PatchSweepPlanner(config)


def test_sweep_default_pins_no_dataset_name():
    assert PatchSweepConfig().dataset_filter == []


def test_step_derives_from_the_resunet_feature_pyramid():
    assert ArchitecturePatchStep("resunet", "conv", {}).resolve() == 16


def test_step_scales_with_the_pyramid_depth():
    assert ArchitecturePatchStep("resunet", "conv", {"features": [32, 64]}).resolve() == 4


def test_sweep_default_narrows_the_pyramid_to_three_levels_for_step_eight():
    config = PatchSweepConfig()

    assert len(config.model_overrides["features"]) == 3
    assert ArchitecturePatchStep(config.backbone_name, config.backbone_head, config.model_overrides).resolve() == 8


def test_step_rejects_backbones_outside_the_verified_unet_family():
    with pytest.raises(ValueError, match="patch.step"):
        ArchitecturePatchStep("pixel_mlp", "conv", {}).resolve()


def test_explicit_step_overrides_the_architecture(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], step=32, maximum=(96, 64))

    assert planner.patch_sizes() == ([32, 64, 96], [32, 64])


def test_sizes_run_from_one_step_to_maximum_per_axis(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(128, 64))

    azimuth_sizes, range_sizes = planner.patch_sizes()

    assert azimuth_sizes == [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    assert range_sizes   == [8, 16, 24, 32, 40, 48, 56, 64]


def test_minimum_must_be_admissible_on_the_azimuth_axis(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], minimum=(20, 0))

    with pytest.raises(ValueError, match="azimuth=20 is not a multiple"):
        planner.patch_sizes()


def test_minimum_must_be_admissible_on_the_range_axis(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], minimum=(0, 20))

    with pytest.raises(ValueError, match="range=20 is not a multiple"):
        planner.patch_sizes()


def test_single_combination_grid_is_rejected(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(8, 8))

    with pytest.raises(ValueError, match="at least 2"):
        planner.patch_sizes()


def test_single_axis_sweep_is_admissible(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(8, 16))

    assert planner.patch_sizes() == ([8], [8, 16])


def test_datasets_must_be_present(tmp_path):
    with pytest.raises(ValueError, match="No datasets to sweep"):
        make_planner(tmp_path, [])


def test_empty_filter_sweeps_every_dataset(tmp_path):
    planner = make_planner(tmp_path, ["w20_10", "w20_20", "w20_30"])

    assert [dataset.name for dataset in planner.datasets] == ["w20_10", "w20_20", "w20_30"]


def test_filter_selects_a_subset_of_the_base(tmp_path):
    planner = make_planner(tmp_path, ["w20_10", "w20_20", "w20_30"], selection=["w20_30", "w20_10"])

    assert [dataset.name for dataset in planner.datasets] == ["w20_10", "w20_30"]
    assert sorted({unit.dataset for unit in planner.units()}) == ["w20_10", "w20_30"]


def test_filter_rejects_unknown_dataset_names(tmp_path):
    with pytest.raises(NotADirectoryError, match="without a data/ directory"):
        make_planner(tmp_path, ["w20_10"], selection=["w99_99"])


def test_filter_names_must_be_unique(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        make_planner(tmp_path, ["w20_10"], selection=["w20_10", "w20_10"])


def test_parameters_template_must_live_inside_the_dataset(tmp_path):
    config                       = PatchSweepConfig()
    config.dataset_base_path     = make_base(tmp_path, ["w20_10"])
    config.dataset_filter        = []
    config.paths.parameters_path = Path("/elsewhere/parameters.npy")

    with pytest.raises(ValueError, match="re-roots"):
        PatchSweepPlanner(config)


def test_parameters_are_rerooted_onto_every_dataset(tmp_path):
    planner  = make_planner(tmp_path, ["w20_10", "w20_20"], maximum=(16, 16))
    template = planner.parameters_template()

    for unit in planner.units():
        assert unit.parameters_path == unit.dataset_path / template
        assert str(unit.parameters_path).startswith(str(unit.dataset_path))


def test_units_cover_the_full_cross_product(tmp_path):
    planner = make_planner(tmp_path, ["w20_10", "w20_20"], maximum=(64, 16))
    units   = planner.units()

    assert len(units) == 2 * 8 * 2
    assert sorted({unit.dataset for unit in units}) == ["w20_10", "w20_20"]
    assert {unit.name for unit in units} == {f"{d}-p{a:03d}x{r:03d}" for d in ("w20_10", "w20_20") for a in range(8, 65, 8) for r in (8, 16)}


def test_units_follow_the_sorted_dataset_names(tmp_path):
    planner = make_planner(tmp_path, ["w20_20", "w20_10"], maximum=(16, 16))

    assert [unit.dataset for unit in planner.units()] == ["w20_10"] * 4 + ["w20_20"] * 4


def test_constant_pixel_budget_rescales_the_batch(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(128, 64))
    by_size = {unit.patch_size: unit.batch_size for unit in planner.units()}

    reference = planner.config.training.batch_size * planner.config.training.patch_size[0] * planner.config.training.patch_size[1]

    assert by_size[(16, 16)]  == reference // (16 * 16)
    assert by_size[(128, 64)] == reference // (128 * 64)
    assert all(batch == max(1, reference // (size[0] * size[1])) for size, batch in by_size.items())


def test_fixed_batch_when_the_budget_is_disabled(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], constant_pixel_budget=False)

    assert {unit.batch_size for unit in planner.units()} == {planner.config.training.batch_size}


def test_constant_pixel_budget_keeps_the_lr_scale_constant(tmp_path):
    planner    = make_planner(tmp_path, ["w20_10"], maximum=(128, 64))
    configured = planner.config.training.batch_size / planner.config.training.lr_reference_batch_size

    for unit in planner.units():
        assert unit.batch_size / unit.lr_reference_batch_size == pytest.approx(configured)


def test_lr_reference_rescales_with_the_pixel_budget(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(128, 64))
    by_size = {unit.patch_size: unit.lr_reference_batch_size for unit in planner.units()}

    reference = planner.config.training.lr_reference_batch_size * planner.config.training.patch_size[0] * planner.config.training.patch_size[1]

    assert by_size[(16, 16)]  == reference // (16 * 16)
    assert by_size[(128, 64)] == reference // (128 * 64)
    assert all(lr_reference == max(1, reference // (size[0] * size[1])) for size, lr_reference in by_size.items())


def test_lr_reference_untouched_when_the_budget_is_disabled(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"], constant_pixel_budget=False)

    assert {unit.lr_reference_batch_size for unit in planner.units()} == {planner.config.training.lr_reference_batch_size}


def test_summary_reports_the_seed_axis_and_both_grids(tmp_path):
    planner = make_planner(tmp_path, ["w20_10", "w20_20"], maximum=(16, 16))
    summary = planner.summary()

    assert summary["Datasets"]      == ["w20_10", "w20_20"]
    assert summary["Azimuth sizes"] == [8, 16]
    assert summary["Range sizes"]   == [8, 16]
    assert summary["Units"]         == 8
    assert summary["Seeds"]         == [0, 1, 2, 3, 4]


def test_unit_lookup_is_loud_for_unknown_names(tmp_path):
    planner = make_planner(tmp_path, ["w20_10"])

    with pytest.raises(KeyError, match="w99-p016x016"):
        planner.unit("w99-p016x016")
