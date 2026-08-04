from __future__ import annotations

import dataclasses

import pytest

from configuration.sar.gaussian_config   import GaussianConfig
from configuration.sar.geometry_config   import GeometryConfig
from configuration.sar.processing_config import (
    TomogramConfig,
    ParallelConfig,
    PathConfig,
    ProcessingConfig,
    PreProcessEntryConfig,
)

from tests.configuration._helpers import make_crop


def test_gaussian_config_instantiates():
    cfg = GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0)
    assert cfg.params_per_gaussian == 3
    assert cfg.x_max > cfg.x_min


def test_gaussian_config_asdict_round_trips():
    cfg     = GaussianConfig(n_default_gaussians=3, x_min=0.0, x_max=1.0)
    payload = dataclasses.asdict(cfg)
    assert GaussianConfig(**payload) == cfg


def test_geometry_config_defaults():
    cfg = GeometryConfig()
    assert cfg.wavelength > 0
    assert cfg.slant_range > 0
    assert 0 < cfg.look_angle_deg < 90
    assert isinstance(cfg.baselines, tuple)
    assert cfg.baselines_source   == "dataset"
    assert cfg.baseline_component == "perpendicular"


def test_geometry_config_asdict_round_trips():
    cfg     = GeometryConfig()
    payload = dataclasses.asdict(cfg)
    assert GeometryConfig(**payload) == cfg


def test_geometry_resolved_manual_returns_self(tmp_path):
    cfg      = GeometryConfig(baselines_source="manual")
    resolved = cfg.resolved(tmp_path)
    assert resolved is cfg


def test_geometry_resolved_with_kz_returns_self(tmp_path):
    cfg      = GeometryConfig(kz_values=(0.1, 0.2))
    resolved = cfg.resolved(tmp_path)
    assert resolved is cfg


def test_geometry_resolved_dataset_missing_file_raises(tmp_path):
    cfg = GeometryConfig(baselines_source="dataset")
    with pytest.raises(FileNotFoundError):
        cfg.resolved(tmp_path)


def test_geometry_resolved_invalid_source_raises(tmp_path):
    cfg = GeometryConfig(baselines_source="bogus")
    with pytest.raises(ValueError):
        cfg.resolved(tmp_path)


def test_tomogram_config_defaults():
    cfg = TomogramConfig()
    assert cfg.height_range[1] > cfg.height_range[0]
    assert cfg.max_crop_azimuth_width > 0
    assert cfg.max_amplitude_clip > 0
    assert isinstance(cfg.filter_arguments, dict)
    assert isinstance(cfg.beamforming_arguments, list)


def test_tomogram_config_asdict_round_trips():
    cfg     = TomogramConfig()
    payload = dataclasses.asdict(cfg)
    assert TomogramConfig(**payload) == cfg


def test_tomogram_config_default_factories_are_independent():
    a = TomogramConfig()
    b = TomogramConfig()
    a.filter_arguments["win"].append(99)
    assert b.filter_arguments["win"] == [20, 10]


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_parallel_config_core_budget_valid(effort):
    cfg = ParallelConfig(effort=effort)
    assert cfg.core_budget() >= 1


def test_parallel_config_unknown_effort_raises():
    cfg = ParallelConfig(effort="extreme")
    with pytest.raises(ValueError):
        cfg.core_budget()


def test_parallel_config_resolve_plan_returns_positive_pair():
    cfg              = ParallelConfig(effort="high")
    workers, threads = cfg.resolve_plan(8)
    assert workers >= 1
    assert threads >= 1


def test_parallel_config_effort_fractions_increase():
    fractions = ParallelConfig.EFFORT_FRACTIONS
    assert fractions["low"] < fractions["medium"] < fractions["high"]


def test_path_config_directory_properties():
    cfg = PathConfig(run_subdirectory="run_x")
    assert cfg.run_directory.name       == "run_x"
    assert cfg.data_directory.name      == cfg.data_subdirectory
    assert cfg.metadata_directory.name  == cfg.metadata_subdirectory
    assert cfg.temporary_directory.name == cfg.temporary_subdirectory


def test_path_config_run_directory_defaults_to_main():
    cfg = PathConfig()
    assert cfg.run_directory == cfg.main_directory


def test_processing_config_requires_crop_and_builds_tags():
    cfg = ProcessingConfig(crop=make_crop())
    assert cfg.stack_identifier in cfg.tomogram_tag
    assert cfg.tomogram_output_tag in cfg.tomogram_tag
    assert cfg.parameter_output_tag in cfg.parameter_tag


def test_processing_config_post_init_sets_run_subdirectory():
    cfg = ProcessingConfig(crop=make_crop())
    assert cfg.paths.run_subdirectory is not None
    assert cfg.paths.run_subdirectory.startswith("run_")


def test_processing_config_subconfig_factories():
    cfg = ProcessingConfig(crop=make_crop())
    assert isinstance(cfg.tomogram_config, TomogramConfig)
    assert isinstance(cfg.parallel, ParallelConfig)
    assert isinstance(cfg.paths, PathConfig)


def test_preprocess_entry_config_defaults():
    cfg = PreProcessEntryConfig()
    assert cfg.azimuth_end > cfg.azimuth_start
    assert cfg.range_end > cfg.range_start
    assert cfg.height_range[1] > cfg.height_range[0]
    assert isinstance(cfg.win_list, list)


def test_preprocess_entry_resolve_dataset_name_single_win():
    cfg  = PreProcessEntryConfig()
    name = cfg.resolve_dataset_name([20, 10], "abc")
    assert "w20_10" in name
    assert "abc" in name


def test_preprocess_entry_resolve_dataset_name_explicit_single():
    cfg  = PreProcessEntryConfig(dataset_name="mydata")
    name = cfg.resolve_dataset_name([20, 10], "abc")
    assert name == "mydata"
