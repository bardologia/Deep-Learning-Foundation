from __future__ import annotations

import dataclasses
import json

from types import SimpleNamespace

import numpy as np
import pytest

from configuration.benchmark.general            import BenchmarkConfig
from configuration.dataset.gaussian             import GaussianConfig
from configuration.synthesis                    import SynthesisConfig
from pipelines.backbone.dataset.pipeline        import BackboneDatasetPreparation
from pipelines.processing.synthesis.pipeline    import SyntheticDatasetPipeline
from pipelines.processing.synthesis.writer      import DatasetWriter
from pipelines.shared.config.config_factory     import ConfigFactory
from pipelines.shared.dataset.dataset_spatial   import Layout
from tools.monitoring.logger                    import Logger


TINY = dict(
    global_crop            = (0, 48, 0, 32),
    profile_length         = 32,
    sigma_range            = (4.0, 9.0),
    field_smoothing_pixels = 2.0,
)


def _tiny_config(tmp_path, **overrides) -> SynthesisConfig:
    return SynthesisConfig(output_directory=tmp_path / "tiny", **{**TINY, **overrides})


def _generate(config, logger_stub):
    return SyntheticDatasetPipeline(config, logger=logger_stub).run()


def test_rejects_a_sigma_finer_than_the_z_bin(tmp_path):
    config = _tiny_config(tmp_path, sigma_range=(0.01, 4.0))

    with pytest.raises(ValueError, match="z bin width"):
        config.validate()


def test_rejects_a_phase_rate_count_that_misses_a_secondary(tmp_path):
    config = _tiny_config(tmp_path, channel_phase_rates=(0.01, 0.02))

    with pytest.raises(ValueError, match="channel_phase_rates"):
        config.validate()


def test_rejects_a_primary_that_is_also_a_secondary(tmp_path):
    config = _tiny_config(tmp_path, primary_label="CH01")

    with pytest.raises(ValueError, match="reference channel"):
        config.validate()


def test_generated_layout_lists_every_artifact(tmp_path, logger_stub):
    config = _tiny_config(tmp_path)
    _generate(config, logger_stub)

    layout = json.loads((config.data_directory / "dataset.json").read_text())

    assert layout["global_crop"] == list(config.global_crop)
    assert layout["channel_labels"] == config.channel_labels
    assert set(layout["artifacts"]) == set(DatasetWriter.ARTIFACTS)
    assert all((config.data_directory / name).is_file() for name in layout["artifacts"].values())


def test_generated_state_carries_crop_and_z_range(tmp_path, logger_stub):
    config = _tiny_config(tmp_path)
    _generate(config, logger_stub)

    state = json.loads((config.metadata_directory / "config_state.json").read_text())

    assert state["crop"]["row_start"] == config.global_crop[0]
    assert state["crop"]["col_end"]     == config.global_crop[3]
    assert state["cube_config"]["z_range"] == list(config.z_range)


def test_generated_arrays_have_the_declared_shapes(tmp_path, logger_stub):
    config = _tiny_config(tmp_path)
    _generate(config, logger_stub)

    spatial = (config.row_size, config.col_size)

    assert np.load(config.data_directory / "primary.npy").shape        == spatial
    assert np.load(config.data_directory / "aux_full.npy").shape       == spatial
    assert np.load(config.data_directory / "secondaries.npy").shape    == (config.n_secondaries,) + spatial
    assert np.load(config.data_directory / "derived.npy").shape == (config.n_secondaries,) + spatial
    assert np.load(config.data_directory / "cube_full.npy").shape  == (config.profile_length,) + spatial
    assert np.load(config.parameters_directory / "parameters.npy").shape == (3 * config.k_max,) + spatial


def test_derived_channels_are_the_primary_against_each_secondary(tmp_path, logger_stub):
    config = _tiny_config(tmp_path)
    _generate(config, logger_stub)

    primary     = np.load(config.data_directory / "primary.npy")
    secondaries = np.load(config.data_directory / "secondaries.npy")
    derived     = np.load(config.data_directory / "derived.npy")

    assert np.allclose(derived, primary[None, :, :] * np.conj(secondaries), rtol=1e-5, atol=1e-7)


def test_secondary_response_tracks_the_component_depth(tmp_path, logger_stub):
    config = _tiny_config(tmp_path, global_crop=(0, 128, 0, 96), dropped_slot_fraction=1.0, empty_pixel_fraction=0.0)
    _generate(config, logger_stub)

    secondaries = np.abs(np.load(config.data_directory / "secondaries.npy"))
    parameters  = np.load(config.parameters_directory / "parameters.npy")

    active   = parameters[0] > config.activity_threshold
    strongest = secondaries.argmax(axis=0)[active]
    depths    = parameters[1][active]

    ordered = [float(np.median(depths[strongest == index])) for index in range(config.n_secondaries) if (strongest == index).any()]

    assert len(ordered) >= 2
    assert ordered == sorted(ordered)


def test_generation_is_reproducible_from_the_seed(tmp_path, logger_stub):
    first  = _tiny_config(tmp_path)
    second = dataclasses.replace(first, output_directory=tmp_path / "tiny_again")

    _generate(first,  logger_stub)
    _generate(second, logger_stub)

    for name in ("cube_full.npy", "primary.npy", "secondaries.npy", "derived.npy", "aux_full.npy"):
        assert np.array_equal(np.load(first.data_directory / name), np.load(second.data_directory / name))

    assert np.array_equal(np.load(first.parameters_directory / "parameters.npy"), np.load(second.parameters_directory / "parameters.npy"))


def test_a_different_seed_moves_the_scene(tmp_path, logger_stub):
    first  = _tiny_config(tmp_path)
    second = dataclasses.replace(first, output_directory=tmp_path / "tiny_seed_1", seed=1)

    _generate(first,  logger_stub)
    _generate(second, logger_stub)

    assert not np.array_equal(np.load(first.parameters_directory / "parameters.npy"), np.load(second.parameters_directory / "parameters.npy"))


def test_layout_reads_the_generated_dataset(test_data_dir, params_dir, logger_stub, scene_config):
    layout = Layout(test_data_dir, logger=logger_stub, parameters_path=params_dir / "parameters.npy")

    assert layout.global_crop.as_tuple() == tuple(scene_config.global_crop)
    assert layout.profile_length         == scene_config.profile_length
    assert layout.channel_labels            == scene_config.channel_labels
    assert layout.secondary_indices(("CH02", "CH04")) == [1, 3]


@pytest.mark.dataset
@pytest.mark.slow
def test_default_training_configuration_builds_all_three_splits(test_data_dir, params_dir, scene_config, tmp_path):
    benchmark                       = BenchmarkConfig()
    benchmark.paths.dataset_path    = test_data_dir
    benchmark.paths.parameters_path = params_dir / "parameters.npy"
    benchmark.training.batch_size   = 4
    benchmark.training.num_workers  = 0

    factory        = ConfigFactory(benchmark)
    dataset_config = factory.training_dataset_config()
    trainer_config = SimpleNamespace(gaussian=factory.gaussian_config(), training=SimpleNamespace(resume=False))
    logger         = Logger(log_dir=str(tmp_path / "logs"), name="synthetic_prep", level="ERROR")

    prep = BackboneDatasetPreparation(dataset_config, trainer_config, tmp_path, logger, seed=0)

    (train_loader, val_loader, test_loader), datasets, z_axis, z_len = prep.run()

    assert z_len == scene_config.profile_length
    assert all(len(datasets[split]) > 0 for split in ("train", "val", "test"))

    inputs, targets = next(iter(train_loader))

    assert inputs.shape[0]  == 4
    assert inputs.shape[2:] == tuple(benchmark.training.patch_size)
    assert targets.shape[1] == 3 * GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy").n_default_gaussians
    assert inputs.isfinite().all()
    assert targets.isfinite().all()
