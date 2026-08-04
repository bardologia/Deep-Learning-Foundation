from __future__ import annotations

import json

import numpy as np
import pytest

from pathlib import Path

from configuration.param_extraction                import ChannelOrder, InjectExternalParamsEntryConfig
from configuration.sar.gaussian_config             import GaussianConfig
from pipelines.processing.external_params.assembly import ChannelMapper, ParameterValidator, TargetAssembler
from pipelines.processing.external_params.injection import ExternalParamsInjectionPipeline
from pipelines.processing.external_params.sources   import ExternalSourceResolver, WindowParser
from pipelines.processing.param_extraction.io       import ParameterRunMeta
from tools.data.rat                                 import RatCube, RatHeaderReader
from tools.data.regions                             import CropRegion
from tools.monitoring.logger                        import Logger


HEIGHT_RANGE = (-20.0, 80.0)
SOURCE_ORDER = "mu_sigma_amp"


@pytest.fixture
def logger(tmp_path):
    log = Logger(log_dir=str(tmp_path / "logs"), name="external_params_test")
    yield log
    log.close()


def write_rat(path: Path, array: np.ndarray) -> Path:
    header = bytearray(RatHeaderReader.HEADER_SIZE)

    header[0:4] = RatHeaderReader.MAGIC
    header[4:8] = np.float32(2.0).tobytes()
    header[8:12] = np.int32(array.ndim).tobytes()

    for axis, size in enumerate(reversed(array.shape)):
        offset = RatHeaderReader.SHAPE_OFFSET + 4 * axis
        header[offset:offset + 4] = np.int32(size).tobytes()

    header[RatHeaderReader.VAR_OFFSET:RatHeaderReader.VAR_OFFSET + 4] = np.int32(5).tobytes()

    with open(path, "wb") as handle:
        handle.write(bytes(header))
        handle.write(np.ascontiguousarray(array, dtype="<f8").tobytes())

    return path


def source_cube(n_gaussians: int, azimuth: int, range_: int, seed: int) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    cube  = np.zeros((3 * n_gaussians, azimuth, range_), dtype=np.float64)

    for gaussian in range(n_gaussians):
        cube[3 * gaussian + 0] = rng.uniform(-15.0 + 20.0 * gaussian, 5.0 + 20.0 * gaussian, size=(azimuth, range_))
        cube[3 * gaussian + 1] = rng.uniform(0.5, 5.0, size=(azimuth, range_))
        cube[3 * gaussian + 2] = rng.uniform(0.5, 10.0, size=(azimuth, range_))

    return cube


def build_dataset(root: Path, crop: CropRegion, height_bins: int = 64) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)

    (root / "data" / "dataset.json").write_text(json.dumps({
        "global_crop"  : list(crop.as_tuple()),
        "dataset_type" : "FSAR",
        "tomogram_tag" : "test",
        "pass_labels"  : ["FL01_PS02"],
        "artifacts"    : {"tomogram_full": "tomogram_full.npy"},
    }))

    (root / "meta" / "config_state.json").write_text(json.dumps({
        "tomogram_config": {"height_range": list(HEIGHT_RANGE)},
    }))

    np.save(root / "data" / "tomogram_full.npy", np.zeros((height_bins, crop.azimuth_size, crop.range_size), dtype=np.float32))

    return root


def entry_config(base: Path, sources: list, **overrides) -> InjectExternalParamsEntryConfig:
    config = InjectExternalParamsEntryConfig(
        dataset_base_path = base,
        source_files      = [str(path) for path in sources],
        source_order      = SOURCE_ORDER,
        output_suffix     = "external",
    )

    for name, value in overrides.items():
        setattr(config, name, value)

    return config


def test_rat_header_reads_shape_dtype_and_offset(tmp_path):
    cube = source_cube(n_gaussians=2, azimuth=7, range_=5, seed=0)
    path = write_rat(tmp_path / "params_0a7a0a5_site.rat", cube)

    reader = RatCube(path)

    assert reader.shape == (6, 7, 5)
    assert reader.spec.dtype == np.dtype("<f8")
    assert np.allclose(reader.channel_window(3, slice(0, 7), slice(0, 5)), cube[3])


def test_rat_header_rejects_a_payload_that_contradicts_the_header(tmp_path):
    path = write_rat(tmp_path / "params_0a7a0a5_site.rat", source_cube(2, 7, 5, seed=0))

    with open(path, "ab") as handle:
        handle.write(b"\x00" * 8)

    with pytest.raises(ValueError, match="header parse and the payload disagree"):
        RatCube(path)


def test_window_parser_reads_the_identifier_from_the_filename():
    window = WindowParser().from_filename(Path("paramsfull_12400a16000a500a4000_flaca_GusTrial0.rat"))

    assert window.as_tuple() == (12400, 16000, 500, 4000)


def test_window_parser_rejects_a_filename_without_an_identifier():
    with pytest.raises(ValueError, match="exactly one"):
        WindowParser().from_filename(Path("paramsfull_flaca.rat"))


def test_channel_order_maps_every_permutation():
    assert ChannelOrder.positions("amp_mu_sigma") == (0, 1, 2)
    assert ChannelOrder.positions("mu_sigma_amp") == (2, 0, 1)

    with pytest.raises(ValueError, match="Unknown channel order"):
        ChannelOrder.positions("amp_amp_sigma")


def test_channel_mapper_pairs_source_channels_onto_the_amp_mu_sigma_convention():
    mapper = ChannelMapper(SOURCE_ORDER, n_source_gaussians=2, k_slots=0)

    assert mapper.n_slots          == 2
    assert mapper.n_target_channels == 6
    assert mapper.channel_pairs()  == [(2, 0), (0, 1), (1, 2), (5, 3), (3, 4), (4, 5)]


def test_channel_mapper_refuses_to_drop_fitted_gaussians():
    with pytest.raises(ValueError, match="smaller than"):
        ChannelMapper(SOURCE_ORDER, n_source_gaussians=3, k_slots=2)


def test_source_resolver_rejects_a_file_whose_grid_contradicts_its_window(tmp_path, logger):
    path = write_rat(tmp_path / "params_0a7a0a5_site.rat", source_cube(2, 9, 5, seed=0))

    with pytest.raises(ValueError, match="declared extent disagree"):
        ExternalSourceResolver([path], [], logger).resolve()


def test_source_resolver_takes_an_explicit_window_over_the_filename(tmp_path, logger):
    path = write_rat(tmp_path / "params_site.rat", source_cube(2, 7, 5, seed=0))

    sources = ExternalSourceResolver([path], ["100a107a20a25"], logger).resolve()

    assert sources[0].window.as_tuple() == (100, 107, 20, 25)
    assert sources[0].n_gaussians       == 2


def test_assembler_stitches_two_windows_into_one_crop(tmp_path, logger):
    left  = write_rat(tmp_path / "params_0a10a0a8_site.rat",  source_cube(2, 10, 8, seed=1))
    right = write_rat(tmp_path / "params_10a20a0a8_site.rat", source_cube(2, 10, 8, seed=2))

    sources    = ExternalSourceResolver([left, right], [], logger).resolve()
    mapper     = ChannelMapper(SOURCE_ORDER, sources[0].n_gaussians, k_slots=0)
    parameters = TargetAssembler(mapper, CropRegion(0, 20, 0, 8), logger).run(sources)

    assert parameters.shape == (6, 20, 8)
    assert np.allclose(parameters[1, :10], sources[0].cube.channel_window(0, slice(0, 10), slice(0, 8)))
    assert np.allclose(parameters[1, 10:], sources[1].cube.channel_window(0, slice(0, 10), slice(0, 8)))


def test_assembler_refuses_a_crop_the_sources_do_not_cover(tmp_path, logger):
    path    = write_rat(tmp_path / "params_0a10a0a8_site.rat", source_cube(2, 10, 8, seed=1))
    sources = ExternalSourceResolver([path], [], logger).resolve()
    mapper  = ChannelMapper(SOURCE_ORDER, sources[0].n_gaussians, k_slots=0)

    with pytest.raises(ValueError, match="unfilled"):
        TargetAssembler(mapper, CropRegion(0, 20, 0, 8), logger).run(sources)


def test_assembler_refuses_two_sources_claiming_the_same_pixels(tmp_path, logger):
    first   = write_rat(tmp_path / "params_0a10a0a8_a.rat", source_cube(2, 10, 8, seed=1))
    second  = write_rat(tmp_path / "params_5a15a0a8_b.rat", source_cube(2, 10, 8, seed=2))
    sources = ExternalSourceResolver([first, second], [], logger).resolve()
    mapper  = ChannelMapper(SOURCE_ORDER, sources[0].n_gaussians, k_slots=0)

    with pytest.raises(ValueError, match="already filled by another source"):
        TargetAssembler(mapper, CropRegion(0, 15, 0, 8), logger).run(sources)


def test_validator_rejects_means_outside_the_dataset_height_axis(logger):
    parameters       = np.zeros((3, 4, 4), dtype=np.float32)
    parameters[0]    = 1.0
    parameters[1]    = 120.0
    parameters[2]    = 2.0

    with pytest.raises(ValueError, match="outside the dataset height axis"):
        ParameterValidator(1, 1e-3, HEIGHT_RANGE, logger).run(parameters)


def test_validator_rejects_non_finite_parameters(logger):
    parameters       = np.zeros((3, 4, 4), dtype=np.float32)
    parameters[0]    = 1.0
    parameters[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        ParameterValidator(1, 1e-3, HEIGHT_RANGE, logger).run(parameters)


def test_injection_writes_a_run_the_training_stack_can_read(tmp_path, logger):
    base    = tmp_path / "Dataset"
    dataset = build_dataset(base / "run_a", CropRegion(100, 110, 20, 28))
    path    = write_rat(tmp_path / "params_100a110a20a28_site.rat", source_cube(2, 10, 8, seed=3))
    sources = ExternalSourceResolver([path], [], logger).resolve()

    outputs = ExternalParamsInjectionPipeline(entry_config(base, [path]), dataset, sources, logger=logger).run()

    parameters = np.load(outputs["parameters_npy"])
    gaussian   = GaussianConfig.from_dataset(dataset, outputs["parameters_npy"])

    assert parameters.shape == (6, 10, 8)
    assert parameters.dtype == np.float32
    assert gaussian.n_default_gaussians == 2
    assert gaussian.x_min == HEIGHT_RANGE[0]
    assert ParameterRunMeta.is_external(outputs["output_directory"])


def test_injection_sorts_slots_by_ascending_mean(tmp_path, logger):
    base    = tmp_path / "Dataset"
    dataset = build_dataset(base / "run_a", CropRegion(100, 110, 20, 28))
    path    = write_rat(tmp_path / "params_100a110a20a28_site.rat", source_cube(2, 10, 8, seed=4))
    sources = ExternalSourceResolver([path], [], logger).resolve()

    outputs    = ExternalParamsInjectionPipeline(entry_config(base, [path]), dataset, sources, logger=logger).run()
    parameters = np.load(outputs["parameters_npy"])

    assert (parameters[1] <= parameters[4]).all()


def test_injection_pads_to_the_requested_slot_count(tmp_path, logger):
    base    = tmp_path / "Dataset"
    dataset = build_dataset(base / "run_a", CropRegion(100, 110, 20, 28))
    path    = write_rat(tmp_path / "params_100a110a20a28_site.rat", source_cube(2, 10, 8, seed=5))
    sources = ExternalSourceResolver([path], [], logger).resolve()
    config  = entry_config(base, [path], k_slots=5)

    outputs    = ExternalParamsInjectionPipeline(config, dataset, sources, logger=logger).run()
    parameters = np.load(outputs["parameters_npy"])

    assert parameters.shape == (15, 10, 8)
    assert np.count_nonzero(parameters[6:]) == 0
    assert json.loads((Path(outputs["output_directory"]) / ParameterRunMeta.FILENAME).read_text())["k_max"] == 5


def test_injection_refuses_to_replace_an_existing_run_without_overwrite(tmp_path, logger):
    base    = tmp_path / "Dataset"
    dataset = build_dataset(base / "run_a", CropRegion(100, 110, 20, 28))
    path    = write_rat(tmp_path / "params_100a110a20a28_site.rat", source_cube(2, 10, 8, seed=6))
    sources = ExternalSourceResolver([path], [], logger).resolve()

    ExternalParamsInjectionPipeline(entry_config(base, [path]), dataset, sources, logger=logger).run()

    with pytest.raises(FileExistsError, match="already exists"):
        ExternalParamsInjectionPipeline(entry_config(base, [path]), dataset, sources, logger=logger).run()


def test_injected_runs_are_not_offered_to_the_gaussian_fit_analysis(tmp_path, logger):
    base    = tmp_path / "Dataset"
    dataset = build_dataset(base / "run_a", CropRegion(100, 110, 20, 28))
    path    = write_rat(tmp_path / "params_100a110a20a28_site.rat", source_cube(2, 10, 8, seed=7))
    sources = ExternalSourceResolver([path], [], logger).resolve()

    outputs = ExternalParamsInjectionPipeline(entry_config(base, [path]), dataset, sources, logger=logger).run()

    from pipelines.processing.param_extraction.inference import ParamInferenceTrialCollector

    collector = ParamInferenceTrialCollector(dataset / "params", [], logger)

    assert collector.collect() == []

    with pytest.raises(ValueError, match="injected from external files"):
        ParameterRunMeta.reject_external(outputs["output_directory"], "be scored")
