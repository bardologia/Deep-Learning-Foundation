from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path

import pytest

from configuration.sar.processing_config  import PathConfig, TomogramConfig
from pipelines.processing.generation.base import GeneratorBase
from tools.data.io                        import FileIO
from tools.monitoring.logger              import Logger


@pytest.fixture
def logger(tmp_path):
    log = Logger(log_dir=str(tmp_path / "logs"), name="spec_test")
    yield log
    log.close()


@pytest.fixture
def spec(tmp_path):
    tomogram_config = asdict(TomogramConfig())

    return {
        "main_directory"   : str(tmp_path / "main"),
        "pyrat_directory"  : str(tmp_path / "pyrat"),
        "run_subdirectory" : "run_w20_10",
        "tomogram_config"  : tomogram_config,
    }


def test_init_stores_spec_and_logger(spec, logger):
    generator = GeneratorBase(spec, logger)

    assert generator.spec   is spec
    assert generator.logger is logger


def test_paths_builds_path_config(spec, logger):
    generator = GeneratorBase(spec, logger)

    paths = generator._paths()

    assert isinstance(paths, PathConfig)
    assert paths.main_directory   == Path(spec["main_directory"])
    assert paths.pyrat_directory  == Path(spec["pyrat_directory"])
    assert paths.run_subdirectory == spec["run_subdirectory"]


def test_tomogram_config_round_trips_through_asdict(spec, logger):
    generator = GeneratorBase(spec, logger)

    rebuilt = generator._tomogram_config()

    assert isinstance(rebuilt, TomogramConfig)
    assert asdict(rebuilt) == spec["tomogram_config"]


def test_tomogram_config_preserves_filter_arguments(spec, logger):
    spec["tomogram_config"]["filter_arguments"] = {"win": [20, 10]}
    generator = GeneratorBase(spec, logger)

    rebuilt = generator._tomogram_config()

    assert rebuilt.filter_arguments == {"win": [20, 10]}


def test_from_spec_file_loads_json(spec, logger, tmp_path):
    spec_path = tmp_path / "spec.json"
    FileIO.save_json(spec, spec_path)

    generator = GeneratorBase.from_spec_file(spec_path, logger)

    assert generator.spec["run_subdirectory"] == spec["run_subdirectory"]
    assert generator.logger is logger


def test_from_spec_file_round_trip_rebuilds_configs(spec, logger, tmp_path):
    spec_path = tmp_path / "spec.json"
    FileIO.save_json(spec, spec_path)

    generator = GeneratorBase.from_spec_file(spec_path, logger)
    reloaded  = FileIO.load_json(spec_path)["tomogram_config"]

    assert generator._paths().run_subdirectory  == "run_w20_10"
    assert asdict(generator._tomogram_config()) == reloaded


def test_from_spec_file_accepts_path_and_str(spec, logger, tmp_path):
    spec_path = tmp_path / "spec.json"
    FileIO.save_json(spec, spec_path)

    from_str  = GeneratorBase.from_spec_file(str(spec_path), logger)
    from_path = GeneratorBase.from_spec_file(spec_path, logger)

    assert from_str.spec == from_path.spec


def test_subclass_from_spec_file_returns_subclass(spec, logger, tmp_path):
    class Dummy(GeneratorBase):
        pass

    spec_path = tmp_path / "spec.json"
    FileIO.save_json(spec, spec_path)

    instance = Dummy.from_spec_file(spec_path, logger)

    assert isinstance(instance, Dummy)
