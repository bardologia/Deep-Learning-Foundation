from __future__ import annotations

import json
from pathlib import Path

from configuration.sar.processing_config  import PathConfig, TomogramConfig
from pipelines.processing.generation.base import GeneratorBase
from tools.monitoring.logger              import Logger


def _logger(tmp_path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="spec_test", level="ERROR")


def _spec() -> dict:
    return {
        "main_directory"   : "/data/runs/example",
        "pyrat_directory"  : "/data/pyrat",
        "run_subdirectory" : "run_001",
        "tomogram_config"  : {
            "polarisation"  : "hv",
            "height_range"  : [-20.0, 80.0],
            "filter_method" : "Boxcar",
        },
    }


def test_paths_built_from_spec(tmp_path):
    gen   = GeneratorBase(_spec(), _logger(tmp_path))
    paths = gen._paths()

    assert isinstance(paths, PathConfig)
    assert paths.main_directory   == Path("/data/runs/example")
    assert paths.pyrat_directory  == Path("/data/pyrat")
    assert paths.run_subdirectory == "run_001"
    assert paths.run_directory    == Path("/data/runs/example/run_001")


def test_tomogram_config_built_from_spec(tmp_path):
    gen  = GeneratorBase(_spec(), _logger(tmp_path))
    tcfg = gen._tomogram_config()

    assert isinstance(tcfg, TomogramConfig)
    assert tcfg.polarisation  == "hv"
    assert tcfg.height_range  == [-20.0, 80.0]
    assert tcfg.filter_method == "Boxcar"


def test_from_spec_file_roundtrip(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec()))

    gen = GeneratorBase.from_spec_file(spec_path, _logger(tmp_path))

    assert gen.spec == _spec()
    assert gen._paths().run_subdirectory == "run_001"


def test_spec_values_propagate_into_configs(tmp_path):
    spec = _spec()
    spec["main_directory"]                  = "/elsewhere"
    spec["tomogram_config"]["polarisation"] = "vv"

    gen = GeneratorBase(spec, _logger(tmp_path))

    assert gen._paths().main_directory         == Path("/elsewhere")
    assert gen._tomogram_config().polarisation == "vv"
