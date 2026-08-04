from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configuration.synthesis import SynthesisConfig

_SCENE_CONFIG = SynthesisConfig()


class SilentProgress:

    def add_task(self, *args, **kwargs):
        return 0

    def advance(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class SilentLogger:

    def section(self, *args, **kwargs):       pass
    def subsection(self, *args, **kwargs):    pass
    def debug(self, *args, **kwargs):         pass
    def info(self, *args, **kwargs):          pass
    def warning(self, *args, **kwargs):       pass
    def error(self, *args, **kwargs):         pass
    def critical(self, *args, **kwargs):      pass
    def ok(self, *args, **kwargs):            pass
    def render(self, *args, **kwargs):        pass
    def panel(self, *args, **kwargs):         pass
    def rule(self, *args, **kwargs):          pass
    def kv_table(self, *args, **kwargs):      pass
    def metrics_table(self, *args, **kwargs): pass
    def close(self, *args, **kwargs):         pass

    @contextlib.contextmanager
    def timer(self, label: str):
        yield

    @contextlib.contextmanager
    def track(self, transient: bool = False):
        yield SilentProgress()

    progress_bar = track

    @contextlib.contextmanager
    def live_monitor(self, title: str = "Training Monitor"):
        yield SilentProgress()


def pytest_configure(config):
    config.addinivalue_line("markers", "dataset: test runs against the generated synthetic dataset scene")
    config.addinivalue_line("markers", "slow: heavier test (full-frame numerics, many model forwards)")


@pytest.fixture
def logger_stub() -> SilentLogger:
    return SilentLogger()


@pytest.fixture
def force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def scene_config() -> SynthesisConfig:
    return _SCENE_CONFIG


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    from pipelines.processing.synthesis.pipeline import SyntheticDatasetPipeline

    config = dataclasses.replace(_SCENE_CONFIG, output_directory=tmp_path_factory.mktemp("synthetic_scene") / "synthetic_small")

    SyntheticDatasetPipeline(config, logger=SilentLogger()).run()

    return config.output_directory


@pytest.fixture(scope="session")
def data_dir(test_data_dir) -> Path:
    return test_data_dir / "data"


@pytest.fixture(scope="session")
def meta_dir(test_data_dir) -> Path:
    return test_data_dir / "meta"


@pytest.fixture(scope="session")
def params_dir(test_data_dir) -> Path:
    return test_data_dir / "params" / _SCENE_CONFIG.params_run_name


@pytest.fixture(scope="session")
def dataset_json(data_dir) -> dict:
    return json.loads((data_dir / "dataset.json").read_text())


@pytest.fixture(scope="session")
def config_state_json(meta_dir) -> dict:
    return json.loads((meta_dir / "config_state.json").read_text())


@pytest.fixture(scope="session")
def param_extraction_meta(params_dir) -> dict:
    return json.loads((params_dir / "param_extraction_meta.json").read_text())


@pytest.fixture(scope="session")
def channel_labels(dataset_json) -> list:
    return dataset_json["channel_labels"]


@pytest.fixture(scope="session")
def cube_full(data_dir) -> np.ndarray:
    return np.load(data_dir / "cube_full.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def aux_full(data_dir) -> np.ndarray:
    return np.load(data_dir / "aux_full.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def primary(data_dir) -> np.ndarray:
    return np.load(data_dir / "primary.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def secondaries(data_dir) -> np.ndarray:
    return np.load(data_dir / "secondaries.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def derived(data_dir) -> np.ndarray:
    return np.load(data_dir / "derived.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def parameters(params_dir) -> np.ndarray:
    return np.load(params_dir / "parameters.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def fit_diagnostics(params_dir) -> dict:
    npz = np.load(params_dir / "fit_diagnostics.npz")
    return {k: npz[k] for k in npz.files}


@pytest.fixture
def small_window():
    return (slice(0, 32), slice(0, 32))
