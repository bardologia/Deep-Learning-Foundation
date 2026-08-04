from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TEST_DATA = _REPO_ROOT / "test_data"
_DATA      = _TEST_DATA / "data"
_META      = _TEST_DATA / "meta"
_PARAMS    = _TEST_DATA / "params" / "params_k5_lam0.01_sig4_sigma"

_HAS_DATA = _TEST_DATA.is_dir() and (_DATA / "dataset.json").is_file()


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
    config.addinivalue_line("markers", "real_data: test depends on the transferred real SAR data under test_data/")
    config.addinivalue_line("markers", "slow: heavier test (full-frame numerics, many model forwards)")


def _require_data():
    if not _HAS_DATA:
        pytest.skip("real data not present under test_data/")


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
def test_data_dir() -> Path:
    _require_data()
    return _TEST_DATA


@pytest.fixture(scope="session")
def data_dir() -> Path:
    _require_data()
    return _DATA


@pytest.fixture(scope="session")
def meta_dir() -> Path:
    _require_data()
    return _META


@pytest.fixture(scope="session")
def params_dir() -> Path:
    _require_data()
    return _PARAMS


@pytest.fixture(scope="session")
def dataset_json(data_dir) -> dict:
    return json.loads((data_dir / "dataset.json").read_text())


@pytest.fixture(scope="session")
def config_state_json(meta_dir) -> dict:
    return json.loads((meta_dir / "config_state.json").read_text())


@pytest.fixture(scope="session")
def baselines_json(meta_dir) -> dict:
    return json.loads((meta_dir / "baselines.json").read_text())


@pytest.fixture(scope="session")
def param_extraction_meta(params_dir) -> dict:
    return json.loads((params_dir / "param_extraction_meta.json").read_text())


@pytest.fixture(scope="session")
def pass_labels(dataset_json) -> list:
    return dataset_json["pass_labels"]


@pytest.fixture(scope="session")
def tomogram_full(data_dir) -> np.ndarray:
    return np.load(data_dir / "tomogram_full.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def dem_full(data_dir) -> np.ndarray:
    return np.load(data_dir / "dem_full.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def primary(data_dir) -> np.ndarray:
    return np.load(data_dir / "primary.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def secondaries(data_dir) -> np.ndarray:
    return np.load(data_dir / "secondaries.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def interferograms(data_dir) -> np.ndarray:
    return np.load(data_dir / "interferograms.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def parameters(params_dir) -> np.ndarray:
    return np.load(params_dir / "parameters.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def fit_diagnostics(params_dir) -> dict:
    npz = np.load(params_dir / "fit_diagnostics.npz")
    return {k: npz[k] for k in npz.files}


@pytest.fixture(scope="session")
def track_profiles(data_dir) -> dict:
    npz = np.load(data_dir / "track_profiles.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


@pytest.fixture
def small_window():
    return (slice(0, 32), slice(0, 32))
