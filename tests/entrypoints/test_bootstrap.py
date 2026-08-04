from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_MAIN_DIR = Path(__file__).resolve().parents[2] / "main"
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))

import _bootstrap
from _bootstrap import EnvironmentPinner


THREAD_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in THREAD_KEYS + ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_thread_vars_cover_all_blas_backends():
    assert EnvironmentPinner.THREAD_VARS == THREAD_KEYS


def test_threads_sets_documented_env_vars(clean_env):
    EnvironmentPinner.threads()

    for key in THREAD_KEYS:
        assert os.environ[key] == "4"


def test_threads_accepts_custom_count(clean_env):
    EnvironmentPinner.threads(1)

    for key in THREAD_KEYS:
        assert os.environ[key] == "1"


def test_gpu_sets_cuda_visible_devices(clean_env):
    EnvironmentPinner.gpu(gpu_id=3)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_gpu_also_pins_threads(clean_env):
    EnvironmentPinner.gpu(gpu_id=0)

    for key in THREAD_KEYS:
        assert os.environ[key] == "4"


def test_gpu_expandable_segments_toggles_alloc_conf(clean_env):
    EnvironmentPinner.gpu(gpu_id=1, expandable_segments=True)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_gpu_without_expandable_segments_leaves_alloc_conf_unset(clean_env):
    EnvironmentPinner.gpu(gpu_id=1, expandable_segments=False)

    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_gpus_joins_ids(clean_env):
    EnvironmentPinner.gpus([0, 1, 2, 3])

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_gpus_coerces_to_int_strings(clean_env):
    EnvironmentPinner.gpus(["1", 2])

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_gpus_pins_threads(clean_env):
    EnvironmentPinner.gpus([0])

    for key in THREAD_KEYS:
        assert os.environ[key] == "4"


def test_gpus_raises_on_empty_list(clean_env):
    with pytest.raises(ValueError):
        EnvironmentPinner.gpus([])


def test_repo_root_is_repository_directory():
    assert _bootstrap._REPO_ROOT == Path(_bootstrap.__file__).resolve().parent.parent


def test_repo_root_inserted_on_sys_path():
    assert str(_bootstrap._REPO_ROOT) in sys.path


def test_import_is_idempotent_for_sys_path():
    before = sys.path.count(str(_bootstrap._REPO_ROOT))

    importlib.reload(_bootstrap)

    after = sys.path.count(str(_bootstrap._REPO_ROOT))
    assert after == before
