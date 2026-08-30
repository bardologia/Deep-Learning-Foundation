"""Tests for the entry-point bootstrap that pins threads, GPUs and sys.path.

Covers the BLAS thread variables set by EnvironmentPinner, single- and
multi-GPU device pinning with the CUDA allocator toggle, and the idempotent
insertion of the repository root on sys.path.
"""

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
    """Returns the monkeypatch fixture with all thread and CUDA environment variables removed."""
    for key in THREAD_KEYS + ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TOMOSAR_INHERITED_CUDA_MASK"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_thread_vars_cover_all_blas_backends():
    """The pinned thread variables are exactly the documented BLAS backend keys."""
    assert EnvironmentPinner.THREAD_VARS == THREAD_KEYS


def test_threads_sets_documented_env_vars(clean_env):
    """Pinning threads without an argument sets every thread variable to four."""
    EnvironmentPinner.threads()

    for key in THREAD_KEYS:
        assert os.environ[key] == "4"


def test_threads_accepts_custom_count(clean_env):
    """An explicit thread count is written to every thread variable."""
    EnvironmentPinner.threads(1)

    for key in THREAD_KEYS:
        assert os.environ[key] == "1"


def test_gpu_sets_cuda_visible_devices(clean_env):
    """Pinning a single GPU exposes only that device id."""
    EnvironmentPinner.gpu(gpu_id=3)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_gpu_also_pins_threads(clean_env):
    """Pinning a GPU also pins the BLAS thread variables."""
    EnvironmentPinner.gpu(gpu_id=0)

    for key in THREAD_KEYS:
        assert os.environ[key] == "4"


def test_gpu_expandable_segments_toggles_alloc_conf(clean_env):
    """Requesting expandable segments sets the CUDA allocator configuration."""
    EnvironmentPinner.gpu(gpu_id=1, expandable_segments=True)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_gpu_without_expandable_segments_leaves_alloc_conf_unset(clean_env):
    """Without expandable segments the CUDA allocator configuration stays unset."""
    EnvironmentPinner.gpu(gpu_id=1, expandable_segments=False)

    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_gpus_joins_ids(clean_env):
    """Pinning several GPUs writes them as a comma-separated device list."""
    EnvironmentPinner.gpus([0, 1, 2, 3])

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_gpus_coerces_to_int_strings(clean_env):
    """Mixed string and integer device ids are normalised to integer strings."""
    EnvironmentPinner.gpus(["1", 2])

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_gpus_pins_threads(clean_env):
    """Pinning several GPUs also pins the BLAS thread variables."""
    EnvironmentPinner.gpus([0])

    for key in THREAD_KEYS:
        assert os.environ[key] == "4"


def test_gpu_indexes_into_an_inherited_slurm_mask(clean_env):
    """Under a scheduler mask, --gpu 0 selects the allocation's device, not physical GPU 0."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "2")
    EnvironmentPinner.gpu(gpu_id=0)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_gpus_index_into_an_inherited_multi_device_mask(clean_env):
    """Positions inside an inherited mask resolve to that mask's tokens, order preserved."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    EnvironmentPinner.gpus([1, 0])

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,3"


def test_gpu_outside_the_inherited_mask_raises(clean_env):
    """An index past the inherited allocation is refused instead of leaking onto foreign devices."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "2")

    with pytest.raises(ValueError, match="outside the inherited"):
        EnvironmentPinner.gpu(gpu_id=1)


def test_gpu_uuid_mask_tokens_pass_through(clean_env):
    """A UUID-token mask is indexed by position and its token kept verbatim."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "GPU-91af9d73,GPU-657ba5d8")
    EnvironmentPinner.gpu(gpu_id=1)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-657ba5d8"


def test_gpus_raises_on_empty_list(clean_env):
    """Pinning an empty GPU list raises ValueError."""
    with pytest.raises(ValueError):
        EnvironmentPinner.gpus([])


def test_repo_root_is_repository_directory():
    """The bootstrap resolves the repository root as the parent of the main directory."""
    assert _bootstrap._REPO_ROOT == Path(_bootstrap.__file__).resolve().parent.parent


def test_repo_root_inserted_on_sys_path():
    """Importing the bootstrap places the repository root on sys.path."""
    assert str(_bootstrap._REPO_ROOT) in sys.path


def test_import_is_idempotent_for_sys_path():
    """Reloading the bootstrap does not duplicate the repository root on sys.path."""
    before = sys.path.count(str(_bootstrap._REPO_ROOT))

    importlib.reload(_bootstrap)

    after = sys.path.count(str(_bootstrap._REPO_ROOT))
    assert after == before


def test_pin_stashes_an_unset_inherited_mask_as_empty(clean_env):
    """Pinning on a bare machine records the absent inherited mask as an empty stash."""
    EnvironmentPinner.gpu(gpu_id=2)

    assert os.environ["TOMOSAR_INHERITED_CUDA_MASK"] == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_pin_stashes_the_inherited_scheduler_mask(clean_env):
    """Pinning inside a scheduler allocation records the allocation's original mask."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    EnvironmentPinner.gpu(gpu_id=1)

    assert os.environ["TOMOSAR_INHERITED_CUDA_MASK"] == "2,3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_a_second_pin_keeps_the_first_stash(clean_env):
    """Later pins in the same process never overwrite the original inherited snapshot."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    EnvironmentPinner.gpus([0, 1])
    EnvironmentPinner.gpu(gpu_id=0)

    assert os.environ["TOMOSAR_INHERITED_CUDA_MASK"] == "2,3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
