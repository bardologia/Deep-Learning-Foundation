
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class EnvironmentPinner:

    THREAD_VARS = (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )

    @classmethod
    def threads(cls, count: int = 4) -> None:
        for key in cls.THREAD_VARS:
            os.environ[key] = str(count)

    @classmethod
    def gpus(cls, gpu_ids: list) -> None:
        cls._stash_inherited_mask()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cls._mask_tokens(gpu_ids))
        cls.threads()

    @classmethod
    def gpu(cls, gpu_id: int, expandable_segments: bool = False) -> None:
        cls._stash_inherited_mask()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cls._mask_tokens([gpu_id]))
        cls.threads()

        if expandable_segments:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    @classmethod
    def _stash_inherited_mask(cls) -> None:
        if "MLTEMPLATE_INHERITED_CUDA_MASK" not in os.environ:
            os.environ["MLTEMPLATE_INHERITED_CUDA_MASK"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    @classmethod
    def _mask_tokens(cls, gpu_ids: list) -> list:
        ids = [int(gpu_id) for gpu_id in gpu_ids]

        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None:
            return [str(gpu_id) for gpu_id in ids]

        inherited = [token.strip() for token in raw.split(",") if token.strip()]
        if not inherited:
            return []

        return [inherited[gpu_id] for gpu_id in ids]
