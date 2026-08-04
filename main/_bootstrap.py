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
        ids = [str(int(gpu_id)) for gpu_id in gpu_ids]
        if not ids:
            raise ValueError("gpu_ids must name at least one CUDA device")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        cls.threads()

    @classmethod
    def gpu(cls, gpu_id: int, expandable_segments: bool = False) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cls.threads()

        if expandable_segments:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
