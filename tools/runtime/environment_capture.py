
from __future__ import annotations

import os
import platform
import socket
import sys

import numpy as np
import scipy
import torch


class EnvironmentCapture:

    @staticmethod
    def _cuda_visible() -> bool:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0

    @classmethod
    def _gpu_entries(cls) -> list[dict]:
        if not cls._cuda_visible():
            return []

        entries = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            entries.append({
                "index"              : index,
                "name"               : properties.name,
                "capability"         : f"{properties.major}.{properties.minor}",
                "total_memory_bytes" : int(properties.total_memory),
            })
        return entries

    @classmethod
    def _driver_version(cls) -> str | None:
        if not cls._cuda_visible():
            return None

        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()

        return version.decode() if isinstance(version, bytes) else str(version)

    @classmethod
    def snapshot(cls) -> dict:
        return {
            "hostname"             : socket.gethostname(),
            "platform"             : platform.platform(),
            "conda_env"            : os.environ.get("CONDA_DEFAULT_ENV"),
            "python"               : sys.version.split()[0],
            "numpy"                : np.__version__,
            "scipy"                : scipy.__version__,
            "torch"                : torch.__version__,
            "torch_cuda_build"     : torch.version.cuda,
            "cudnn"                : torch.backends.cudnn.version() if cls._cuda_visible() else None,
            "cuda_visible_devices" : os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver_version"       : cls._driver_version(),
            "gpus"                 : cls._gpu_entries(),
        }
