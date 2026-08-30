"""Software and hardware environment snapshots recorded alongside run outputs.

Captures the interpreter, library and device state a run executes under, so a
result can be traced back to the exact torch build, CUDA stack and GPU model
that produced it after any of them change.
"""

from __future__ import annotations

import os
import platform
import socket
import sys

import numpy as np
import scipy
import torch


class EnvironmentCapture:
    """Collects the interpreter, library and device state of the running process."""

    @staticmethod
    def _cuda_visible() -> bool:
        """Returns whether CUDA is usable with at least one visible device.

        A process whose CUDA_VISIBLE_DEVICES masks every device reports CUDA as
        available with a zero device count, and torch's cudnn version probe
        crashes on that state, so both conditions are checked.
        """
        return torch.cuda.is_available() and torch.cuda.device_count() > 0

    @classmethod
    def _gpu_entries(cls) -> list[dict]:
        """Returns one entry per visible CUDA device, empty on CPU-only hosts."""
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
        """Returns the NVIDIA driver version, or None on CPU-only hosts."""
        if not cls._cuda_visible():
            return None

        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()

        return version.decode() if isinstance(version, bytes) else str(version)

    @classmethod
    def snapshot(cls) -> dict:
        """Returns the environment as a JSON-serialisable mapping.

        Returns:
            Mapping with the hostname, platform string, conda environment name, Python, NumPy, SciPy and torch versions, the torch CUDA and cuDNN build versions, the CUDA_VISIBLE_DEVICES mask, the NVIDIA driver version and one entry per visible GPU; the GPU list is empty and the driver version None on CPU-only hosts.
        """
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
