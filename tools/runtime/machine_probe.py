"""Host facts the installer needs, detected with the standard library only.

Runs before the project environment exists, so it imports nothing beyond the
standard library: ``install.sh`` calls it under the conda base interpreter to pick
the torch wheel index, and the setup wizard imports it to seed its questions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path


class MachineProbe:
    """Detects hostname, CPU, RAM, NVIDIA driver and GPUs, and the conda installation.

    Attributes:
        TORCH_INDEX_BY_DRIVER : Pairs of minimum NVIDIA driver major version and the torch wheel index it supports, newest first.
        CONDA_DIRS            : Directory names under the home directory that hold a conda installation.
        GPU_QUERY             : nvidia-smi query fields, in output column order.
    """

    TORCH_INDEX_BY_DRIVER = ((580, "cu130"), (525, "cu128"))
    CONDA_DIRS            = ("miniconda3", "anaconda3", "miniforge3", "mambaforge", ".conda")
    GPU_QUERY             = "index,name,memory.total,driver_version"

    def hostname(self) -> str:
        """Returns the short host name."""
        return socket.gethostname()

    def cpu_count(self) -> int:
        """Returns the number of CPUs this process may run on."""
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))

        return os.cpu_count() or 1

    def ram_gb(self) -> float:
        """Returns the physical memory in gigabytes, rounded to one decimal."""
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

        return round(total / 1024 ** 3, 1)

    def nvidia(self) -> dict:
        """Returns the NVIDIA driver version and the GPUs nvidia-smi reports.

        Returns:
            Mapping with ``driver`` (empty when no driver answers) and ``gpus``, one
            entry per device with ``index``, ``name`` and ``memory_gb``.
        """
        binary = shutil.which("nvidia-smi")

        if binary is None:
            return {"driver": "", "gpus": []}

        try:
            result = subprocess.run([binary, f"--query-gpu={self.GPU_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return {"driver": "", "gpus": []}

        if result.returncode != 0:
            return {"driver": "", "gpus": []}

        return self._parse_gpus(result.stdout)

    def _parse_gpus(self, text: str) -> dict:
        """Parses the csv rows of the GPU query into the driver and device list.

        Args:
            text : nvidia-smi output, one ``index, name, memory MiB, driver`` row per device.

        Returns:
            Mapping with ``driver`` and ``gpus``.
        """
        driver = ""
        gpus   = []

        for line in text.strip().splitlines():
            cells = [cell.strip() for cell in line.split(",")]

            if len(cells) < 4:
                continue

            driver = cells[-1]
            gpus.append({"index": int(cells[0]), "name": ", ".join(cells[1:-2]), "memory_gb": round(float(cells[-2]) / 1024, 1)})

        return {"driver": driver, "gpus": gpus}

    def torch_index(self, driver: str) -> str:
        """Returns the torch wheel index matching a driver version.

        Args:
            driver : NVIDIA driver version such as ``580.173.02``, or empty for no driver.

        Returns:
            ``cu130`` for drivers from 580, ``cu128`` from 525, ``cpu`` otherwise.
        """
        if not driver:
            return "cpu"

        major = int(driver.split(".")[0])

        for minimum, index in self.TORCH_INDEX_BY_DRIVER:
            if major >= minimum:
                return index

        return "cpu"

    def conda_root(self) -> Path | None:
        """Returns the conda installation root, or None when none is found.

        The active installation (``CONDA_EXE``) wins, then a ``conda`` on the PATH,
        then the usual directories under the home directory.
        """
        exe = os.environ.get("CONDA_EXE")

        if exe and Path(exe).is_file():
            return Path(exe).resolve().parent.parent

        found = shutil.which("conda")

        if found:
            return Path(found).resolve().parent.parent

        for name in self.CONDA_DIRS:
            candidate = Path.home() / name

            if (candidate / "bin" / "conda").is_file() or (candidate / "condabin" / "conda").is_file():
                return candidate

        return None

    def env_interpreter(self, conda_root: Path | None, env_name: str) -> Path | None:
        """Returns the python of a named environment under a conda root, or None when absent."""
        if conda_root is None:
            return None

        candidate = Path(conda_root) / "envs" / env_name / "bin" / "python"

        return candidate if candidate.is_file() else None

    def facts(self) -> dict:
        """Returns every detected fact in one mapping."""
        nvidia = self.nvidia()
        root   = self.conda_root()

        return {
            "hostname"   : self.hostname(),
            "cpu_count"  : self.cpu_count(),
            "ram_gb"     : self.ram_gb(),
            "driver"     : nvidia["driver"],
            "gpus"       : nvidia["gpus"],
            "torch_index": self.torch_index(nvidia["driver"]),
            "conda_root" : "" if root is None else str(root),
        }


def main() -> None:
    """Prints the detected facts as JSON, or one requested fact as plain text."""
    parser = argparse.ArgumentParser(description="Detect the host facts the installer needs: GPUs and driver, CPU count, RAM, the torch wheel index and the conda installation.")
    parser.add_argument("--torch-index", action="store_true", help="Print only the torch wheel index (cu130, cu128 or cpu).")
    parser.add_argument("--conda-root",  action="store_true", help="Print only the conda installation root, or nothing when none is found.")
    args   = parser.parse_args()
    probe  = MachineProbe()

    if args.torch_index:
        print(probe.torch_index(probe.nvidia()["driver"]))
    elif args.conda_root:
        root = probe.conda_root()
        print("" if root is None else root)
    else:
        print(json.dumps(probe.facts(), indent=2))


if __name__ == "__main__":
    main()
