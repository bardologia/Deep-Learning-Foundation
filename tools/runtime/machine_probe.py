
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path


class MachineProbe:

    TORCH_INDEX_BY_DRIVER = ((580, "cu130"), (525, "cu128"))
    CONDA_DIRS            = ("miniconda3", "anaconda3", "miniforge3", "mambaforge", ".conda")
    GPU_QUERY             = "index,name,memory.total,driver_version"

    def hostname(self) -> str:
        return socket.gethostname()

    def cpu_count(self) -> int:
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))

        return os.cpu_count() or 1

    def ram_gb(self) -> float:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

        return round(total / 1024 ** 3, 1)

    def nvidia(self) -> dict:
        binary = shutil.which("nvidia-smi")

        if binary is None:
            return {"driver": "", "gpus": [], "driver_error": ""}

        try:
            result = subprocess.run([binary, f"--query-gpu={self.GPU_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"driver": "", "gpus": [], "driver_error": f"nvidia-smi {type(error).__name__}: {error}"}

        if result.returncode != 0:
            message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no message on stderr"
            return {"driver": "", "gpus": [], "driver_error": f"nvidia-smi exited {result.returncode}: {message}"}

        return {**self._parse_gpus(result.stdout), "driver_error": ""}

    def _parse_gpus(self, text: str) -> dict:
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
        if not driver:
            return "cpu"

        major = int(driver.split(".")[0])

        for minimum, index in self.TORCH_INDEX_BY_DRIVER:
            if major >= minimum:
                return index

        return "cpu"

    def torch_note(self, driver: str, driver_error: str) -> str:
        if driver_error:
            return driver_error
        if not driver:
            return "no NVIDIA driver: nvidia-smi is not installed"
        if self.torch_index(driver) == "cpu":
            return f"driver {driver} is below {self.TORCH_INDEX_BY_DRIVER[-1][0]}, the oldest major the torch wheels support"

        return ""

    def conda_root(self) -> Path | None:
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
        if conda_root is None:
            return None

        candidate = Path(conda_root) / "envs" / env_name / "bin" / "python"

        return candidate if candidate.is_file() else None

    def facts(self) -> dict:
        nvidia = self.nvidia()
        root   = self.conda_root()

        return {
            "hostname"     : self.hostname(),
            "cpu_count"    : self.cpu_count(),
            "ram_gb"       : self.ram_gb(),
            "driver"       : nvidia["driver"],
            "driver_error" : nvidia["driver_error"],
            "gpus"         : nvidia["gpus"],
            "torch_index"  : self.torch_index(nvidia["driver"]),
            "torch_note"   : self.torch_note(nvidia["driver"], nvidia["driver_error"]),
            "conda_root"   : "" if root is None else str(root),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the host facts the installer needs: GPUs and driver, CPU count, RAM, the torch wheel index and the conda installation.")
    parser.add_argument("--torch-index", action="store_true", help="Print only the torch wheel index (cu130, cu128 or cpu); fails when nvidia-smi is present but does not answer.")
    parser.add_argument("--conda-root",  action="store_true", help="Print only the conda installation root, or nothing when none is found.")
    args   = parser.parse_args()
    probe  = MachineProbe()

    if args.torch_index:
        nvidia = probe.nvidia()
        note   = probe.torch_note(nvidia["driver"], nvidia["driver_error"])

        if nvidia["driver_error"]:
            print(f"{nvidia['driver_error']}; fix the driver and rerun, or pass --torch-index cu130, cu128 or cpu to install.sh to choose the wheel yourself", file=sys.stderr)
            sys.exit(2)
        if note:
            print(f"CPU wheel: {note}", file=sys.stderr)

        print(probe.torch_index(nvidia["driver"]))
    elif args.conda_root:
        root = probe.conda_root()
        print("" if root is None else root)
    else:
        print(json.dumps(probe.facts(), indent=2))


if __name__ == "__main__":
    main()
