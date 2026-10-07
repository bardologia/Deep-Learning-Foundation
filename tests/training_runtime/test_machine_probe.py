"""Tests for the standard-library host probe used by the installer.

Covers the nvidia-smi row parsing, a failing nvidia-smi told apart from a missing
one, the driver-to-torch-index mapping and its explanation, the conda root search
order and the environment interpreter lookup.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import tools.runtime.machine_probe as probe_module
from tools.runtime.machine_probe import MachineProbe

PROBE_SCRIPT = Path(probe_module.__file__)

ROWS = "0, NVIDIA GeForce GTX 1650, 4096, 580.173.02\n1, NVIDIA A100-SXM4-40GB, 40960, 580.173.02\n"


def test_gpu_rows_parse_into_driver_and_devices():
    """Each csv row yields an index, a name and the memory in GB; the driver is taken from the rows."""
    parsed = MachineProbe()._parse_gpus(ROWS)

    assert parsed["driver"] == "580.173.02"
    assert parsed["gpus"]   == [{"index": 0, "name": "NVIDIA GeForce GTX 1650", "memory_gb": 4.0}, {"index": 1, "name": "NVIDIA A100-SXM4-40GB", "memory_gb": 40.0}]
    assert MachineProbe()._parse_gpus("") == {"driver": "", "gpus": []}


def test_a_gpu_name_holding_a_comma_survives_parsing():
    """Extra commas inside the device name are folded back into the name."""
    parsed = MachineProbe()._parse_gpus("0, Tesla V100, PCIe, 16384, 535.1\n")

    assert parsed["gpus"][0]["name"]      == "Tesla V100, PCIe"
    assert parsed["gpus"][0]["memory_gb"] == 16.0


def test_a_missing_nvidia_smi_means_no_driver_and_no_error(monkeypatch):
    """A host without nvidia-smi is a CPU machine: no driver, no devices and nothing to complain about."""
    monkeypatch.setattr(probe_module.shutil, "which", lambda name: None)

    assert MachineProbe().nvidia() == {"driver": "", "gpus": [], "driver_error": ""}


def test_a_failing_nvidia_smi_carries_its_exit_and_last_stderr_line(monkeypatch):
    """A present nvidia-smi that exits non-zero yields no driver but names the failure instead of looking like a CPU machine."""
    monkeypatch.setattr(probe_module.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(probe_module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 9, "", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"))

    nvidia = MachineProbe().nvidia()

    assert nvidia["driver"] == "" and nvidia["gpus"] == []
    assert nvidia["driver_error"] == "nvidia-smi exited 9: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."


def test_a_hung_nvidia_smi_is_reported_as_a_timeout(monkeypatch):
    """A query that never answers is reported as the timeout it hit."""
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 20)

    monkeypatch.setattr(probe_module.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(probe_module.subprocess, "run", hang)

    assert MachineProbe().nvidia()["driver_error"].startswith("nvidia-smi TimeoutExpired")


@pytest.mark.parametrize("driver, index", [("580.173.02", "cu130"), ("581.0", "cu130"), ("570.86", "cu128"), ("525.60.13", "cu128"), ("515.1", "cpu"), ("", "cpu")])
def test_the_torch_index_follows_the_driver_major(driver, index):
    """Drivers from 580 take cu130, from 525 cu128, older or absent drivers the CPU wheels."""
    assert MachineProbe().torch_index(driver) == index


def test_the_torch_note_explains_every_cpu_choice():
    """A CUDA wheel needs no note; a CPU wheel is explained by the failure, the missing driver or the old major."""
    probe = MachineProbe()

    assert probe.torch_note("580.1", "")  == ""
    assert probe.torch_note("", "nvidia-smi exited 9: no driver") == "nvidia-smi exited 9: no driver"
    assert probe.torch_note("", "")       == "no NVIDIA driver: nvidia-smi is not installed"
    assert probe.torch_note("515.1", "")  == "driver 515.1 is below 525, the oldest major the torch wheels support"


def test_torch_index_fails_the_installer_when_nvidia_smi_is_present_but_broken(tmp_path):
    """The --torch-index entry exits non-zero with the failure instead of printing cpu for a GPU machine whose driver is down."""
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.' >&2\nexit 9\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run([sys.executable, str(PROBE_SCRIPT), "--torch-index"], capture_output=True, text=True, env={**os.environ, "PATH": str(tmp_path)})

    assert result.returncode == 2
    assert result.stdout == ""
    assert "nvidia-smi exited 9" in result.stderr
    assert "--torch-index" in result.stderr


def test_torch_index_prints_cpu_with_a_note_when_no_nvidia_smi_exists(tmp_path):
    """Without nvidia-smi the entry still chooses the CPU wheel and says why on stderr."""
    result = subprocess.run([sys.executable, str(PROBE_SCRIPT), "--torch-index"], capture_output=True, text=True, env={**os.environ, "PATH": str(tmp_path)})

    assert result.returncode == 0
    assert result.stdout.strip() == "cpu"
    assert "nvidia-smi is not installed" in result.stderr


def test_conda_root_prefers_the_active_installation(tmp_path, monkeypatch):
    """CONDA_EXE wins over the PATH, which wins over the home directory scan."""
    active = tmp_path / "active" / "bin" / "conda"
    active.parent.mkdir(parents=True)
    active.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(active))

    assert MachineProbe().conda_root() == tmp_path / "active"

    monkeypatch.delenv("CONDA_EXE")
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    assert MachineProbe().conda_root() is None

    (tmp_path / "home" / "miniforge3" / "condabin").mkdir(parents=True)
    (tmp_path / "home" / "miniforge3" / "condabin" / "conda").write_text("")

    assert MachineProbe().conda_root() == tmp_path / "home" / "miniforge3"


def test_env_interpreter_requires_the_python_to_exist(tmp_path):
    """The environment interpreter is returned only when the file exists under the conda root."""
    probe = MachineProbe()

    assert probe.env_interpreter(None, "Dune")     is None
    assert probe.env_interpreter(tmp_path, "Dune") is None

    python = tmp_path / "envs" / "Dune" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")

    assert probe.env_interpreter(tmp_path, "Dune") == python


def test_facts_carry_every_key_the_wizard_reads(monkeypatch):
    """The facts mapping has the host, CPU, RAM, driver and its error, GPU, torch index and note, and conda keys."""
    monkeypatch.setattr(MachineProbe, "nvidia", lambda self: {"driver": "", "gpus": [], "driver_error": "nvidia-smi exited 9: no driver"})
    facts = MachineProbe().facts()

    assert set(facts) == {"hostname", "cpu_count", "ram_gb", "driver", "driver_error", "gpus", "torch_index", "torch_note", "conda_root"}
    assert facts["cpu_count"] >= 1
    assert facts["ram_gb"]    > 0
    assert facts["torch_index"]  == "cpu"
    assert facts["driver_error"] == "nvidia-smi exited 9: no driver"
    assert facts["torch_note"]   == "nvidia-smi exited 9: no driver"
