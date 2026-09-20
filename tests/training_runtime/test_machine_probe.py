"""Tests for the standard-library host probe used by the installer.

Covers the nvidia-smi row parsing, the driver-to-torch-index mapping, the conda
root search order and the environment interpreter lookup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.runtime.machine_probe import MachineProbe

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


@pytest.mark.parametrize("driver, index", [("580.173.02", "cu130"), ("581.0", "cu130"), ("570.86", "cu128"), ("525.60.13", "cu128"), ("515.1", "cpu"), ("", "cpu")])
def test_the_torch_index_follows_the_driver_major(driver, index):
    """Drivers from 580 take cu130, from 525 cu128, older or absent drivers the CPU wheels."""
    assert MachineProbe().torch_index(driver) == index


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
    """The facts mapping has the host, CPU, RAM, driver, GPU, torch index and conda keys."""
    monkeypatch.setattr(MachineProbe, "nvidia", lambda self: {"driver": "", "gpus": []})
    facts = MachineProbe().facts()

    assert set(facts) == {"hostname", "cpu_count", "ram_gb", "driver", "gpus", "torch_index", "conda_root"}
    assert facts["cpu_count"] >= 1
    assert facts["ram_gb"]    > 0
    assert facts["torch_index"] == "cpu"
