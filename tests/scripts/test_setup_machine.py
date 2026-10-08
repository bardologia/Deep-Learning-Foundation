"""Tests for the guided machine setup.

Covers the prompter's preset, unattended and terminal paths with validation, the
validators, and the wizard run end to end against a stub probe: the profile it
writes, the roots it creates, the interpreter hand-over from the installer, the
way a rerun defaults to the saved answers, and the refusal of an invalid preset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.reference_layout import ReferenceLayout
from scripts.setup_machine          import MachineSetup, Prompter, Validators
from tools.runtime.machine_profile  import MachineProfile

FACTS = {
    "hostname"     : "box",
    "cpu_count"    : 8,
    "ram_gb"       : 31.2,
    "driver"       : "580.1",
    "driver_error" : "",
    "gpus"         : [{"index": 0, "name": "A", "memory_gb": 24.0}, {"index": 1, "name": "B", "memory_gb": 12.0}],
    "torch_index"  : "cu130",
    "torch_note"   : "",
    "conda_root"   : "/opt/conda",
}


class StubProbe:
    """Probe stand-in returning fixed facts and a fixed environment interpreter.

    Attributes:
        facts_value : Facts mapping returned by ``facts``.
        interpreter : Path returned by ``env_interpreter``.
    """

    def __init__(self, facts: dict, interpreter: Path) -> None:
        """Stores the canned facts and interpreter."""
        self.facts_value = facts
        self.interpreter = interpreter

    def facts(self) -> dict:
        """Returns the canned facts."""
        return dict(self.facts_value)

    def env_interpreter(self, conda_root, env_name: str):
        """Returns the canned interpreter regardless of the environment asked for."""
        return self.interpreter


def _setup(tmp_path: Path, logger_stub, answers: dict | None = None, assume: bool = True, saved: dict | None = None, handed: bool = True) -> MachineSetup:
    """Builds a wizard over a temporary repository, interpreter and profile location.

    Args:
        tmp_path    : Root for the fake repository, data and profile.
        logger_stub : Silent logger.
        answers     : Preset answers merged over the minimal set that keeps every path under tmp_path.
        assume      : Whether unanswered questions take their defaults.
        saved       : Profile content already on disk before the run, or None for a first run.
        handed      : Whether the installer hands the interpreter over, as install.sh does.

    Returns:
        A wizard with verification and the desktop launcher disabled.
    """
    repo        = tmp_path / "repo"
    interpreter = tmp_path / "envs" / "Dune" / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("")
    (repo / "webui").mkdir(parents=True, exist_ok=True)

    path = tmp_path / "cfg" / "machine.json"
    if saved is not None:
        MachineProfile(path, {"schema": MachineProfile.SCHEMA, **saved}).save()

    profile = MachineProfile.load(path) if path.is_file() else MachineProfile(path)
    presets = {"data_root": str(tmp_path / "data"), "runs_root": str(tmp_path / "runs"), "results_root": str(tmp_path / "results"), **(answers or {})}

    return MachineSetup(repo, Prompter(presets, assume, logger_stub), StubProbe(FACTS, interpreter), profile, logger_stub, "Dune", interpreter if handed else None, verify=False, desktop=False)


def test_prompter_prefers_presets_then_defaults_then_the_terminal(logger_stub, monkeypatch):
    """A preset answer wins, unattended mode takes the default, and the terminal answer is validated until accepted."""
    preset = Prompter({"port": "9000"}, False, logger_stub)
    assert preset.ask("port", "Port", "8772", validate=Validators.port) == "9000"

    unattended = Prompter({}, True, logger_stub)
    assert unattended.ask("port", "Port", "8772", validate=Validators.port) == "8772"
    assert unattended.confirm("go", "Go?", True) is True

    typed = iter(["70000", "", "9001"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    terminal = Prompter({}, False, logger_stub)

    assert terminal.ask("port", "Port", "8772", validate=Validators.port) == "8772"
    assert terminal.ask("port", "Port", "8772", validate=Validators.port) == "9001"


def test_prompter_refuses_an_invalid_preset_or_default(logger_stub):
    """A bad preset or an unattended default that fails validation raises instead of being written."""
    with pytest.raises(ValueError, match="preset answer port"):
        Prompter({"port": "abc"}, False, logger_stub).ask("port", "Port", "8772", validate=Validators.port)

    with pytest.raises(ValueError, match="unattended"):
        Prompter({}, True, logger_stub).ask("port", "Port", "", validate=Validators.port)


def test_prompter_choose_accepts_numbers_and_free_text(logger_stub, monkeypatch):
    """A numbered option, a typed option and free text when nothing was detected are all accepted."""
    typed = iter(["2", "b", "free"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    prompter = Prompter({}, False, logger_stub)

    assert prompter.choose("k", "Pick", ["a", "b"], "a") == "b"
    assert prompter.choose("k", "Pick", ["a", "b"], "a") == "b"
    assert prompter.choose("k", "Pick", [], "")          == "free"


def test_validators(tmp_path):
    """Each validator accepts its valid input and names the problem otherwise."""
    (tmp_path / "f").write_text("")

    assert Validators.existing_dir(str(tmp_path))              is None
    assert Validators.existing_dir(str(tmp_path / "nope"))     is not None
    assert Validators.existing_file(str(tmp_path / "f"))       is None
    assert Validators.creatable_dir(str(tmp_path / "a" / "b")) is None
    assert Validators.creatable_dir("relative/path")           is not None
    assert Validators.creatable_dir(str(tmp_path / "f" / "x")) is not None
    assert Validators.optional_dir("")                         is None
    assert Validators.optional_file("")                        is None
    assert Validators.port("8772")                             is None
    assert Validators.port("80")                               is not None
    assert Validators.host("0.0.0.0")                          is None
    assert Validators.email("a@b.co")                          is None
    assert Validators.email("nope")                            is not None
    assert Validators.optional_email("")                       is None
    assert Validators.topic("my-topic_1")                      is None
    assert Validators.topic("bad topic")                       is not None
    assert Validators.gpu_list([0, 1])("0,1")                  is None
    assert Validators.gpu_list([0, 1])("2")                    is not None
    assert Validators.gpu_list([0, 1])("")                     is not None
    assert Validators.positive_int("3")                        is None
    assert Validators.positive_float("0.5")                    is None
    assert Validators.positive_float("x")                      is not None


def test_the_unattended_run_writes_a_complete_profile(tmp_path, logger_stub):
    """Defaults from the probe fill every section, the roots are created and the shell twin is written."""
    setup = _setup(tmp_path, logger_stub)
    setup.run()

    profile = MachineProfile.load(tmp_path / "cfg" / "machine.json")

    assert profile.machine["env_name"]    == "Dune"
    assert profile.machine["interpreter"] == str(tmp_path / "envs" / "Dune" / "bin" / "python")
    assert profile.machine["gpus"]        == FACTS["gpus"]
    assert profile.console                == {"host": ReferenceLayout.CONSOLE_HOST, "port": ReferenceLayout.CONSOLE_PORT}
    assert profile.roots["data"]          == str(tmp_path / "data")
    assert profile.roots["runs"]          == str(tmp_path / "runs")
    assert profile.roots["results"]       == str(tmp_path / "results")
    assert profile.roots["repo"]          == str(tmp_path / "repo")
    assert profile.dataset                == {"path": "", "parameters_path": ""}
    assert profile.hardware               == {"gpu_pool": [0, 1], "vram_budget_gb": 10.8, "cpu_workers": 8, "gif_workers": 4, "max_workers": 8}
    assert profile.ntfy["enabled"]        is False
    assert profile.mail["recipient"]      == ""
    assert (tmp_path / "data").is_dir() and (tmp_path / "runs").is_dir() and (tmp_path / "results").is_dir()
    assert f"export MLTEMPLATE_CONSOLE_PORT={ReferenceLayout.CONSOLE_PORT}" in profile.env_path.read_text()


def test_the_default_roots_are_the_checkout_reference_layout(tmp_path, logger_stub):
    """Without presets the data, runs and results roots default to the reference layout below the checkout."""
    setup = _setup(tmp_path, logger_stub)
    setup.prompter.answers = {}
    setup.run()

    profile = MachineProfile.load(tmp_path / "cfg" / "machine.json")

    assert profile.roots["data"]    == str(tmp_path / "repo" / ReferenceLayout.DATA_ROOT)
    assert profile.roots["runs"]    == str(tmp_path / "repo" / ReferenceLayout.RUNS_ROOT)
    assert profile.roots["results"] == str(tmp_path / "repo" / ReferenceLayout.RESULTS_ROOT)


def test_detected_datasets_and_parameter_files_become_the_defaults(tmp_path, logger_stub):
    """A dataset with a layout file below the data root and its extraction runs are offered and taken by default."""
    scene = tmp_path / "data" / "scene_b"
    (scene / "data").mkdir(parents=True)
    (scene / "data" / "dataset.json").write_text("{}")
    (scene / "params" / "run_1").mkdir(parents=True)
    (scene / "params" / "run_1" / "parameters.npy").write_text("")
    (tmp_path / "data" / "not_a_dataset").mkdir()

    _setup(tmp_path, logger_stub).run()
    profile = MachineProfile.load(tmp_path / "cfg" / "machine.json")

    assert profile.dataset["path"]            == str(scene)
    assert profile.dataset["parameters_path"] == str(scene / "params" / "run_1" / "parameters.npy")


def test_the_handed_over_interpreter_skips_the_question_unless_a_preset_names_one(tmp_path, logger_stub):
    """The installer's interpreter is recorded without asking; a preset or a missing hand-over goes through the question."""
    handed = _setup(tmp_path, logger_stub)
    handed.run()
    assert MachineProfile.load(tmp_path / "cfg" / "machine.json").machine["interpreter"] == str(tmp_path / "envs" / "Dune" / "bin" / "python")

    other = tmp_path / "other_python"
    other.write_text("")
    preset = _setup(tmp_path, logger_stub, {"interpreter": str(other)})
    preset.run()
    assert MachineProfile.load(tmp_path / "cfg" / "machine.json").machine["interpreter"] == str(other)

    detected = _setup(tmp_path, logger_stub, handed=False)
    detected.run()
    assert MachineProfile.load(tmp_path / "cfg" / "machine.json").machine["interpreter"] == str(tmp_path / "envs" / "Dune" / "bin" / "python")


def test_hardware_and_notification_answers_are_stored(tmp_path, logger_stub):
    """The GPU pool, VRAM budget, ntfy and mail answers land in their sections."""
    answers = {
        "ntfy_enabled": "yes", "ntfy_topic": "secret-topic", "ntfy_server": "https://ntfy.sh/",
        "mail_enabled": "yes", "mail_sender": "me@example.org", "mail_recipient": "you@example.org",
        "gpu_pool": "1", "vram_budget_gb": "9.5",
    }

    _setup(tmp_path, logger_stub, answers).run()
    profile = MachineProfile.load(tmp_path / "cfg" / "machine.json")

    assert profile.ntfy                       == {"enabled": True, "topic": "secret-topic", "server": "https://ntfy.sh"}
    assert profile.mail                       == {"sender": "me@example.org", "recipient": "you@example.org", "smtp_host": "smtp.gmail.com", "smtp_port": 465}
    assert profile.hardware["gpu_pool"]       == [1]
    assert profile.hardware["vram_budget_gb"] == 9.5


def test_a_rerun_defaults_to_the_saved_answers(tmp_path, logger_stub):
    """An unattended rerun keeps the previously saved console port, data root and ntfy topic."""
    saved = {
        "console" : {"host": "0.0.0.0", "port": 9100},
        "roots"   : {"data": str(tmp_path / "data")},
        "ntfy"    : {"enabled": True, "topic": "kept-topic", "server": "https://ntfy.sh"},
    }

    _setup(tmp_path, logger_stub, saved=saved).run()
    profile = MachineProfile.load(tmp_path / "cfg" / "machine.json")

    assert profile.console       == {"host": "0.0.0.0", "port": 9100}
    assert profile.ntfy["topic"] == "kept-topic"
    assert profile.ntfy["enabled"] is True


def test_an_invalid_preset_stops_the_run_before_anything_is_written(tmp_path, logger_stub):
    """A rejected preset raises and leaves no profile behind."""
    with pytest.raises(ValueError, match="console_port"):
        _setup(tmp_path, logger_stub, {"console_port": "12"}).run()

    assert not (tmp_path / "cfg" / "machine.json").exists()


def test_a_failing_nvidia_smi_is_reported_by_the_host_step(tmp_path, logger_stub, monkeypatch):
    """The host step shows nvidia-smi's failure in the table and warns with it instead of reporting a CPU machine."""
    setup    = _setup(tmp_path, logger_stub)
    broken   = {**FACTS, "driver": "", "driver_error": "nvidia-smi exited 9: could not communicate with the NVIDIA driver", "gpus": [], "torch_index": "cpu", "torch_note": "nvidia-smi exited 9: could not communicate with the NVIDIA driver"}
    tables   = []
    warnings = []

    setup.probe.facts_value = broken
    monkeypatch.setattr(setup.logger, "kv_table", lambda data, *args, **kwargs: tables.append(dict(data)))
    monkeypatch.setattr(setup.logger, "warning", warnings.append)

    setup.probe_host()

    assert tables[0]["driver"] == "nvidia-smi failed"
    assert tables[0]["torch"]  == "cpu (nvidia-smi exited 9: could not communicate with the NVIDIA driver)"
    assert len(warnings) == 1
    assert warnings[0].startswith("nvidia-smi exited 9")
    assert "linux-modules-nvidia" in warnings[0]


def test_a_cpu_machine_gets_the_plain_no_gpu_warning(tmp_path, logger_stub, monkeypatch):
    """Without nvidia-smi the host step keeps the old no-GPU warning and explains the CPU wheel."""
    setup    = _setup(tmp_path, logger_stub)
    tables   = []
    warnings = []

    setup.probe.facts_value = {**FACTS, "driver": "", "gpus": [], "torch_index": "cpu", "torch_note": "no NVIDIA driver: nvidia-smi is not installed"}
    monkeypatch.setattr(setup.logger, "kv_table", lambda data, *args, **kwargs: tables.append(dict(data)))
    monkeypatch.setattr(setup.logger, "warning", warnings.append)

    setup.probe_host()

    assert tables[0]["driver"] == "no NVIDIA driver"
    assert tables[0]["torch"]  == "cpu (no NVIDIA driver: nvidia-smi is not installed)"
    assert warnings == ["no GPU detected: the console and the tests work, training and inference need an NVIDIA GPU"]


def test_verification_quotes_torch_when_a_detected_gpu_is_unusable(tmp_path, logger_stub, monkeypatch):
    """The CUDA warning carries torch's own stderr explanation instead of a generic hint."""
    setup       = _setup(tmp_path, logger_stub)
    interpreter = tmp_path / "fake-python"
    warnings    = []

    interpreter.write_text("#!/bin/sh\necho 'torch/cuda/__init__.py:1: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080).' >&2\necho '{\"torch\": \"2.8.0\", \"cuda\": false, \"devices\": 0}'\n")
    interpreter.chmod(0o755)

    setup.verify_env = True
    setup.facts      = dict(FACTS)
    setup.profile.data["machine"] = {"interpreter": str(interpreter)}
    monkeypatch.setattr(setup.logger, "warning", warnings.append)

    setup.verify()

    assert len(warnings) == 1
    assert warnings[0].startswith("a GPU was detected but torch cannot use it: ")
    assert "The NVIDIA driver on your system is too old (found version 12080)" in warnings[0]
    assert "README section 1" in warnings[0]


def test_the_wizard_refuses_a_disabled_profile_location(logger_stub):
    """Without a location the wizard cannot save, which the profile reports."""
    with pytest.raises(RuntimeError, match=MachineProfile.ENV_VAR):
        MachineProfile(None, {"machine": {}}).save()
