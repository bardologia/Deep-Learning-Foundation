
from __future__ import annotations

from unittest.mock import Mock
import ast
import json
import subprocess
import sys
import threading
import time

from types import SimpleNamespace

import pytest

import script_config_resolver

from project_paths                           import ProjectPaths
from script_config_resolver                  import ScriptConfigResolver

from tests.webui.conftest import REPO_ROOT


@pytest.fixture(scope="module")
def backbone_leaves():
    argv = [sys.executable, "-c", ScriptConfigResolver.BOOTSTRAP, str(REPO_ROOT), "configuration.training", "BackboneEntryConfig"]
    proc = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_container_leaves_render_as_python_literals(backbone_leaves):
    containers = [leaf for leaf in backbone_leaves if leaf["type"] in ("list", "tuple", "dict")]

    assert containers
    for leaf in containers:
        assert "<" not in leaf["value"], leaf["path"]
        ast.literal_eval(leaf["value"])


class RecordingPaths:

    def __init__(self, root) -> None:
        self.repo_root  = root
        self.config_dir = root / "configuration"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def tree_signature(self, paths) -> tuple:
        return ProjectPaths.tree_signature(self, paths)

    def profile_signature(self) -> tuple:
        return ()

    def script_entry(self, key: str) -> dict:
        return {
            "path"          : self.repo_root / "main" / f"{key}.py",
            "rel"           : f"main/{key}.py",
            "config_module" : "configuration.training",
            "config_class"  : "BackboneEntryConfig",
        }


class DetectingPaths(RecordingPaths):

    def script_entry(self, key: str) -> dict:
        return {
            "path"          : self.repo_root / "main" / f"{key}.py",
            "rel"           : f"main/{key}.py",
            "config_module" : None,
            "config_class"  : None,
        }


def test_entry_detection_reparses_only_when_the_script_changes(tmp_path):
    script = tmp_path / "main" / "analyze.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("from configuration.analysis import AnalysisConfig\nConfigCli(AnalysisConfig())\n", encoding="utf-8")

    resolver = ScriptConfigResolver(DetectingPaths(tmp_path))
    parses   = []
    detect   = resolver._detect_entry_config

    resolver._detect_entry_config = lambda path: parses.append(path) or detect(path)

    assert resolver.entry_config("analyze") == {"module": "configuration.analysis", "class": "AnalysisConfig"}
    assert resolver.entry_config("analyze") == {"module": "configuration.analysis", "class": "AnalysisConfig"}
    assert len(parses) == 1

    script.write_text("from configuration.other import OtherConfig\nConfigCli(OtherConfig())\n", encoding="utf-8")

    assert resolver.entry_config("analyze") == {"module": "configuration.other", "class": "OtherConfig"}
    assert len(parses) == 2


class CountingEvent(threading.Event):

    waiters = 0

    def wait(self, timeout=None) -> bool:
        CountingEvent.waiters += 1
        return super().wait(timeout)


def test_concurrent_resolves_share_a_single_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(script_config_resolver, "threading", SimpleNamespace(Event=CountingEvent, Lock=threading.Lock))
    monkeypatch.setattr(CountingEvent, "waiters", 0)

    resolver = ScriptConfigResolver(RecordingPaths(tmp_path))
    started  = threading.Event()
    calls    = []

    def slow_bootstrap(entry, interpreter):
        calls.append(interpreter)
        started.set()
        deadline = time.monotonic() + 5.0
        while CountingEvent.waiters < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        return {"ok": True, "module": entry["module"], "config_class": entry["class"], "leaves": []}

    resolver._run_bootstrap = slow_bootstrap

    results = {}
    threads = [threading.Thread(target=lambda index=index: results.__setitem__(index, resolver.resolve("train_backbone", "python"))) for index in range(4)]

    threads[0].start()
    assert started.wait(timeout=5.0)

    for thread in threads[1:]:
        thread.start()

    for thread in threads:
        thread.join(timeout=5.0)

    assert CountingEvent.waiters == 3
    assert len(calls) == 1
    assert all(results[index]["ok"] for index in range(4))


def test_a_changed_configuration_signature_invalidates_the_cache(tmp_path):
    resolver = ScriptConfigResolver(RecordingPaths(tmp_path))
    calls    = []

    def counting_bootstrap(entry, interpreter):
        calls.append(interpreter)
        return {"ok": True, "module": entry["module"], "config_class": entry["class"], "leaves": []}

    resolver._run_bootstrap = counting_bootstrap

    resolver.resolve("train_backbone", "python")
    resolver.resolve("train_backbone", "python")
    assert len(calls) == 1

    (tmp_path / "configuration" / "touched.py").write_text("x = 1\n", encoding="utf-8")
    resolver.resolve("train_backbone", "python")

    assert len(calls) == 2


def test_a_failing_bootstrap_raises_for_the_waiting_callers(tmp_path):
    resolver = ScriptConfigResolver(RecordingPaths(tmp_path))
    started  = threading.Event()
    release  = threading.Event()

    def failing_bootstrap(entry, interpreter):
        started.set()
        release.wait(timeout=5.0)
        Mock(side_effect=MemoryError("bootstrap died"))()

    resolver._run_bootstrap = failing_bootstrap

    outcomes = {}

    def call(index):
        try:
            outcomes[index] = resolver.resolve("train_backbone", "python")
        except MemoryError as exc:
            outcomes[index] = exc

    threads = [threading.Thread(target=call, args=(index,)) for index in range(3)]

    threads[0].start()
    assert started.wait(timeout=5.0)

    for thread in threads[1:]:
        thread.start()

    release.set()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(isinstance(outcomes[index], MemoryError) for index in range(3))
