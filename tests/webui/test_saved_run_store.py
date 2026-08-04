from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from saved_run_store import SavedRunStore
from web_logger      import WebLogger

from tests.webui.conftest import ARGS_DUMP, wait_for_status


class SavedRunPaths:

    def __init__(self, root: Path) -> None:
        self.saved_runs_dir = root / "logs" / "saved_runs"

    def has_script(self, key: str) -> bool:
        return key in ("train_backbone", "infer_backbone")


@pytest.fixture
def store(tmp_path):
    return SavedRunStore(SavedRunPaths(tmp_path), WebLogger())


def _payload(**extra) -> dict:
    base = {"script_key": "train_backbone", "title": "Train backbone", "name": "wide patches", "interpreter": "/usr/bin/python3", "overrides": {"training.seed": "7"}, "follow_up": None, "detach": True}
    return {**base, **extra}


def test_save_rejects_unknown_script(store):
    result = store.save(_payload(script_key="missing"))
    assert result == {"ok": False, "error": "unknown script 'missing'"}


def test_save_rejects_unknown_follow_up(store):
    result = store.save(_payload(follow_up="missing"))
    assert result == {"ok": False, "error": "unknown follow-up script 'missing'"}


def test_save_rejects_missing_interpreter(store):
    result = store.save(_payload(interpreter=""))
    assert result == {"ok": False, "error": "no interpreter given"}


def test_save_writes_one_file_per_entry(store, tmp_path):
    first  = store.save(_payload())
    second = store.save(_payload(name="narrow patches"))

    files = list((tmp_path / "logs" / "saved_runs").glob("*.json"))
    assert first["ok"] and second["ok"]
    assert len(files) == 2


def test_entry_carries_launchable_fields(store):
    entry = store.save(_payload(follow_up="infer_backbone", detach=False))["entry"]

    assert entry["script"]      == "train_backbone"
    assert entry["title"]       == "Train backbone"
    assert entry["name"]        == "wide patches"
    assert entry["interpreter"] == "/usr/bin/python3"
    assert entry["overrides"]   == {"training.seed": "7"}
    assert entry["follow_up"]   == "infer_backbone"
    assert entry["detach"]      is False


def test_blank_name_falls_back_to_title(store):
    entry = store.save(_payload(name="  "))["entry"]
    assert entry["name"] == ""
    assert entry["title"] == "Train backbone"


def test_list_persists_across_instances(store, tmp_path):
    saved = store.save(_payload())["entry"]

    fresh = SavedRunStore(SavedRunPaths(tmp_path), WebLogger())
    assert fresh.list()["saved"] == [saved]


def test_list_orders_newest_first(store, tmp_path):
    older = store.save(_payload(name="older"))["entry"]
    newer = store.save(_payload(name="newer"))["entry"]

    stale = {**older, "saved_at": "2026-01-01T00:00:00"}
    (tmp_path / "logs" / "saved_runs" / f"{older['saved_id']}.json").write_text(json.dumps(stale) + "\n")

    names = [entry["name"] for entry in store.list()["saved"]]
    assert names == ["newer", "older"]


def test_get_returns_saved_entry(store):
    saved = store.save(_payload())["entry"]
    assert store.get(saved["saved_id"]) == saved


def test_get_rejects_unknown_and_malformed_ids(store):
    assert store.get("0123456789ab") is None
    assert store.get("../escape") is None


def test_delete_removes_entry(store, tmp_path):
    saved  = store.save(_payload())["entry"]
    result = store.delete(saved["saved_id"])

    assert result == {"ok": True}
    assert store.list()["saved"] == []
    assert not (tmp_path / "logs" / "saved_runs" / f"{saved['saved_id']}.json").exists()


def test_delete_rejects_unknown_and_malformed_ids(store):
    assert store.delete("0123456789ab") == {"ok": False, "error": "saved run not found"}
    assert store.delete("../escape")    == {"ok": False, "error": "saved run not found"}


def test_saved_entry_launches_through_process_manager(make_manager, tmp_path):
    manager = make_manager({"args_dump": ARGS_DUMP})
    store   = SavedRunStore(manager.paths, WebLogger())

    entry  = store.save({"script_key": "args_dump", "title": "Args dump", "name": "", "interpreter": sys.executable, "overrides": {"training.seed": "7"}, "follow_up": None, "detach": False})["entry"]
    result = manager.launch(entry["script"], entry["interpreter"], entry["overrides"], entry["follow_up"], entry["detach"])

    assert result["ok"]
    assert wait_for_status(manager, result["job_id"], "finished")
    assert (tmp_path / "argv.txt").read_text() == "--training.seed 7"
