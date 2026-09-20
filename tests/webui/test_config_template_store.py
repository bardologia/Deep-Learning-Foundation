"""Tests for the config-template store and the router serving it.

Covers validation of a save payload, one-file-per-template persistence under
``logs/config_templates``, the per-entry-point uniqueness of template names and
the replace flag that overwrites one in place, newest-first listing, renaming
with its own clash guard, and deletion guards against malformed ids. Also walks
the four routes end to end through a fake exchange.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config_template_store  import ConfigTemplateStore
from routers.dispatch       import HttpExchange
from routers.launch_routers import ConfigTemplateRouter
from web_logger             import WebLogger

from tests.webui.conftest import FakeHandler


class TemplatePaths:
    """Minimal project-paths stand-in exposing a template directory and a script whitelist.

    Attributes:
        templates_dir : Directory the store writes one JSON file per template into.
        scripts       : Entry-point keys treated as present in the repository.
    """

    def __init__(self, root: Path, scripts: tuple = ("train_backbone", "train_dual")) -> None:
        """Points the template directory at ``<root>/logs/config_templates`` and records the known scripts."""
        self.templates_dir = root / "logs" / "config_templates"
        self.scripts       = scripts

    def has_script(self, key: str) -> bool:
        """Returns whether the entry-point key is one of the configured scripts."""
        return key in self.scripts


@pytest.fixture
def store(tmp_path):
    """Returns a store writing into a temporary template directory."""
    return ConfigTemplateStore(TemplatePaths(tmp_path), WebLogger())


def _payload(**extra) -> dict:
    """Returns a valid save payload for ``train_backbone`` with the given fields overridden."""
    base = {"script_key": "train_backbone", "title": "Train backbone", "name": "wide patches", "overrides": {"dataset.patch_size": "(32, 16)", "training.seed": "7"}}
    return {**base, **extra}


def _post(router, path: str, body: dict, handler_id: str = "") -> FakeHandler:
    """Routes one POST body through a router method addressed by the wildcard id."""
    handler  = FakeHandler("POST", path, json.dumps(body).encode("utf-8"))
    exchange = HttpExchange.of(handler)
    exchange.read_body()

    if path.endswith("/rename"):
        router.rename(exchange, handler_id)
    elif path.endswith("/delete"):
        router.remove(exchange, handler_id)
    else:
        router.save(exchange)

    return handler


def test_save_rejects_unknown_script(store):
    """Saving a template whose script key has no entry point is refused by name."""
    assert store.save(_payload(script_key="missing")) == {"ok": False, "error": "unknown script 'missing'"}


def test_save_rejects_blank_name(store):
    """A whitespace-only name is refused, since the name is the template's handle."""
    assert store.save(_payload(name="  ")) == {"ok": False, "error": "no template name given"}


def test_entry_carries_the_overrides_and_its_entry_point(store):
    """A stored template keeps the entry point, title, name and overrides it was saved with."""
    entry = store.save(_payload())["entry"]

    assert entry["script"]    == "train_backbone"
    assert entry["title"]     == "Train backbone"
    assert entry["name"]      == "wide patches"
    assert entry["overrides"] == {"dataset.patch_size": "(32, 16)", "training.seed": "7"}


def test_save_writes_one_file_per_template(store, tmp_path):
    """Two differently named templates produce two separate JSON files."""
    first  = store.save(_payload())
    second = store.save(_payload(name="narrow patches"))

    files = list((tmp_path / "logs" / "config_templates").glob("*.json"))
    assert first["ok"] and second["ok"]
    assert len(files) == 2


def test_a_repeated_name_on_the_same_entry_point_is_refused(store):
    """Saving over an existing name without the replace flag reports the clash instead of writing."""
    store.save(_payload())
    result = store.save(_payload(overrides={"training.seed": "9"}))

    assert result == {"ok": False, "error": "a template named 'wide patches' already exists for train_backbone"}


def test_the_same_name_on_another_entry_point_is_a_separate_template(store):
    """Names are unique per entry point, so the same name is free on a different script."""
    first  = store.save(_payload())["entry"]
    second = store.save(_payload(script_key="train_dual"))["entry"]

    assert first["template_id"] != second["template_id"]
    assert len(store.list()["templates"]) == 2


def test_replace_overwrites_the_template_in_place(store, tmp_path):
    """A replacing save keeps the identifier and file, and carries the new overrides."""
    first  = store.save(_payload())["entry"]
    result = store.save(_payload(overrides={"training.seed": "9"}, replace=True))

    assert result["replaced"] is True
    assert result["entry"]["template_id"] == first["template_id"]
    assert result["entry"]["overrides"]   == {"training.seed": "9"}
    assert len(list((tmp_path / "logs" / "config_templates").glob("*.json"))) == 1


def test_list_persists_across_instances(store, tmp_path):
    """A freshly constructed store reads back the templates written by an earlier instance."""
    saved = store.save(_payload())["entry"]

    fresh = ConfigTemplateStore(TemplatePaths(tmp_path), WebLogger())
    assert fresh.list()["templates"] == [saved]


def test_list_orders_newest_first(store, tmp_path):
    """Listing orders templates by their saved_at timestamp, newest first."""
    older = store.save(_payload(name="older"))["entry"]
    store.save(_payload(name="newer"))

    stale = {**older, "saved_at": "2026-01-01T00:00:00"}
    (tmp_path / "logs" / "config_templates" / f"{older['template_id']}.json").write_text(json.dumps(stale) + "\n")

    assert [entry["name"] for entry in store.list()["templates"]] == ["newer", "older"]


def test_get_rejects_unknown_and_malformed_ids(store):
    """Unknown ids and path-traversal ids both resolve to None."""
    assert store.get("0123456789ab") is None
    assert store.get("../escape") is None


def test_rename_keeps_the_identifier_and_overrides(store):
    """Renaming rewrites the same file with a new name and leaves the overrides untouched."""
    saved   = store.save(_payload())["entry"]
    renamed = store.rename(saved["template_id"], "  wide patches, long  ")["entry"]

    assert renamed["template_id"] == saved["template_id"]
    assert renamed["name"]        == "wide patches, long"
    assert renamed["overrides"]   == saved["overrides"]
    assert store.get(saved["template_id"])["name"] == "wide patches, long"


def test_rename_onto_a_taken_name_is_refused(store):
    """A rename colliding with another template of the same entry point reports the clash."""
    store.save(_payload(name="taken"))
    saved = store.save(_payload(name="free"))["entry"]

    assert store.rename(saved["template_id"], "taken") == {"ok": False, "error": "a template named 'taken' already exists for train_backbone"}


def test_rename_to_its_own_name_is_allowed(store):
    """Renaming a template to the name it already carries is not treated as a clash."""
    saved = store.save(_payload())["entry"]
    assert store.rename(saved["template_id"], "wide patches")["ok"] is True


def test_rename_rejects_blank_names_and_unknown_ids(store):
    """A blank name and an absent template are both refused by name."""
    assert store.rename("0123456789ab", "x") == {"ok": False, "error": "template not found"}
    assert store.rename("0123456789ab", " ") == {"ok": False, "error": "no template name given"}


def test_delete_removes_the_template(store, tmp_path):
    """Deleting a template drops it from the listing and removes its file."""
    saved = store.save(_payload())["entry"]

    assert store.delete(saved["template_id"]) == {"ok": True}
    assert store.list()["templates"] == []
    assert not (tmp_path / "logs" / "config_templates" / f"{saved['template_id']}.json").exists()


def test_delete_rejects_unknown_and_malformed_ids(store):
    """Deleting an unknown or path-traversal id reports not-found instead of touching the filesystem."""
    assert store.delete("0123456789ab") == {"ok": False, "error": "template not found"}
    assert store.delete("../escape")    == {"ok": False, "error": "template not found"}


def test_router_lists_saves_renames_and_deletes(store):
    """The four routes cover the round trip a launch page drives, with failures answered as 400."""
    router = ConfigTemplateRouter(store)

    saved = _post(router, "/api/config-templates", _payload())
    assert saved.status == 200
    entry = saved.payload()["entry"]

    clash = _post(router, "/api/config-templates", _payload())
    assert clash.status == 400

    listing  = FakeHandler("GET", "/api/config-templates")
    exchange = HttpExchange.of(listing)
    router.listing(exchange)
    assert [item["template_id"] for item in listing.payload()["templates"]] == [entry["template_id"]]

    renamed = _post(router, f"/api/config-templates/{entry['template_id']}/rename", {"name": "renamed"}, entry["template_id"])
    assert renamed.payload()["entry"]["name"] == "renamed"

    removed = _post(router, f"/api/config-templates/{entry['template_id']}/delete", {}, entry["template_id"])
    assert removed.payload() == {"ok": True}
    assert store.list()["templates"] == []
