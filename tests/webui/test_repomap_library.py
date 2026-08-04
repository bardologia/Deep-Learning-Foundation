from __future__ import annotations

import pytest

from repomap_library import RepoMapLibrary


ROLES  = {"entry", "orchestrator", "config", "io", "transform", "model", "data", "metric", "external"}
KINDS  = {"data", "control", "io"}
SCOPES = {"within", "cross"}


@pytest.fixture(scope="module")
def folders():
    return RepoMapLibrary().collect()


def _diagrams(folders):
    for folder in folders:
        for diagram in folder["diagrams"]:
            yield folder, diagram


def test_folders_and_diagrams_present(folders):
    assert folders
    assert all(f["diagrams"] for f in folders)


def test_folder_keys_unique(folders):
    keys = [f["folder"] for f in folders]
    assert len(keys) == len(set(keys))


def test_diagram_keys_unique(folders):
    keys = [d["key"] for _, d in _diagrams(folders)]
    assert len(keys) == len(set(keys))


def test_nodes_wellformed(folders):
    for _, d in _diagrams(folders):
        assert d["nodes"], d["key"]
        ids = [n["id"] for n in d["nodes"]]
        assert len(ids) == len(set(ids)), d["key"]
        for n in d["nodes"]:
            assert n["role"] in ROLES, (d["key"], n["role"])
            assert isinstance(n["col"], int) and n["col"] >= 0
            assert n["label"] and n["fn"] and n["module"]


def test_grid_positions_unique(folders):
    for _, d in _diagrams(folders):
        cells = [(n["col"], n.get("row", 0)) for n in d["nodes"]]
        assert len(cells) == len(set(cells)), d["key"]
        for n in d["nodes"]:
            assert isinstance(n.get("row", 0), int) and n.get("row", 0) >= 0, d["key"]


def test_edges_reference_existing_nodes(folders):
    for _, d in _diagrams(folders):
        ids = {n["id"] for n in d["nodes"]}
        for e in d["edges"]:
            assert e["from"] in ids, (d["key"], e["from"])
            assert e["to"] in ids, (d["key"], e["to"])
            assert e["kind"] in KINDS, (d["key"], e["kind"])


def test_artifacts_wellformed(folders):
    for _, d in _diagrams(folders):
        for a in d["artifacts"]:
            assert a["name"] and a["producer"]
            assert a["scope"] in SCOPES, (d["key"], a["scope"])
            assert isinstance(a["consumers"], list)
