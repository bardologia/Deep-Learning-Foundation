from __future__ import annotations

import pytest

from project_paths import ProjectPaths

from tests.webui.conftest import REPO_ROOT


def test_the_repository_root_is_accepted():
    paths = ProjectPaths(REPO_ROOT)

    assert paths.repo_root == REPO_ROOT


def test_a_root_without_the_repository_layout_is_refused(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        ProjectPaths(tmp_path)

    message = str(excinfo.value)

    assert str(tmp_path) in message
    assert "main" in message and "configuration" in message and "tools" in message


def test_a_root_missing_one_directory_names_only_that_one(tmp_path):
    for name in ("main", "configuration"):
        (tmp_path / name).mkdir()

    with pytest.raises(ValueError) as excinfo:
        ProjectPaths(tmp_path)

    assert "missing tools" in str(excinfo.value)


def test_the_configuration_signature_tracks_file_mtimes(tmp_path):
    paths  = ProjectPaths(REPO_ROOT)
    before = paths.config_signature()

    assert before
    assert before == paths.config_signature()
