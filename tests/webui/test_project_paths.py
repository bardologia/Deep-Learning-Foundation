from __future__ import annotations


from project_paths import ProjectPaths

from tests.webui.conftest import REPO_ROOT


def test_the_repository_root_is_accepted():
    paths = ProjectPaths(REPO_ROOT)

    assert paths.repo_root == REPO_ROOT


def test_the_configuration_signature_tracks_file_mtimes(tmp_path):
    paths  = ProjectPaths(REPO_ROOT)
    before = paths.config_signature()

    assert before
    assert before == paths.config_signature()
