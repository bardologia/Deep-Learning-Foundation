from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

from configuration.training.general.ablation import AblationCatalog
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


def test_ablation_leaves_expose_the_feature_plan(backbone_leaves):
    by_path  = {leaf["path"]: leaf for leaf in backbone_leaves}
    features = ast.literal_eval(by_path["ablation_features"]["value"])

    assert [feature["label"] for feature in features] == list(AblationCatalog.DEFAULT_ORDER)
    assert by_path["ablation_include_full"]["value"] == "True"
    assert "ablation_catalog" not in by_path
