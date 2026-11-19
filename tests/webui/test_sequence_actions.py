
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS = WEBUI_ROOT / "tools" / "visual" / "sequence_actions.js"
ACTIONS = ("start", "stop", "release")


def _probe() -> dict:
    result = subprocess.run(["node", str(HARNESS)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_overlapping_clicks_post_each_action_once():
    outcomes = _probe()

    for action in ACTIONS:
        assert outcomes[action]["concurrent"] == 1, action


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_the_guard_lifts_once_the_action_lands():
    outcomes = _probe()

    for action in ACTIONS:
        assert outcomes[action]["after"] == 1, action
        assert outcomes[action]["busy"] is False, action
        assert outcomes[action]["errors"] == [], action


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_the_drawer_fetches_a_script_config_and_model_list_once():
    drawer = _probe()["drawer"]

    assert drawer["failed"] is True
    assert drawer["configGets"] == 2
    assert drawer["modelGets"] == 2


def test_every_private_call_in_the_sequence_editor_resolves():
    js      = WEBUI_ROOT / "static" / "js"
    source  = (js / "sequences.js").read_text()
    pattern = re.compile(r"^  (?:static )?(?:async )?(_\w+)\(", re.M)
    defined = set(pattern.findall(source)) | set(pattern.findall((js / "config_form.js").read_text()))

    assert sorted(set(re.findall(r"this\.(_\w+)\(", source)) - defined) == []
