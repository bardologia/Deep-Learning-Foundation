"""Tests for the sequence editor's start, stop and release actions and its block drawer fetches.

Each action posts to the sequence router and repaints the board when the reply
lands. A second click while the first request is in flight posted the same
action again, so the second start or release raced the first and came back as
an error toast. The editor now ignores clicks while an action is in flight.

Opening a block drawer fetched the script's resolved configuration and the
model list again on every open. The editor now keeps each successful reply for
the page session and retries a failed one on the next open.

Both live in the browser, so this module runs the real editor module under node
against a stub DOM and a stub router that answers on the next tick.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS = WEBUI_ROOT / "tools" / "visual" / "sequence_actions.js"
ACTIONS = ("start", "stop", "release")


def _probe() -> dict:
    """Returns the action click counts and the drawer fetch counts the harness recorded.

    Returns:
        Mapping of each action to its ``concurrent`` and ``after`` post counts, the guard flag left behind and the errors the clicks raised, plus a ``drawer`` entry with the configuration and model GET counts over one failed and three later opens.
    """
    result = subprocess.run(["node", str(HARNESS)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_overlapping_clicks_post_each_action_once():
    """Three clicks on start, stop or release while the first request is in flight post that action once."""
    outcomes = _probe()

    for action in ACTIONS:
        assert outcomes[action]["concurrent"] == 1, action


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_the_guard_lifts_once_the_action_lands():
    """A click after the reply has landed posts again, so the guard never locks the editor, and no click raises."""
    outcomes = _probe()

    for action in ACTIONS:
        assert outcomes[action]["after"] == 1, action
        assert outcomes[action]["busy"] is False, action
        assert outcomes[action]["errors"] == [], action


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_the_drawer_fetches_a_script_config_and_model_list_once():
    """After a failed first open, three drawer opens fetch the configuration and the model list once more in total."""
    drawer = _probe()["drawer"]

    assert drawer["failed"] is True
    assert drawer["configGets"] == 2
    assert drawer["modelGets"] == 2
