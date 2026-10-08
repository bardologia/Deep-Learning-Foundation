"""Tests for the sequence editor's start, stop and release actions.

Each action posts to the sequence router and repaints the board when the reply
lands. A second click while the first request is in flight posted the same
action again, so the second start or release raced the first and came back as
an error toast. The editor now ignores clicks while an action is in flight.

The actions live in the browser, so this module runs the real editor module
under node against a stub DOM and a stub router that answers on the next tick.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS = WEBUI_ROOT / "tools" / "visual" / "sequence_actions.js"


def _probe() -> dict:
    """Returns, per action, the posts three overlapping clicks sent and the posts one later click sent.

    Returns:
        Mapping of action name to its ``concurrent`` and ``after`` post counts, the guard flag left behind and the errors the clicks raised.
    """
    result = subprocess.run(["node", str(HARNESS)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_overlapping_clicks_post_each_action_once():
    """Three clicks on start, stop or release while the first request is in flight post that action once."""
    for action, outcome in _probe().items():
        assert outcome["concurrent"] == 1, action


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the sequence editor")
def test_the_guard_lifts_once_the_action_lands():
    """A click after the reply has landed posts again, so the guard never locks the editor, and no click raises."""
    for action, outcome in _probe().items():
        assert outcome["after"] == 1, action
        assert outcome["busy"] is False, action
        assert outcome["errors"] == [], action
