"""Tests for the cube page's front-end resource handling, run against the real cube_view.js.

The cube view keeps decoded slice images as ImageBitmaps, whose pixel memory is only
returned when they are closed. This module runs ``cube_view.js`` under node with no DOM
through ``webui/tools/visual/cube_view_checks.js`` and checks that evicted, cleared,
released and replaced bitmaps are closed while a bitmap a sweep still shows is kept.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS  = WEBUI_ROOT / "tools" / "visual" / "cube_view_checks.js"
NEEDS_JS = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run cube_view.js")


def _check(scenario: str) -> dict:
    """Runs one harness scenario and returns what it observed.

    Args:
        scenario : Name of the CubeViewChecks method to run.

    Returns:
        The JSON object the scenario printed.
    """
    result = subprocess.run(["node", str(HARNESS), scenario], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@NEEDS_JS
def test_cache_trim_closes_evicted_bitmaps_but_keeps_the_one_a_sweep_shows():
    """Trimming the sweep cache closes what it evicts unless a sweep panel still draws that bitmap."""
    trim = _check("bitmaps")["trim"]

    assert trim["evicted_closed"]
    assert not trim["shown_closed"]
    assert not trim["kept_closed"]
    assert trim["keys"] == ["kept"] and trim["bytes"] == 1


@NEEDS_JS
def test_cache_clear_closes_every_unheld_bitmap():
    """Clearing the sweep cache closes its bitmaps and resets the byte count."""
    clear = _check("bitmaps")["clear"]

    assert clear == {"kept_closed": True, "size": 0, "bytes": 0}


@NEEDS_JS
def test_released_and_replaced_slice_bitmaps_are_closed():
    """Resetting a cut panel closes its bitmap, and a new slice closes the one it replaces."""
    observed = _check("bitmaps")

    assert observed["release"] == {"closed": True, "bitmap": None, "key": None, "space": None}
    assert observed["replace"] == {"stale_closed": True, "fresh_closed": False, "holds_fresh": True, "key": "1:2"}
