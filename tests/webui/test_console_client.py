"""Runs the browser-side console tile tests under node.

``console_client_test.mjs`` loads ``static/js/console.js`` with no DOM and
drives its tiles through fake terminals and a fake page stream hub: a tile
follows its job over the hub, notes a lost link once, a truncation and an
unavailable stream, and resets a sub-run terminal when its tail replays. The
sizing cases check that a tile sets its quarter-pixel font size and its
dimensions through the shared hidden gauge before opening its terminal, that
refitting at the same size leaves the font alone, and that the console fits
its tiles one per animation frame. This test runs that script and reports its
outcome.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("console_client_test.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the console tile suite")
def test_the_console_tiles_pass_their_node_suite():
    """Every case of the node suite passes."""
    result = subprocess.run(["node", str(SCRIPT)], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "all " in result.stdout and " passing" in result.stdout
