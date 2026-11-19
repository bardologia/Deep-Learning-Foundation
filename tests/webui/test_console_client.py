
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("console_client_test.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the console tile suite")
def test_the_console_tiles_pass_their_node_suite():
    result = subprocess.run(["node", str(SCRIPT)], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "all " in result.stdout and " passing" in result.stdout
