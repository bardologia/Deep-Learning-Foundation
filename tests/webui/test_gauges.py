
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS  = WEBUI_ROOT / "tools" / "visual" / "gauge_checks.js"
NEEDS_JS = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run gauges.js")


def _check(scenario: str) -> dict:
    result = subprocess.run(["node", str(HARNESS), scenario], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@NEEDS_JS
def test_gauges_redraw_once_the_web_fonts_are_ready():
    observed = _check("fonts")

    assert observed["after"]["dial"] == observed["before"]["dial"] + 1
    assert observed["after"]["meter"] == observed["before"]["meter"] + 1


@NEEDS_JS
def test_dial_sweeps_to_full_scale_before_settling_on_the_latest_reading():
    observed = _check("sweep")

    assert observed["first"] == {"target": 100, "held": 42, "trail": [], "timer": True, "frames": 1}
    assert observed["during"] == {"target": 100, "held": 55, "trail": []}
    assert observed["landed"] == {"target": 55, "held": None, "trail": [55], "timer": None}


@NEEDS_JS
def test_dial_trail_keeps_only_the_latest_readings():
    capped = _check("sweep")["capped"]

    assert capped["length"] == capped["cap"] == 24
    assert capped["last"] == 29


@NEEDS_JS
def test_disposing_a_dial_mid_sweep_clears_its_timer():
    observed = _check("dispose")

    assert observed["cleared"] is True
    assert observed["pending"] is False


@NEEDS_JS
def test_dial_keeps_animating_only_while_in_alarm():
    observed = _check("alarm")

    assert observed["calm"]["value"] == 40 and observed["calm"]["raf"] is None
    assert observed["hot"]["value"] == 96 and observed["hot"]["alarm"] is True and observed["hot"]["raf"] is True
    assert observed["hot"]["frames"] == 600
    assert observed["cool"]["value"] == 20 and observed["cool"]["raf"] is None
