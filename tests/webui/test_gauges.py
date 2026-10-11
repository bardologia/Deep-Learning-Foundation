"""Tests for the Home board's canvas gauges, run against the real gauges.js.

This module runs ``canvas_base.js`` and ``gauges.js`` under node with no DOM through
``webui/tools/visual/gauge_checks.js``. Canvas text drawn before the web fonts arrive
keeps the fallback face until the next reading, so every gauge redraws once
``document.fonts.ready`` settles. A dial sweeps to full scale before its first reading,
holds the readings that arrive during the sweep and then settles on the latest; it keeps
a trail of at most ``DialGauge.TRAIL`` readings, clears the sweep timer when disposed,
and keeps its frame loop running only while the needle sits in its last alarm zone.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS  = WEBUI_ROOT / "tools" / "visual" / "gauge_checks.js"
NEEDS_JS = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run gauges.js")


def _check(scenario: str) -> dict:
    """Runs one harness scenario and returns what it observed.

    Args:
        scenario : Name of the GaugeChecks method to run.

    Returns:
        The JSON object the scenario printed.
    """
    result = subprocess.run(["node", str(HARNESS), scenario], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@NEEDS_JS
def test_gauges_redraw_once_the_web_fonts_are_ready():
    """A dial and a linear meter each draw once more when document.fonts.ready resolves."""
    observed = _check("fonts")

    assert observed["after"]["dial"] == observed["before"]["dial"] + 1
    assert observed["after"]["meter"] == observed["before"]["meter"] + 1


@NEEDS_JS
def test_dial_sweeps_to_full_scale_before_settling_on_the_latest_reading():
    """The first reading sweeps the needle to full scale and is held, later ones replace it until the sweep ends."""
    observed = _check("sweep")

    assert observed["first"] == {"target": 100, "held": 42, "trail": [], "timer": True, "frames": 1}
    assert observed["during"] == {"target": 100, "held": 55, "trail": []}
    assert observed["landed"] == {"target": 55, "held": None, "trail": [55], "timer": None}


@NEEDS_JS
def test_dial_trail_keeps_only_the_latest_readings():
    """The trail of recent readings is capped at DialGauge.TRAIL and ends on the newest one."""
    capped = _check("sweep")["capped"]

    assert capped["length"] == capped["cap"] == 24
    assert capped["last"] == 29


@NEEDS_JS
def test_disposing_a_dial_mid_sweep_clears_its_timer():
    """A dial disposed during its ignition sweep cancels the timer that would apply the held reading."""
    observed = _check("dispose")

    assert observed["cleared"] is True
    assert observed["pending"] is False


@NEEDS_JS
def test_dial_keeps_animating_only_while_in_alarm():
    """The frame loop stops once the needle lands, except while it sits in the last zone, where the zone breathes."""
    observed = _check("alarm")

    assert observed["calm"]["value"] == 40 and observed["calm"]["raf"] is None
    assert observed["hot"]["value"] == 96 and observed["hot"]["alarm"] is True and observed["hot"]["raf"] is True
    assert observed["hot"]["frames"] == 600
    assert observed["cool"]["value"] == 20 and observed["cool"]["raf"] is None
