"""Tests for the cube page's front-end lifecycle, run against the real cube_view.js.

This module runs ``cube_view.js`` under node with no DOM through
``webui/tools/visual/cube_view_checks.js``. The cube view keeps decoded slice images as
ImageBitmaps, whose pixel memory is only returned when they are closed, so evicted,
cleared, released and replaced bitmaps must close while a bitmap a sweep still shows is
kept. The point cloud's section-cut plane must bound itself by the cloud's true extent,
hide the points beyond it, and stop its sweep loop when the view or the tab is left.
"""

from __future__ import annotations

import json
import re
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


@NEEDS_JS
def test_value_bounds_skip_non_finite_values():
    """The cloud's bounds helper returns the finite extremes of one column, or None when there is none."""
    bounds = _check("cut")["bounds"]

    assert bounds == {"mixed": [-2, 3], "empty": None}


@NEEDS_JS
def test_cut_spec_spans_the_grid_for_row_and_col_and_the_true_mu_extent_for_z():
    """The cut threshold runs over the full row or col count, and over the cloud's exact mu range for z."""
    observed = _check("cut")
    specs    = observed["specs"]

    assert specs["off@50"] is None
    assert specs["row@50"] == {"axis": "row", "frac": 0.5, "threshold": 5}
    assert specs["col@25"] == {"axis": "col", "frac": 0.25, "threshold": 5}
    assert specs["z@50"]["threshold"] == 5 and specs["z@0"]["threshold"] == -5
    assert observed["unbounded"] is None


@NEEDS_JS
def test_cut_hides_points_beyond_the_plane_and_draws_it():
    """With a cut, points past the plane are skipped, the readout counts the shown ones and names the cut."""
    observed = _check("cut")

    assert observed["cutDraw"]["text"].startswith("2 of 3 components · cut row @ 50% ·")
    assert observed["cutDraw"]["fills"] == 1
    assert observed["openDraw"]["text"].startswith("3 of 3 components · drag")
    assert observed["openDraw"]["fills"] == 1


@NEEDS_JS
def test_cut_sweep_runs_a_triangle_wave_from_the_current_position():
    """Ticking the sweep with no axis picks row, and the plane moves 50, 100, 0 percent over one period."""
    observed = _check("cut")

    assert observed["started"] == {"axis": "row", "running": True}
    assert observed["positions"] == ["50", "100", "0"]


@NEEDS_JS
def test_cut_sweep_stops_when_the_cloud_or_the_cube_tab_is_left():
    """Leaving the cloud, the Cube tab, the cloud view, reloading a cube or switching the axis off ends the loop."""
    observed = _check("cut")

    for key in ("left", "viewLeft", "offView"):
        assert observed[key] == {"running": False, "pending": 0}, key

    assert observed["reconfigured"] == {"running": False, "muBounds": None}
    assert observed["axisOff"] == {"running": False, "sweep": False}


def test_cut_controls_are_on_the_page_and_wired_to_the_cloud():
    """The cloud bar carries the cut axis select, position slider and sweep box, and app.js hands them to CubeCloud."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    assert '<input type="range" id="cube-cloud-cut-pos"' in html
    assert '<input type="checkbox" id="cube-cloud-cut-sweep"' in html
    for value in ("off", "row", "col", "z"):
        assert f'<option value="{value}"' in html.split('id="cube-cloud-cut-axis"')[1].split("</select>")[0]

    for ref, element in (("cutAxis", "cube-cloud-cut-axis"), ("cutPos", "cube-cloud-cut-pos"), ("cutSweep", "cube-cloud-cut-sweep")):
        assert re.search(rf'{ref}\s*: document\.getElementById\("{element}"\)', app), ref
