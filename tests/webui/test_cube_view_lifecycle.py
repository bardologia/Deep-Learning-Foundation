"""Tests for the cube page's front-end lifecycle, run against the real cube_view.js.

This module runs ``cube_view.js`` under node with no DOM through
``webui/tools/visual/cube_view_checks.js``. The cube view keeps decoded slice images as
ImageBitmaps, whose pixel memory is only returned when they are closed, so evicted,
cleared, released and replaced bitmaps must close while a bitmap a sweep still shows is
kept. The point cloud's section-cut plane must bound itself by the cloud's true extent,
hide the points beyond it, and stop its sweep loop when the view or the tab is left, and the
cloud must drop a points blob that lands after a newer request or another cube. Each cut
panel marks its position in the colour of its own cut axis. The
parameter map's pixel readout must keep the pixel picked last when answers arrive out of
order. Class metric layers must swap the colour bar for a named legend and read out class
names, and the layer histogram must draw one coloured bar per class, shade the colour range,
mark the keep threshold, and never draw a distribution fetched for another layer or cube.
The cut, profile and transect downloads must hand the visible panels and the profiles to the
shared exporter under names carrying the run, stamp and pixel. The Figures view must list a
stamp's saved figures once per cube, drop a listing that lands after the cube changed, and
list the new cube's figures when one loads while the view is open. A pixel locked in the cuts,
the parameter map or the metric map becomes the one shared focus the other views adopt. The
mixture overlay must rebuild the Gaussian mixture from the active slots at the profiled pixel,
scale it with the profiles, and never draw the slots of another pixel or cube. A shift-drag on
the metric map must ask for the statistics of the rectangle it covers, swallow the click that
ends it but no later one, and show only the answer for the latest rectangle of the current cube.
Signed delta layers must use the diverging colour bar, and attaching or detaching a comparison
must refresh the metric view's layers. A transect is a polyline traced click by click, with
undo, an along-line metric strip that never shows an answer for a line that was shortened or
cleared meanwhile, and named lines remembered per cube footprint. The volume view must ask for a
readable WebGL2 context, map its clip axes onto the texture axes, drop a volume streamed for a cube
or comparison that was replaced meanwhile, and render when its view opens or a cube loads into it.
Every axis sweep must ask for a plane or cut the explorer renders. The point cloud strides its rows by the column count the blob header declares and can colour each
point by the seed disagreement of its pixel, greying points without one, while the profile line
names the profiled pixel's seed std. The selective-prediction mask paints only the pixels the
coverage keeps, and its Play sweep runs the coverage down to 5% and back until the view, the tab or
the mask is left. The timeline replays one cut at the shared focus across the run's sibling stamps,
starting on the current one, never draws a frame or listing that lands late, names a stamp without
the chosen source, and stops playing when its view or the tab is left. A #/cube/<state> link reopens
the cube, view, locked pixel, metric layer, value space and colormap, loads another cube first when it
names one, and says so when its view is hidden or unknown or its cube is outside the runs root. The
story recorder keeps named steps with their link state and a snapshot, replays them in order, copies
their links, exports the snapshots through the shared exporter and stops playing with the tab. The
cube view marks its page section is-staged exactly while the stage shows, and the stylesheet hides
the page header on that class rather than on a :has() rule over the stage.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from urllib.parse import quote

import pytest

from routers.cube_routers import CubeRouter
from routers.dispatch     import HttpExchange

from tests.webui.conftest import WEBUI_ROOT, FakeHandler, loaded_cube


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


@NEEDS_JS
def test_params_pixel_readout_ignores_a_late_answer_for_an_earlier_click():
    """When the first pixel's answer lands after the second's, the readout keeps the pixel picked last."""
    observed = _check("pixel")

    assert observed == {"coords": "row = 2 · col = 2 · threshold 0.1", "rendered": [2], "open": True}


@NEEDS_JS
def test_class_layers_swap_the_colour_bar_for_a_named_legend():
    """A class layer hides the range controls and colour bar and lists its classes; a continuous layer restores them."""
    observed = _check("legend")

    assert observed["discrete"]["legend"] and not observed["discrete"]["range"] and not observed["discrete"]["cbar"] and not observed["discrete"]["bounds"]
    assert "#f58518" in observed["discrete"]["html"] and ">suspect</span>" in observed["discrete"]["html"]
    assert observed["continuous"] == {"range": True, "cbar": True, "bounds": True, "legend": False, "html": ""}


@NEEDS_JS
def test_class_layers_read_out_class_names():
    """Hovering a class layer names the class under the cursor, and the locked readout names it beside continuous values."""
    observed = _check("legend")

    assert observed["hover"] == "row = 1 · col = 2 · label suspect = suspect"
    assert "<td>suspect</td>" in observed["readout"] and "<td>0.25</td>" in observed["readout"]


@NEEDS_JS
def test_class_histogram_draws_one_bar_per_class_in_its_legend_colour():
    """A class layer's histogram paints each class count in that class's legend colour, scaled to the largest."""
    classes = _check("histogram")["classes"]

    assert classes["fills"] == ["#4c78a8", "#f58518"]
    assert classes["heights"][0] == pytest.approx(3 * classes["heights"][1])
    assert not classes["note"]


@NEEDS_JS
def test_continuous_histogram_shades_the_range_marks_the_threshold_and_counts_the_tails():
    """A continuous histogram shades the colour range, draws one bar per bin, marks the keep threshold and notes the tails."""
    continuous = _check("histogram")["continuous"]

    assert continuous["fills"] == 1 + 4
    assert continuous["band"] == "rgba(163, 192, 255, 0.1)"
    assert continuous["bars"] == "rgba(163, 192, 255, 0.72)"
    assert continuous["strokes"] == 1
    assert continuous["note"] == "3 px below and 1,200 px above the shown range"


@NEEDS_JS
def test_histogram_never_draws_a_distribution_of_another_layer_or_cube():
    """Switching layers before the new distribution arrives clears the canvas, and an answer for a replaced cube is dropped."""
    observed = _check("histogram")

    assert observed["stale"] == {"fills": 0, "cleared": True}
    assert observed["late"] == {"hist": None, "histKey": None}


@NEEDS_JS
def test_downloads_refuse_until_a_cut_and_profiles_exist():
    """Without a cut position or loaded profiles the PNG and CSV buttons warn instead of saving."""
    assert _check("downloads")["refused"] == ["warn", "warn"]


@NEEDS_JS
def test_cut_download_saves_each_visible_rendered_panel():
    """Only shown panels holding a bitmap are saved, named by run, stamp, padded pixel, axis, source and space."""
    observed = _check("downloads")

    assert observed["tag"] == "group_run_a_stamp_1"
    assert observed["cuts"] == [{"via": "bitmap", "name": "group_run_a_stamp_1_row0003_col0012_col_pred_physical.png", "bitmap": "shown"}]


@NEEDS_JS
def test_profile_csv_lists_every_source_sample_at_the_profiled_pixel():
    """The CSV carries one source, z, value row per sample of every source, named after the profiled pixel."""
    csv = _check("downloads")["csv"]

    assert csv["name"] == "group_run_a_stamp_1_profiles_row0003_col0004.csv"
    assert csv["type"] == "text/csv"
    assert csv["text"].split("\n") == ["source,z,value", "pred,-1,0.25", "pred,0.5,2", "gt,-1,0", "gt,0.5,1"]


@NEEDS_JS
def test_transect_download_saves_each_drawn_panel_named_by_its_end_points():
    """The transect PNG button saves every drawn panel under the line's end points and skips undrawn ones."""
    assert _check("downloads")["line"] == [{"via": "canvas", "name": "group_run_a_stamp_1_transect_row1_col2_to_row5_col0_pred_physical.png"}]


@NEEDS_JS
def test_bent_transect_download_names_its_interior_vertices():
    """A polyline's PNG names carry the same _via suffix as its saved figure folder."""
    assert _check("downloads")["bent"] == ["group_run_a_stamp_1_transect_row1_col2_to_row5_col0_via1_pred_physical.png"]


@NEEDS_JS
def test_metric_overlay_download_saves_the_shown_overlay_and_reports_a_failed_fetch():
    """The metric PNG button saves the overlay the map shows under the layer key and toasts an error when it cannot fetch it."""
    overlay = _check("downloads")["overlay"]

    assert overlay["saved"] == [{"via": "download", "name": "group_run_a_stamp_1_metric_failure_mode.png", "source": "png-bytes"}]
    assert overlay["toasts"] == ["error"]


def test_download_buttons_are_on_the_page_and_wired_to_the_cube_view():
    """The cuts bar carries PNG and CSV buttons and the transect bar a PNG button, and app.js hands each to the cube view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    for element in ("cube-jump-download", "cube-jump-csv", "cube-transect-download", "cube-metric-download"):
        assert html.count(f'id="{element}"') == 1, element

    for ref, element in (("jumpDownload", "cube-jump-download"), ("jumpCsv", "cube-jump-csv"), ("download", "cube-transect-download"), ("download", "cube-metric-download")):
        assert re.search(rf'{ref}\s*: document\.getElementById\("{element}"\)', app), element


@NEEDS_JS
def test_focus_point_clamps_into_the_cube_and_centres_on_the_pixel():
    """The shared focus is clamped into the cube and placed at the centre of its pixel."""
    assert _check("focus")["clamped"] == {"row": 3, "col": 0, "fx": pytest.approx(0.5 / 6), "fy": pytest.approx(3.5 / 4)}


@NEEDS_JS
def test_a_pixel_picked_on_the_parameter_map_becomes_the_shared_focus():
    """Clicking the parameter map sets the shared focus at that pixel and marks it as already adopted there."""
    assert _check("focus")["paramPick"] == {"focus": {"row": 1, "col": 3}, "seq": 2, "adopted": 2, "fetched": [[1, 3]]}


@NEEDS_JS
def test_the_metric_view_adopts_the_shared_focus_once_and_reads_every_layer_there():
    """Opening the metric view locks its crosshair on the shared focus and reads it out, and a second render does not read again."""
    adopted = _check("focus")["adopted"]

    assert adopted["picked"] == [1, 3]
    assert adopted["coords"] == "locked at row = 1 · col = 3 · click again to unlock"
    assert adopted["cross"] == [f"{3.5 / 6 * 100}%", "37.5%"]
    assert adopted["open"] and adopted["readouts"] == [[1, 3]]


@NEEDS_JS
def test_the_parameter_view_adopts_a_pixel_locked_on_the_metric_map():
    """A pixel locked on the metric map becomes the focus, and the parameter view then reads its slots at that pixel."""
    observed = _check("focus")

    assert observed["metricPick"] == {"focus": {"row": 3, "col": 5}, "adopted": True, "readouts": [[3, 5]]}
    assert observed["paramAdopt"] == {"picked": [3, 5], "fetched": [[3, 5]], "cross": True}


@NEEDS_JS
def test_returning_to_the_cuts_locks_them_on_a_new_focus_once():
    """Back on the cuts a focus set elsewhere moves the cut and locks the slices there, and it is not entered twice."""
    observed = _check("focus")

    assert observed["entered"] == {"followed": [[3, 5]], "mode": "slices", "locked": {"row": 3, "col": 5}, "profiled": [[3, 5]], "adopted": True}
    assert observed["kept"] == {"followed": [], "fetched": [[3, 5]]}


@NEEDS_JS
def test_opening_the_cube_at_a_pixel_moves_the_focus_without_locking_the_previous_one():
    """Opening the cube at a pixel from another tab makes that pixel the focus instead of locking the cuts on the previous focus."""
    observed = _check("focus")

    assert observed["opened"] == {"focus": {"row": 3, "col": 2}, "mode": "map", "followed": [[3, 2]], "view": "explorer"}
    assert observed["reopened"] == {"picked": [3, 2], "readouts": [[3, 2]]}


@NEEDS_JS
def test_params_here_button_shows_only_for_cubes_with_parameters():
    """The cuts bar offers the Params here jump only when the loaded cube carries Gaussian parameters."""
    assert _check("focus")["button"] == {"withParams": True, "withoutParams": False}


@NEEDS_JS
def test_shift_drag_paints_the_rectangle_and_asks_for_the_rows_and_cols_it_covers():
    """A shift-drag paints the dashed rectangle and requests the region spanning the pixels under it."""
    observed = _check("region")

    assert observed["painted"]["hidden"] is False and observed["painted"]["prevented"] == 1
    assert observed["painted"]["style"]["top"] == "12.5%" and observed["painted"]["style"]["height"] == "75%"
    assert observed["request"] == {"bounds": [0, 3, 1, 4], "id": "cube"}
    assert observed["clipped"] == [0, 3, 0, 5]


@NEEDS_JS
def test_region_answer_fills_the_table_and_draws_one_mean_profile_per_source():
    """The region table lists each layer's region and scene means with a dash for a missing one, and each source draws its mean profile."""
    shown = _check("region")["shown"]

    assert shown["hint"] == "rows 0–3 · cols 1–4 · 16 pixels · shift-drag again to replace"
    assert "<tr><td>R2</td><td>0.5</td><td>0.25</td></tr>" in shown["table"]
    assert "<tr><td>label R2</td><td>–</td><td>0.75</td></tr>" in shown["table"]
    assert shown["wrap"] and shown["strokes"] == ["#a3c0ff", "#34d399"]


@NEEDS_JS
def test_the_click_ending_a_drag_and_a_shift_click_lock_no_pixel():
    """The click that ends a drag is swallowed, and a shift-click too short to span a region neither asks nor locks a pixel."""
    observed = _check("region")

    assert observed["swallowed"] == {"picked": None, "suppress": False, "asked": 0}
    assert observed["tiny"] == {"rect": True, "asked": 0, "picked": None}


@NEEDS_JS
def test_a_drag_ending_off_the_map_does_not_swallow_the_next_click():
    """After a drag released outside the map, whose click never reaches it, the next plain click still locks its pixel."""
    assert _check("region")["offMap"] == {"picked": [2, 3]}


@NEEDS_JS
def test_region_answers_for_an_earlier_rectangle_or_another_cube_are_dropped():
    """Only the latest rectangle's statistics are shown, an answer landing after the cube changed is dropped, and a refusal is toasted."""
    observed = _check("region")

    assert observed["late"] == "rows 2–3 · cols 2–3 · 4 pixels · shift-drag again to replace"
    assert observed["replaced"] == observed["late"]
    assert observed["toasts"] == [{"text": "cube not loaded", "kind": "error"}]


@NEEDS_JS
def test_configuring_the_metric_view_clears_the_region():
    """Loading a cube hides the rectangle and the profile chart, empties the table and restores the hint."""
    assert _check("region")["reset"] == {"rect": True, "table": "", "wrap": True, "hint": "Shift-drag a rectangle on the map to compare it against the scene."}


def test_region_panel_is_on_the_page_and_wired_to_the_metric_view():
    """The metric map carries the rectangle and the side panel the region hint, table and profile chart, all handed to the metric view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    for ref, element in (("rect", "cube-metric-rect"), ("regionHint", "cube-region-hint"), ("regionTab", "cube-region-table"), ("regionWrap", "cube-region-prof-wrap"), ("regionProf", "cube-region-prof")):
        assert html.count(f'id="{element}"') == 1, element
        assert re.search(rf'{ref}\s*: document\.getElementById\("{element}"\)', app), ref


@NEEDS_JS
def test_delta_layers_use_the_diverging_colour_bar():
    """A signed delta layer draws the coolwarm colour bar and a plain continuous layer keeps viridis."""
    observed = _check("delta")

    assert observed["diverging"] == "/api/cubes/cbar?cmap=coolwarm"
    assert observed["plain"] == "/api/cubes/cbar?cmap=viridis"


@NEEDS_JS
def test_attaching_a_comparison_refreshes_the_metric_layers():
    """Attaching reconfigures the open metric view with the delta layers and shows its button; losing every layer leaves the view."""
    observed = _check("delta")

    assert observed["attached"] == {"calls": [["configure", "pixel_mse", "delta_pixel_mse"], ["render"]], "button": True}
    assert observed["emptied"] == {"calls": [["configure"], ["view", "explorer"]], "button": False}


def _mixture(z: float, slots: list) -> float:
    """Returns the Gaussian mixture of the given (amp, mu, sigma) slots at one z."""
    return sum(amp * math.exp(-((z - mu) ** 2) / (2 * sigma * sigma)) for amp, mu, sigma in slots)


def _trapezoid(xs: list, ys: list) -> float:
    """Returns the trapezoid area under ys over xs."""
    return sum(0.5 * (ys[i] + ys[i - 1]) * abs(xs[i] - xs[i - 1]) for i in range(1, len(xs)))


@NEEDS_JS
def test_mixture_toggle_fetches_the_slots_at_the_profiled_pixel_before_drawing():
    """Ticking the mixture box asks for the Gaussian slots at the profiled pixel and draws nothing until they arrive."""
    assert _check("mixture")["requested"] == {"calls": ["/api/cubes/params_at 1,2"], "drawn": {"dashed": 0, "stems": []}}


@NEEDS_JS
def test_mixture_draws_one_dashed_curve_per_panel_and_one_stem_per_active_slot():
    """Each parameter source's panel gets the dashed mixture and a coloured stem for every active slot only."""
    assert _check("mixture")["drawn"] == {"dashed": 2, "stems": ["#e45756", "#4c78a8", "#e45756"]}


@NEEDS_JS
def test_mixture_curve_sums_the_active_gaussians_and_stems_sit_at_mu_with_the_amplitude():
    """The dashed curve is the sum of the active slots' Gaussians on the profile's z axis, and each stem reaches its amplitude at its mu."""
    raw    = _check("mixture")["raw"]
    active = [(2.0, 1.0, 0.5), (1.0, 3.0, 0.25)]

    assert [point[1] for point in raw["curve"]] == [0, 1, 2, 3, 4]
    assert [point[0] for point in raw["curve"]] == pytest.approx([_mixture(z, active) for z in range(5)])
    assert raw["stems"] == [{"color": "#e45756", "points": [[0, 1], [2, 1]]}, {"color": "#4c78a8", "points": [[0, 3], [1, 3]]}]


@NEEDS_JS
def test_unit_area_mode_scales_the_mixture_and_its_stems_to_unit_area():
    """In unit-area mode the dashed mixture integrates to one over z and the stems shrink by the same factor."""
    observed = _check("mixture")
    raw      = [point[0] for point in observed["raw"]["curve"]]
    unit     = observed["unit"]

    factor = 1.0 / _trapezoid(list(range(5)), raw)

    assert _trapezoid(list(range(5)), [point[0] for point in unit["curve"]]) == pytest.approx(1.0)
    assert [stem["points"][1][0] for stem in unit["stems"]] == pytest.approx([2.0 * factor, 1.0 * factor])


@NEEDS_JS
def test_unticking_the_mixture_redraws_without_it_and_ticking_again_reuses_the_slots():
    """Unticking redraws the profiles without the overlay, and ticking again at the same pixel draws it without asking again."""
    observed = _check("mixture")

    assert observed["off"] == {"calls": [], "drawn": {"dashed": 0, "stems": []}}
    assert observed["cached"]["calls"] == [] and observed["cached"]["drawn"]["dashed"] == 2


@NEEDS_JS
def test_profiles_at_a_new_pixel_fetch_its_slots_while_the_mixture_is_on():
    """With the overlay on, profiles landing for another pixel ask for that pixel's slots."""
    assert _check("mixture")["followed"] == ["/api/cubes/profiles 3,4", "/api/cubes/params_at 3,4"]


@NEEDS_JS
def test_mixture_drops_slots_answered_for_an_earlier_pixel_or_another_cube():
    """Slots arriving after the profiles moved on, or after the cube changed, are dropped instead of replacing the current ones."""
    observed = _check("mixture")

    assert observed["late"] == {"params": [2, 2], "matches": True}
    assert observed["replaced"] == {"params": None}


@NEEDS_JS
def test_mixture_box_shows_only_for_cubes_with_parameters():
    """The profiles header offers the mixture box only when the loaded cube carries Gaussian parameters."""
    assert _check("mixture")["toggle"] == {"withParams": True, "withoutParams": False}


def test_mixture_box_is_on_the_page_and_wired_to_the_cube_view():
    """The profiles header carries a hidden mixture box and app.js hands it and its wrapper to the cube view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    assert html.count('id="cube-prof-mix-wrap" hidden') == 1
    assert html.count('<input type="checkbox" id="cube-prof-mix" />') == 1
    for ref, element in (("profMixWrap", "cube-prof-mix-wrap"), ("profMix", "cube-prof-mix")):
        assert re.search(rf'{ref}\s*: document\.getElementById\("{element}"\)', app), ref


def test_params_here_button_is_on_the_page_and_wired_to_the_cube_view():
    """The cuts bar carries a hidden Params here button and app.js hands it to the cube view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    assert html.count('id="cube-jump-params" hidden') == 1
    assert re.search(r'jumpParams\s*: document\.getElementById\("cube-jump-params"\)', app)


@NEEDS_JS
def test_figures_view_lists_each_folder_with_linked_thumbnails():
    """Each folder becomes a group, the first one open, with thumbnails linking to the path-guarded figure route."""
    observed = _check("figures")
    listed   = observed["listed"]

    assert observed["loading"] == "<orb>Loading figures…</orb>"
    assert observed["calls"][0] == "/api/cubes/figures?id=stamp%2Fa"
    assert "3 saved figures in 2 folders" in listed
    assert listed.count("<details") == 2 and listed.count(" open>") == 1
    assert 'href="/api/cubes/figure?id=stamp%2Fa&rel=figures%2Fpixel_maps%2Fmse.png"' in listed
    assert '<summary>figures/pixel_maps <em>2</em></summary>' in listed


@NEEDS_JS
def test_figures_view_lists_once_per_cube_and_drops_a_listing_for_a_replaced_cube():
    """Rendering again for the same cube reuses the listing, and an answer for a cube that was replaced meanwhile is dropped."""
    observed = _check("figures")

    assert observed["cached"] == 1
    assert observed["dropped"] == {"loadedFor": None, "html": "<orb>Loading figures…</orb>"}


@NEEDS_JS
def test_figures_view_reports_an_empty_stamp_and_a_failed_listing():
    """A stamp without figure trees and a refused listing each show a hint instead of an empty panel."""
    observed = _check("figures")

    assert observed["none"] == '<p class="cube-hint">This stamp holds no saved figures.</p>'
    assert observed["error"] == '<p class="cube-hint">cube not loaded</p>'


@NEEDS_JS
def test_loading_a_cube_while_the_figures_view_is_open_lists_its_figures():
    """A cube that finishes loading while the Figures view is shown lists its figures instead of staying on the loading panel."""
    reloaded = _check("figures")["reloaded"]

    assert reloaded == {"figures": ["configure", "render"], "explorer": ["configure"]}


@NEEDS_JS
def test_transect_clicks_trace_a_polyline_and_skip_a_repeated_pixel():
    """Each click adds a vertex, a click on the last vertex's pixel is ignored, and every line of two or more vertices asks for its strip."""
    observed = _check("transect")

    assert observed["configured"] == {"layers": ["pixel_mse", "pixel_r2"], "saved": [""], "undo": True, "strip": True}
    assert observed["single"] == {"points": [[3, 2]], "asked": 0, "undo": False, "print": True, "at": "start row = 3 · col = 2 · click the next point"}
    assert observed["traced"] == {"pts": ["3,2;3,5", "3,2;3,5;7,5"], "print": False, "at": "transect row 3,2 to row 7,5 · 3 points · click to extend", "segments": 2}


@NEEDS_JS
def test_transect_strip_draws_only_the_latest_line_and_skips_missing_samples():
    """A strip answer overtaken by a longer line is dropped, and the latest one draws its finite samples and range."""
    observed = _check("transect")

    assert observed["overtaken"] == {"shown": False, "data": None}
    assert observed["drawn"] == {"shown": True, "value": "1 … 3 over 3 samples", "strokes": 1, "segments": 2}
    assert observed["picked"] == "-3 … -1 over 3 samples"


@NEEDS_JS
def test_transect_undo_shortens_the_line_and_hides_the_strip_below_two_points():
    """Undo refetches the shorter line and, at one vertex left, hides the strip and drops its data."""
    assert _check("transect")["afterUndo"] == {"shortened": ["3,2;3,5"], "points": 1, "strip": True, "data": None, "asked": 0}


@NEEDS_JS
def test_transect_strip_answers_for_an_undone_or_cleared_line_are_dropped():
    """A strip answer landing after undo took the line below two points, or after Clear line, stays hidden."""
    observed = _check("transect")

    assert observed["lateAfterUndo"] == {"strip": True, "data": None}
    assert observed["lateAfterClear"] == {"strip": True, "data": None, "points": 0}


@NEEDS_JS
def test_attaching_a_comparison_keeps_the_strip_layer_and_refetches_the_line():
    """New delta layers join the strip select without losing the picked layer, and the traced line asks for its strip again."""
    assert _check("transect")["attached"] == {"layers": ["pixel_mse", "pixel_r2", "delta_pixel_mse"], "kept": "pixel_r2", "refetched": ["3,2;3,5;7,5"]}


@NEEDS_JS
def test_saved_transects_are_kept_per_footprint_and_replay_onto_pixel_centres():
    """Saving stores the vertices under the cube footprint, unnamed lines get a numbered name, and replaying redraws the line at pixel centres."""
    observed = _check("transect")
    line     = [[3, 2], [3, 5], [7, 5]]

    assert observed["unnamed"] == [{"name": "line 1", "points": line}]
    assert observed["named"] == {"stored": [{"name": "line 1", "points": line}, {"name": "ridge", "points": line}], "options": ["", "line 1", "ridge"], "selected": "ridge", "remove": False, "name": ""}
    assert observed["replayed"]["points"] == [[3, 2, pytest.approx(2.5 / 6), pytest.approx(3.5 / 8)], [3, 5, pytest.approx(5.5 / 6), pytest.approx(3.5 / 8)], [7, 5, pytest.approx(5.5 / 6), pytest.approx(7.5 / 8)]]
    assert observed["replayed"]["asked"] == ["3,2;3,5;7,5"]
    assert observed["deleted"] == {"stored": ["line 1"], "options": ["", "line 1"]}


@NEEDS_JS
def test_transects_of_another_footprint_or_without_metric_maps_stay_apart():
    """Another footprint lists none of the saved lines, a cube without metric maps asks for no strip, and saving a line of fewer than two points warns."""
    observed = _check("transect")

    assert observed["otherFootprint"] == {"options": [""], "layers": []}
    assert observed["noLayers"] == {"asked": 0, "strip": True}
    assert observed["refusedSave"] == ["Trace a line with at least two points first."]


def test_transect_controls_are_on_the_page_once_and_wired_to_the_cube_view():
    """Every transect control id appears once in the page and app.js hands each to the transect view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    refs = {"at": "at", "undo": "undo", "clear": "clear", "print": "print", "download": "download", "name": "name", "save": "save", "saved": "saved", "remove": "delete", "stripWrap": "strip-wrap", "stripLayer": "strip-layer", "stripVal": "strip-val", "strip": "strip", "map": "map", "overlay": "overlay", "grid": "grid"}
    for ref, suffix in refs.items():
        assert html.count(f'id="cube-transect-{suffix}"') == 1, suffix
        assert re.search(rf'{ref}\s*: document\.getElementById\("cube-transect-{suffix}"\)', app), ref


@NEEDS_JS
def test_the_volume_view_keeps_its_drawing_buffer():
    """The volume asks for a WebGL2 context that preserves its drawing buffer, so the figure saver can capture it."""
    assert _check("volume")["context"] == {"kind": "webgl2", "opts": {"alpha": True, "antialias": False, "preserveDrawingBuffer": True}}


@NEEDS_JS
def test_volume_falls_back_to_a_loaded_source_and_streams_it_once():
    """A source the cube lacks falls back to the first loaded one, the volume is fetched once at the capped size and uploaded column-major."""
    observed = _check("volume")

    assert observed["configured"] == {"source": "pred", "disabled": [False, True, True, False, True], "active": [True, False, False, False, False]}
    assert observed["request"] == {"id": "cube/a", "source": "pred", "max": "192", "at": "<orb>Streaming volume…</orb>"}
    assert observed["drawn"]["dims"] == [2, 3, 4] and observed["drawn"]["key"] == "cube/a|pred" and observed["drawn"]["voxels"] == 24
    assert observed["drawn"]["upload"] == [[4, 3, 2]] and observed["drawn"]["draws"] == 1
    assert observed["drawn"]["at"].startswith("3 × 4 × 2 voxels (row × col × z) · intensity 0.5 … 2")
    assert observed["cached"] == {"asked": 0, "draws": 1}


@NEEDS_JS
def test_volume_streamed_for_a_replaced_cube_is_dropped():
    """A volume landing after another cube was configured is dropped instead of being shown under the new cube."""
    assert _check("volume")["late"] == {"volume": None, "key": None}


@NEEDS_JS
def test_volume_reports_a_failed_fetch_and_a_missing_webgl2():
    """A refused volume request names the failure, and a browser without WebGL2 disables the view without fetching."""
    observed = _check("volume")

    assert observed["failed"] == {"volume": None, "at": "Volume fetch failed."}
    assert observed["unavailable"] == {"available": False, "at": "WebGL2 is not available in this browser; the volume view is disabled.", "asked": 0}


@NEEDS_JS
def test_volume_clip_axes_cut_the_matching_texture_axis():
    """Row, col and z clip the texture's second, first and third axis; off leaves the box whole."""
    assert _check("volume")["clips"] == {"off": [1, 1, 1], "row": [1, 0.25, 1], "col": [0.25, 1, 1], "z": [1, 1, 0.25]}


@NEEDS_JS
def test_volume_refreshes_comparison_sources_without_moving_the_camera():
    """Attaching a comparison drops the cached volume and enables pred B and A − B while the camera stays where it was."""
    assert _check("volume")["refreshed"] == {"volume": None, "yaw": 2, "source": "gt", "disabled": [False, False, False, False, True]}


@NEEDS_JS
def test_volume_draws_only_while_its_view_is_shown():
    """A volume streamed while another view is open is kept but not drawn."""
    assert _check("volume")["hiddenDraws"] == 0


@NEEDS_JS
def test_cube_view_renders_the_volume_on_load_attach_and_view_switch():
    """Loading a cube or attaching a comparison with the volume view open re-renders it, and a browser without WebGL2 hides the mode and leaves the view."""
    views = _check("volume")["views"]

    assert views["loaded"] == {"calls": ["configure", "render"], "hidden": False}
    assert views["blind"] == {"calls": ["configure"], "hidden": True, "view": "explorer"}
    assert views["attached"] == ["refresh", "render"]
    assert views["picked"] == {"calls": ["render"], "view": "volume"}


def test_volume_controls_are_on_the_page_once_and_wired_to_the_cube_view():
    """The Volume mode button and every volume control exist once, carry no foreign source, and app.js hands each to the volume view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    assert html.count('class="cube-mode" data-view="volume"') == 1
    assert re.findall(r'data-source="(\w+)"', html.split('id="cube-volume-source"')[1].split("</div>")[0]) == ["pred", "predb", "diff", "gt", "full"]
    assert re.findall(r'<option value="(\w+)"', html.split('id="cube-volume-clip-axis"')[1].split("</select>")[0]) == ["off", "row", "col", "z"]

    refs = {"source": "source", "lo": "lo", "hi": "hi", "opacity": "opacity", "steps": "steps", "clipAxis": "clip-axis", "clipPos": "clip-pos", "at": "at", "canvas": "canvas"}
    for ref, suffix in refs.items():
        assert html.count(f'id="cube-volume-{suffix}"') == 1, suffix
        assert re.search(rf'{ref}\s*: document\.getElementById\("cube-volume-{suffix}"\)', app), ref


@NEEDS_JS
def test_finite_range_skips_missing_seed_disagreement():
    """The cloud's robust range helper ignores NaN entries and returns None for a column without a finite value."""
    assert _check("uncertainty")["ranges"] == {"mixed": [0.25, 0.5], "empty": None}


@NEEDS_JS
def test_uncertainty_colour_needs_the_seed_std_layer():
    """Without a seed_std_profile layer the uncertainty button is disabled and the colour falls back to z; with it the mode stays."""
    observed = _check("uncertainty")

    assert observed["absent"] == {"colorBy": "mu", "disabled": True}
    assert observed["present"] == {"colorBy": "std", "disabled": False}


@NEEDS_JS
def test_uncertainty_colour_paints_points_by_their_seed_std_and_greys_missing_ones():
    """Colouring by uncertainty maps each finite seed std over the cloud's range and paints a NaN one grey, unlike colouring by z."""
    observed = _check("uncertainty")

    assert observed["byStd"]["ts"] == [0, 1]
    assert observed["byStd"]["greys"] == 4
    assert observed["byStd"]["text"].startswith("3 of 3 components")
    assert observed["byZ"] == {"ts": [0, 0.5, 1], "greys": 0}


@NEEDS_JS
def test_cloud_reads_its_row_width_from_the_blob_header():
    """A fetched blob is strided by the column count its header declares, and the seed std range comes from the fifth column."""
    assert _check("uncertainty")["fetched"] == {"width": 5, "total": 7, "muBounds": [3, 8], "stdRange": [5, 10]}


@NEEDS_JS
def test_cloud_drops_a_blob_answered_for_an_earlier_request_or_cube():
    """A points blob that lands after a newer request, or after another cube configured the cloud, is not drawn."""
    observed = _check("cloudRace")

    assert observed["raced"] == {"total": 2, "mu": 2}
    assert observed["reconfigured"] == {"points": None, "total": 2}


@NEEDS_JS
def test_cut_markers_take_the_colour_of_their_cut_axis():
    """The col and row cut panels draw their position marker in the --cut-col and --cut-row colours over a white halo."""
    markers = _check("cutColors")["markers"]

    assert markers["col"] == ["rgba(255, 255, 255, 0.9)", "--cut-col"]
    assert markers["row"] == ["rgba(255, 255, 255, 0.9)", "--cut-row"]


@NEEDS_JS
def test_coverage_play_sweeps_down_to_five_percent_and_back_with_the_mask_on():
    """Play ticks the mask on and steps the coverage by 5 down to 5 percent before turning back."""
    observed = _check("coverage")

    assert observed["started"] == {"mask": True, "label": "Pause", "running": 1}
    assert observed["swept"] == {"values": ["10", "5", "10", "15"], "dir": 1}


@NEEDS_JS
def test_coverage_mask_keeps_the_most_confident_side_of_the_threshold():
    """With the mask on, the overlay keeps pixels below the threshold of an error-like layer and above that of an R2-like one."""
    observed = _check("coverage")

    assert observed["lowWindow"] == {"keep_min": "-inf", "keep_max": 0.2}
    assert observed["highWindow"] == {"keep_min": 0.7, "keep_max": "inf"}


@NEEDS_JS
def test_coverage_play_stops_when_the_view_or_the_tab_is_left_or_the_mask_cleared():
    """Leaving the metric view or the Cube tab, unticking the mask or loading another cube ends the sweep and drops the mask."""
    observed = _check("coverage")

    assert observed["leftView"] == {"running": 0, "label": "Play"}
    assert observed["left"] == {"running": 0, "label": "Play"}
    assert observed["unmasked"] == {"running": 0, "window": None, "keep": {"keep_min": "-inf", "keep_max": "inf"}}
    assert observed["unmaskedAnswer"] is None
    assert observed["reconfigured"] == {"running": 0, "mask": False, "window": None}


@NEEDS_JS
def test_profile_line_shows_the_seed_std_of_the_profiled_pixel_only():
    """The profile line gains the pixel's seed std, a dash when it is missing, and ignores an answer for a pixel left meanwhile."""
    observed = _check("profileStd")

    assert observed["calls"][0] == "/api/cubes/metric_at?id=cube&key=seed_std_profile&row=1&col=2"
    assert observed["value"] == "profiles at row = 1 · col = 2 · seed std 0.13"
    assert observed["moved"] == "profiles at row = 3 · col = 3"
    assert observed["missing"] == "profiles at row = 3 · col = 3 · seed std –"
    assert observed["layers"] == {"present": True, "absent": False, "unloaded": False}


def test_uncertainty_and_coverage_controls_are_on_the_page_and_wired():
    """The cloud carries one uncertainty colour button, and the selective block a mask box and Play button that app.js hands to the metric view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    cloud_colours = html.split('id="cube-cloud-color"')[1].split("</div>")[0]
    assert re.findall(r'data-color="(\w+)"', cloud_colours) == ["mu", "amp", "std"]

    assert html.count('<input type="checkbox" id="cube-selective-mask" />') == 1
    assert html.count('id="cube-selective-play"') == 1
    assert re.search(r'selMask\s*: document\.getElementById\("cube-selective-mask"\)', app)
    assert re.search(r'selPlay\s*: document\.getElementById\("cube-selective-play"\)', app)


@NEEDS_JS
def test_every_axis_sweep_asks_for_a_frame_the_explorer_renders(tmp_path):
    """The z, row and col sweeps request planes and cuts the cube router answers with a PNG; the col sweep used to ask for an axis the explorer refuses."""
    explorer, cube_id = loaded_cube(tmp_path)
    router            = CubeRouter(explorer)
    urls              = _check("sweepUrls")

    assert set(urls) == {"z", "row", "col"}
    for axis, url in urls.items():
        handler = FakeHandler("GET", url.replace("id=stamp%2Fa", f"id={quote(cube_id, safe='')}"))

        assert router.handle(HttpExchange.of(handler)), axis
        assert handler.status == 200, axis


@NEEDS_JS
def test_timeline_opens_on_the_current_stamp_and_replays_the_focused_col_cut():
    """The first render lists the run's stamps once, starts on the current one, labels its curve MSE and draws its col cut while prefetching the next stamp."""
    first = _check("timeline")["first"]

    assert first["idx"] == 1 and first["max"] == "2" and first["pos"] == "1"
    assert first["at"] == "stamp_2 · 2 / 3 · col = 3"
    assert first["label"] == "stamp_2 · curve MSE 0.125"
    assert [url["id"] for url in first["urls"]] == ["/runs/a/inference/stamp_3", "/runs/a/inference/stamp_2"]
    assert first["urls"][1] == {"id": "/runs/a/inference/stamp_2", "source": "pred", "axis": "col", "row": "4", "col": "3", "space": "physical", "cmap": "jet"}
    assert len(first["drawn"]) == 1 and first["canvas"] == [6, 5]
    assert first["disabled"] == [False, False, True]


@NEEDS_JS
def test_timeline_reuses_its_listing_and_refuses_a_source_the_cube_lacks():
    """A second render reuses the stamp list, and a source button the cube lacks changes nothing."""
    observed = _check("timeline")

    assert observed["cached"] == {"asked": 0, "urls": 2, "drawn": 1}
    assert observed["refused"] == {"source": "pred", "urls": 0}


@NEEDS_JS
def test_timeline_follows_the_shared_focus_axis_and_scrubber():
    """The row cut is taken at the shared focus, and scrubbing stops playback and shows the picked stamp."""
    observed = _check("timeline")

    assert observed["row"]["at"] == "stamp_2 · 2 / 3 · row = 2"
    assert observed["row"]["url"]["axis"] == "row" and observed["row"]["url"]["row"] == "2" and observed["row"]["url"]["col"] == "3"
    assert observed["scrubbed"] == {"idx": 0, "playing": False, "label": "stamp_1", "prefetched": ["/runs/a/inference/stamp_2", "/runs/a/inference/stamp_1"], "drawn": 1}


@NEEDS_JS
def test_timeline_names_a_missing_cut_and_draws_only_the_latest_frame():
    """A stamp without the chosen source clears the canvas and says so, and a frame that lands after a newer one is dropped."""
    observed = _check("timeline")

    assert observed["missing"] == {"at": "stamp_1 · 1 / 3 · row = 2 · no gt cut for this stamp", "drawn": ["cleared"]}
    assert observed["stale"] == ["new"]


@NEEDS_JS
def test_timeline_play_advances_and_stops_when_the_view_is_left():
    """Play steps through the stamps and stops, relabelling the button, once the view changes."""
    observed = _check("timeline")

    assert observed["started"] == {"playing": True, "label": "Pause"}
    assert observed["advanced"] is True
    assert observed["stopped"] == {"playing": False, "label": "Play"}


@NEEDS_JS
def test_timeline_drops_a_late_listing_and_reports_failures_and_single_stamps():
    """A listing for a cube replaced meanwhile is dropped, an error is shown, and a lone stamp says so and does not play."""
    observed = _check("timeline")

    assert observed["dropped"] == {"stamps": 0, "loadedFor": None}
    assert observed["failed"] == "unknown cube id: stamp/c"
    assert observed["alone"]["at"].endswith("this run holds a single inference stamp; run inference again to grow the timeline")
    assert observed["alone"]["playing"] is False


@NEEDS_JS
def test_cube_view_renders_the_timeline_on_load_switch_and_recolour_and_leaves_it():
    """Loading a cube with the timeline open configures and renders it, switching views leaves or renders it, a colormap change redraws it, and leaving the tab stops it."""
    assert _check("timelineViews") == {"loaded": ["configure", "render"], "left": ["leave"], "opened": ["render"], "recoloured": ["render"], "tab": ["leave"]}


def test_timeline_controls_are_on_the_page_once_and_wired_to_the_cube_view():
    """The Timeline mode button and every timeline control exist once, offer the template sources and axes, and app.js hands each to the timeline."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    assert html.count('class="cube-mode" data-view="timeline"') == 1
    assert re.findall(r'data-source="(\w+)"', html.split('id="cube-timeline-source"')[1].split("</div>")[0]) == ["pred", "gt", "full"]
    assert re.findall(r'data-axis="(\w+)"', html.split('id="cube-timeline-axis"')[1].split("</div>")[0]) == ["col", "row"]

    for ref in ("source", "axis", "play", "pos", "at", "canvas", "label"):
        assert html.count(f'id="cube-timeline-{ref}"') == 1, ref
        assert re.search(rf'{ref}\s*: document\.getElementById\("cube-timeline-{ref}"\)', app), ref


@NEEDS_JS
def test_state_param_round_trips_the_cube_view_pixel_layer_space_and_colormap():
    """The link state names the cube, view, value space, colormap, locked pixel (else the focus) and metric layer, and parses back to the same state."""
    observed = _check("deepLink")

    assert "row%3D2%26col%3D3" in observed["plain"]
    assert observed["query"] == {"id": "/runs/a/inference/stamp_1", "view": "metrics", "space": "physical", "cmap": "jet", "row": "5", "col": "1", "layer": "pixel_mse"}
    assert observed["parsed"] == {"id": "/runs/a/inference/stamp_1", "view": "metrics", "row": 5, "col": 1, "layer": "pixel_mse", "space": "physical", "cmap": "jet"}


@NEEDS_JS
def test_state_param_refuses_links_without_a_cube_and_says_so_for_unreadable_ones():
    """A link without a cube id or param opens nothing, and one that cannot be decoded raises a warning instead of failing the tab."""
    assert _check("deepLink")["refused"] == {"noId": None, "empty": None, "broken": None, "toasts": ["The cube link could not be read."]}


@NEEDS_JS
def test_a_link_to_the_loaded_cube_restores_its_settings_pixel_layer_and_view():
    """Applying a link sets a known colormap and space, locks the pixel clamped into the cube, picks the layer and opens the view."""
    applied = _check("deepLink")["applied"]

    assert applied["calls"] == ["cmap viridis", "space normalized", "follow 7,0,0.08333333333333333,0.9375", "lock 7,0", "layer pixel_r2", "view metrics"]
    assert applied["toasts"] == [] and applied["pending"] is None


@NEEDS_JS
def test_a_link_naming_a_hidden_or_unknown_view_opens_without_it_and_says_so():
    """An unknown colormap or layer is skipped, and a view whose button is hidden or that does not exist leaves the current view with a warning."""
    observed = _check("deepLink")

    assert observed["hidden"] == {"calls": [], "toasts": ["This cube has no Parameters view; the link opened without it."]}
    assert observed["unknown"] == {"calls": [], "toasts": ["The link names an unknown view 'sideways'; it opened without it."]}


@NEEDS_JS
def test_a_link_to_another_cube_loads_it_once_the_current_load_ends_and_applies_after_it():
    """A link to another listed cube waits for a running load, then selects it, and its state applies once that load finishes; a failed load drops it."""
    observed = _check("deepLink")

    assert observed["waiting"] == {"calls": [], "toasts": [], "pending": True}
    assert observed["selected"] == {"calls": ["select /runs/a/inference/stamp_2"], "toasts": [], "pending": True}
    assert observed["failed"] == {"calls": [], "toasts": [], "pending": None}
    assert observed["polled"] == {"calls": ["display", "follow 3,4,0.75,0.4375", "lock 3,4"], "toasts": [], "polling": False, "pending": None}


@NEEDS_JS
def test_a_link_outside_the_runs_root_is_refused_and_reentering_the_tab_applies_a_link():
    """A cube missing from the listing is refused with a warning, and entering the tab again with a link refreshes the strip and applies it."""
    observed = _check("deepLink")

    assert observed["foreign"] == {"calls": [], "toasts": ["The linked cube is not in the current runs root."], "pending": None}
    assert observed["reentered"] == {"calls": ["strip", "view explorer", "follow 1,1,0.25,0.1875", "lock 1,1"], "toasts": []}


@NEEDS_JS
def test_story_records_named_steps_with_snapshots_and_lists_them():
    """Adding steps stores each link state under a trimmed or numbered name, keeps a snapshot when the view shows one, and lists every step with its buttons."""
    observed = _check("story")

    assert observed["emptyPlay"] == {"toasts": ["The story has no steps yet."], "timers": 0}
    assert observed["opened"] == {"hidden": False, "hint": "No steps yet; set up a view and press Add step."}

    added = observed["added"]
    assert [step["name"] for step in added["steps"]] == ["overview", "step 2"]
    assert added["steps"][0]["param"].endswith("row=2&col=3&layer=pixel_mse")
    assert added["steps"][1]["param"].endswith("row=5&col=1&layer=pixel_mse")
    assert added["snapshots"] == 1 and added["name"] == ""
    assert added["rows"] == [["1. overview", "Go", "Copy link", "Remove"], ["2. step 2", "Go", "Copy link", "Remove"]]
    assert added["hint"] == "2 steps · snapshots last until the page reloads"


@NEEDS_JS
def test_story_exports_snapshots_through_the_shared_exporter_and_copies_step_links():
    """Snapshots download under numbered, slugged names, and a step's link points at the cube route with its state."""
    observed = _check("story")

    assert observed["exported"] == [{"name": "story_01_overview.png", "source": "data:image/png;base64,AAA"}]
    assert observed["linked"] == ["http://host:8772/#/cube/id%3D%252Fruns%252Fa%252Finference%252Fstamp_1%26view%3Dexplorer%26space%3Dphysical%26cmap%3Djet%26row%3D5%26col%3D1%26layer%3Dpixel_mse"]
    assert observed["none"] == {"saved": 0, "toast": ["warn"]}


@NEEDS_JS
def test_story_go_and_play_replay_the_steps_in_order_and_stop_with_the_tab():
    """Go applies one step, Play applies every step in order and relabels its button, and leaving the tab stops it."""
    observed = _check("story")

    assert observed["went"][-2:] == ["layer pixel_mse", "view explorer"] and "lock 2,3" in observed["went"]
    assert observed["playing"] == {"label": "Stop", "timers": 1}
    assert "lock 2,3" in observed["first"]
    assert observed["stopped"] == {"label": "Play", "timer": None, "timers": 0}
    assert observed["replayed"] == {"calls": ["lock 2,3", "lock 5,1"], "label": "Play"}


@NEEDS_JS
def test_story_removes_and_clears_steps_and_refuses_without_a_cube():
    """Removing a step drops it and its snapshot, Clear empties the story, and adding or linking without a loaded cube warns."""
    observed = _check("story")

    assert observed["removed"] == {"names": ["step 2"], "snapshots": 0, "rows": 1}
    assert observed["cleared"] == {"steps": 0, "rows": 0, "hint": "No steps yet; set up a view and press Add step."}
    assert observed["unloaded"] == {"toasts": ["Load a cube first.", "Load a cube first."], "steps": 0}


@NEEDS_JS
def test_links_are_shown_when_the_clipboard_is_unavailable():
    """On a page without clipboard access the link is shown in a toast instead of failing."""
    insecure = _check("story")["insecure"]

    assert len(insecure) == 1 and insecure[0].startswith("http://host:8772/#/cube/id%3D")


@NEEDS_JS
def test_story_snapshot_takes_the_visible_canvas_else_the_visible_image():
    """A snapshot prefers a drawn, visible canvas of the open view, else copies its visible image, else takes none."""
    assert _check("story")["snapshots"] == {"canvas": "canvas", "img": "copy 6x8", "none": None, "drawn": ["map"]}


def test_link_and_story_controls_are_on_the_page_once_and_wired_to_the_cube_view():
    """The Link and Story buttons sit with the colormap in the mode bar, the story panel carries every control once, and app.js hands each to the cube view."""
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    app  = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    tools = html.split('<span class="cube-modes__tools">')[1].split("</span>")[0]
    assert 'id="cube-link"' in tools and 'id="cube-story-toggle"' in tools and 'id="cube-cmap"' in tools

    assert html.count('id="cube-link"') == 1
    assert re.search(r'linkBtn\s*: document\.getElementById\("cube-link"\)', app)

    refs = {"toggle": "story-toggle", "panel": "story", "name": "story-name", "add": "story-add", "play": "story-play", "export": "story-export", "clear": "story-clear", "hint": "story-hint", "list": "story-list"}
    for ref, suffix in refs.items():
        assert html.count(f'id="cube-{suffix}"') == 1, suffix
        assert re.search(rf'{ref}\s*: document\.getElementById\("cube-{suffix}"\)', app), ref


def test_the_cube_route_hands_its_link_state_to_the_cube_view():
    """The router passes the #/cube/<state> parameter to the cube view's enter."""
    app = (WEBUI_ROOT / "static" / "js" / "app.js").read_text()

    assert 'if (route === "cube") this.cubeView.enter(param);' in app


def test_a_hidden_class_legend_takes_no_room_beside_the_colour_bar():
    """The class legend span sets display:flex, so its hidden state needs its own rule or it keeps half the legend row from a continuous layer's colour bar."""
    css = (WEBUI_ROOT / "static" / "css" / "styles.css").read_text()

    assert ".cube-pmap__classes { display: flex;" in css
    assert ".cube-pmap__classes[hidden] { display: none; }" in css


@NEEDS_JS
def test_cube_view_marks_its_page_staged_exactly_while_the_stage_shows():
    """Showing a cube stages the page, and selecting another cube or finding no cubes unstages it with the stage hidden."""
    observed = _check("stage")

    assert observed["before"] == {"hidden": True, "staged": False}
    assert observed["shown"] == {"hidden": False, "staged": True}
    assert observed["loading"] == {"hidden": True, "staged": False}
    assert observed["reshown"] == {"hidden": False, "staged": True}
    assert observed["empty"] == {"hidden": True, "staged": False}


def test_cube_page_header_hides_on_the_staged_class_and_not_on_a_has_rule():
    """The header hides on the is-staged class of the page holding the stage; no :has() rule watches the stage, since one on an ancestor re-checks on every DOM mutation."""
    css  = (WEBUI_ROOT / "static" / "css" / "styles.css").read_text()
    html = (WEBUI_ROOT / "static" / "index.html").read_text()
    view = (WEBUI_ROOT / "static" / "js" / "cube_view.js").read_text()

    assert '[data-page="cube"].is-staged .page__head { display: none; }' in css
    assert re.findall(r":has\([^)]*cube-stage", css) == []
    assert re.findall(r"^body:has\(", css, re.M) == []

    opening = html[:html.index('id="cube-stage"')].rsplit("<section ", 1)[1]
    assert opening.startswith('class="page page--pad" data-page="cube">')
    assert re.search(r'this\.page\s+= refs\.stage\.closest\("\.page"\);', view)
    assert len(re.findall(r"this\.stage\.hidden =", view)) == 1
