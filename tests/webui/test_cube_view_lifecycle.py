"""Tests for the cube page's front-end lifecycle, run against the real cube_view.js.

This module runs ``cube_view.js`` under node with no DOM through
``webui/tools/visual/cube_view_checks.js``. The cube view keeps decoded slice images as
ImageBitmaps, whose pixel memory is only returned when they are closed, so evicted,
cleared, released and replaced bitmaps must close while a bitmap a sweep still shows is
kept. The point cloud's section-cut plane must bound itself by the cloud's true extent,
hide the points beyond it, and stop its sweep loop when the view or the tab is left. The
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
cleared meanwhile, and named lines remembered per cube footprint.
"""

from __future__ import annotations

import json
import math
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
