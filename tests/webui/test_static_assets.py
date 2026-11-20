
from __future__ import annotations

import re

from tests.webui.conftest import WEBUI_ROOT


JS_DIR = WEBUI_ROOT / "static" / "js"
INDEX  = WEBUI_ROOT / "static" / "index.html"


def _loaded_scripts() -> list:
    html = INDEX.read_text()
    return [name for name in re.findall(r'<script src="/static/js/([\w.]+\.js)', html)]


def test_every_module_is_loaded_by_the_page_exactly_once():
    loaded = _loaded_scripts()
    on_disk = sorted(path.name for path in JS_DIR.glob("*.js"))

    assert sorted(loaded) == on_disk
    assert len(loaded) == len(set(loaded))


def test_shared_helpers_load_before_their_users():
    loaded = _loaded_scripts()

    for helper in ("api.js", "results_sources.js", "shared_charts.js", "canvas_base.js"):
        assert loaded.index(helper) < loaded.index("app.js")

    assert loaded.index("shared_charts.js") < loaded.index("home.js")
    assert loaded.index("canvas_base.js") < loaded.index("gauges.js")

    assert loaded.index("api.js") < loaded.index("stream_hub.js")

    assert loaded.index("stream_hub.js") < loaded.index("console.js")


def test_static_assets_carry_no_hand_written_cache_tags():
    assert "?v=" not in INDEX.read_text()


def test_the_html_escaper_is_defined_once():
    definitions = [path.name for path in JS_DIR.glob("*.js") if re.search(r"\b_esc\s*\(", path.read_text())]
    assert definitions == []

    escapers = [path.name for path in JS_DIR.glob("*.js") if "static esc(" in path.read_text()]
    assert escapers == ["shared_charts.js"]


def test_the_results_roots_are_declared_once():
    holders = [path.name for path in JS_DIR.glob("*.js") if "DEFAULT_RUNS" in path.read_text()]
    assert holders == ["results_sources.js"]

    configured = [path.name for path in JS_DIR.glob("*.js") if "ResultsSources.configure(" in path.read_text()]
    assert configured == ["app.js"]

    readers = [path.name for path in JS_DIR.glob("*.js") if '"results-sources"' in path.read_text()]
    assert readers == ["results_sources.js"]


def test_the_network_layer_is_reached_through_its_class():
    for path in JS_DIR.glob("*.js"):
        text = path.read_text()
        assert "window.apiGet(" not in text, path.name
        assert "window.apiPost(" not in text, path.name
        assert "window.toast(" not in text, path.name

    assert "class Api {" in (JS_DIR / "api.js").read_text()


def test_every_input_declares_a_valid_html_type():
    valid = {"button", "checkbox", "color", "date", "datetime-local", "email", "file", "hidden", "image", "month", "number", "password", "radio", "range", "reset", "search", "submit", "tel", "text", "time", "url", "week"}
    types = re.findall(r'<input[^>]*\btype="([^"]*)"', INDEX.read_text())

    assert types
    assert sorted(set(types) - valid) == []


def test_cube_sliders_are_range_inputs():
    html = INDEX.read_text()

    for control in ("cube-cloud-thr", "cube-metric-thr", "cube-metric-alpha", "cube-selective-cov"):
        assert re.search(rf'<input type="range" id="{control}"', html), control


def test_every_cube_view_panel_has_a_mode_button():
    html    = INDEX.read_text()
    panels  = re.findall(r'<div class="cube-view"[^>]*\bdata-view="([^"]+)"', html)
    buttons = re.findall(r'<button type="button" class="cube-mode[^"]*" data-view="([^"]+)"', html)

    assert panels
    assert sorted(panels) == sorted(buttons)


def test_cube_page_names_its_axes_row_col_and_z():
    html = INDEX.read_text()
    page = html[html.index('<section class="page page--pad" data-page="cube">'):html.index('<section class="page page--pad" data-page="slices">')]

    labels = re.findall(r'<span class="cube-axis cube-axis--[xy]">(\w+) &rarr;</span>', page)
    cuts   = re.findall(r'<section class="cube-cutgroup" data-axis="(\w+)">\s*<h3 class="cube-cutgroup__title">\s*\w+ cut <span class="cube-cutgroup__sub">[^<]* at the marked (\w+)</span>', page)
    hints  = re.findall(r'<span class="cube-z__at" id="cube-(\w+)-at">Play the sweep, or scroll a panel to move through (\w+)\.</span>', page)
    bins   = re.findall(r'aria-label="Set a (\w+) bin">\s*<span class="cube-jump__cell">\s*<label for="cube-(\w+)-input">', page)

    assert sorted(labels) == ["col", "row"]
    assert sorted(cuts) == [("col", "row"), ("row", "col")]
    assert sorted(hints) == [("col", "col"), ("row", "row"), ("z", "z")]
    assert sorted(bins) == [("col", "col"), ("row", "row"), ("z", "z")]
    assert 'aria-label="Set a cut by row and col"' in page

def test_launch_widget_modules_hold_one_class_family_each():
    counts = {path.name: len(re.findall(r"^class ", path.read_text(), re.M)) for path in JS_DIR.glob("*.js")}

    assert counts["python_literal.js"] == 1
    assert counts["config_form.js"] == 1
    assert counts["form_widgets.js"] == 3
    assert counts["gpu_pickers.js"] == 2
    assert counts["model_panels.js"] == 3
    assert "launch_widgets.js" not in counts
