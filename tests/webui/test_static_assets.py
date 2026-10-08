"""Tests for the console's static asset layout and front-end module hygiene.

Checks that every JavaScript module on disk is loaded by index.html exactly once and in
dependency order, that shared helpers (MathJax bootstrap, HTML escaper, results roots,
network layer) are each defined in a single module, that launch widget modules hold
the expected number of classes, that every input tag declares a valid HTML type, and
that every cube view panel has the mode button that opens it.
"""

from __future__ import annotations

import re

from tests.webui.conftest import WEBUI_ROOT


JS_DIR = WEBUI_ROOT / "static" / "js"
INDEX  = WEBUI_ROOT / "static" / "index.html"


def _loaded_scripts() -> list:
    """Returns the JavaScript file names index.html loads, in document order."""
    html = INDEX.read_text()
    return [name for name in re.findall(r'<script src="/static/js/([\w.]+\.js)', html)]


def test_every_module_is_loaded_by_the_page_exactly_once():
    """The set of loaded scripts matches the modules on disk, with no duplicate tags."""
    loaded = _loaded_scripts()
    on_disk = sorted(path.name for path in JS_DIR.glob("*.js"))

    assert sorted(loaded) == on_disk
    assert len(loaded) == len(set(loaded))


def test_shared_helpers_load_before_their_users():
    """Helper modules appear in the page ahead of every module that depends on them."""
    loaded = _loaded_scripts()

    for helper in ("api.js", "results_sources.js", "shared_charts.js", "canvas_base.js"):
        assert loaded.index(helper) < loaded.index("app.js")

    assert loaded.index("shared_charts.js") < loaded.index("home.js")
    assert loaded.index("canvas_base.js") < loaded.index("gauges.js")

    for user in ("equations.js", "flow_view.js"):
        assert loaded.index("mathjax.js") < loaded.index(user)


def test_mathjax_bootstrap_lives_in_one_module():
    """Only mathjax.js holds the tex2svgPromise bootstrap."""
    holders = [path.name for path in JS_DIR.glob("*.js") if "tex2svgPromise" in path.read_text()]
    assert holders == ["mathjax.js"]


def test_static_assets_carry_no_hand_written_cache_tags():
    """index.html contains no manual ?v= cache-busting query strings."""
    assert "?v=" not in INDEX.read_text()


def test_the_html_escaper_is_defined_once():
    """No module keeps a private _esc helper; the single escaper lives on the shared charts class."""
    definitions = [path.name for path in JS_DIR.glob("*.js") if re.search(r"\b_esc\s*\(", path.read_text())]
    assert definitions == []

    escapers = [path.name for path in JS_DIR.glob("*.js") if "static esc(" in path.read_text()]
    assert escapers == ["shared_charts.js"]


def test_the_results_roots_are_declared_once():
    """The results root defaults and their storage key live only in results_sources.js, and only app.js feeds them from /api/project."""
    holders = [path.name for path in JS_DIR.glob("*.js") if "DEFAULT_RUNS" in path.read_text()]
    assert holders == ["results_sources.js"]

    configured = [path.name for path in JS_DIR.glob("*.js") if "ResultsSources.configure(" in path.read_text()]
    assert configured == ["app.js"]

    readers = [path.name for path in JS_DIR.glob("*.js") if '"results-sources"' in path.read_text()]
    assert readers == ["results_sources.js"]


def test_the_network_layer_is_reached_through_its_class():
    """No module calls the old window-level api or toast helpers; the Api class is the entry point."""
    for path in JS_DIR.glob("*.js"):
        text = path.read_text()
        assert "window.apiGet(" not in text, path.name
        assert "window.apiPost(" not in text, path.name
        assert "window.toast(" not in text, path.name

    assert "class Api {" in (JS_DIR / "api.js").read_text()


def test_every_input_declares_a_valid_html_type():
    """Every input tag in index.html uses a type the browser knows, so no slider degrades to a text box."""
    valid = {"button", "checkbox", "color", "date", "datetime-local", "email", "file", "hidden", "image", "month", "number", "password", "radio", "range", "reset", "search", "submit", "tel", "text", "time", "url", "week"}
    types = re.findall(r'<input[^>]*\btype="([^"]*)"', INDEX.read_text())

    assert types
    assert sorted(set(types) - valid) == []


def test_cube_sliders_are_range_inputs():
    """The point-cloud threshold, metric threshold, overlay alpha and coverage controls render as sliders."""
    html = INDEX.read_text()

    for control in ("cube-cloud-thr", "cube-metric-thr", "cube-metric-alpha", "cube-selective-cov"):
        assert re.search(rf'<input type="range" id="{control}"', html), control


def test_every_cube_view_panel_has_a_mode_button():
    """Each cube view panel on the page is opened by a mode button, so no panel ships without a way to reach it."""
    html    = INDEX.read_text()
    panels  = re.findall(r'<div class="cube-view"[^>]*\bdata-view="([^"]+)"', html)
    buttons = re.findall(r'<button type="button" class="cube-mode[^"]*" data-view="([^"]+)"', html)

    assert panels
    assert sorted(panels) == sorted(buttons)


def test_launch_widget_modules_hold_one_class_family_each():
    """Each launch widget module declares its expected class count and the merged module is gone."""
    counts = {path.name: len(re.findall(r"^class ", path.read_text(), re.M)) for path in JS_DIR.glob("*.js")}

    assert counts["python_literal.js"] == 1
    assert counts["config_form.js"] == 1
    assert counts["form_widgets.js"] == 3
    assert counts["gpu_pickers.js"] == 2
    assert counts["model_panels.js"] == 3
    assert "launch_widgets.js" not in counts
