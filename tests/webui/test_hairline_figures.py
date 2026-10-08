"""Tests that the hairline figures and thinking orbs the frontend names all exist.

The `<hairline-figure>` element throws on an unknown figure name and `<thinking-orb>` on an
unknown state, so a typo only surfaces when the page that carries it renders. These checks
read the figure catalogs, the orb states and every place index.html and the view modules
name a figure or a state, and pin the order the engines and their elements load in.
"""
from __future__ import annotations

import re

from tests.webui.conftest import WEBUI_ROOT


STATIC  = WEBUI_ROOT / "static"
JS_DIR  = STATIC / "js"
INDEX   = STATIC / "index.html"
STOCK   = STATIC / "vendor" / "hairline-figures.js"
PROJECT = JS_DIR / "hairline_project.js"
ORB     = JS_DIR / "thinking_orb.js"
SCRIPTS = re.compile(r'<script src="/static/((?:js|vendor)/[\w.-]+\.js)"')


def _stock_figures() -> set[str]:
    """Returns the figure names the vendored stock bundle exports."""
    body = re.search(r"window\.HairlineStock = \{([^}]*)\}", STOCK.read_text()).group(1)
    return {name.strip() for name in body.split(",")}


def _project_figures() -> set[str]:
    """Returns the figure names the project catalog registers."""
    return set(re.findall(r"^    (\w+):\s+ProjectFigures\.entry\(", PROJECT.read_text(), re.M))


def _view_sources() -> dict[str, str]:
    """Returns the text of index.html and of every view module except the figure and orb definitions."""
    skip  = {PROJECT.name, "hairline_figure.js", ORB.name}
    texts = {path.name: path.read_text() for path in JS_DIR.glob("*.js") if path.name not in skip}
    texts[INDEX.name] = INDEX.read_text()
    return texts


def _named_figures() -> set[str]:
    """Returns every figure name index.html and the view modules mention."""
    patterns = [
        r'<hairline-figure name="(\w+)"',
        r'data-figure="(\w+)"',
        r'HairlineFigure\.(?:html|empty)\("(\w+)"',
        r'\[/\^[^\]]*/, "(\w+)"\]',
        r'\breturn match \? match\[1\] : "(\w+)"',
    ]
    names = set()
    for text in _view_sources().values():
        for pattern in patterns:
            names.update(re.findall(pattern, text))
        for table in re.findall(r"static FIGURES\s*= \{([^}]*)\}", text):
            names.update(re.findall(r':\s*"(\w+)"', table))
    return names


def _orb_states() -> set[str]:
    """Returns the states the thinking-orb element accepts."""
    body = re.search(r"static STATES = \[([^\]]*)\]", ORB.read_text()).group(1)
    return set(re.findall(r'"(\w+)"', body))


def _named_states() -> set[str]:
    """Returns every orb state index.html and the view modules mention."""
    names = set()
    for text in _view_sources().values():
        names.update(re.findall(r'<thinking-orb[^>]*?\sstate="(\w+)"', text))
        names.update(re.findall(r'ThinkingOrb\.(?:html|busy|panel)\("(\w+)"', text))
        for table in re.findall(r"static ORB_STATES\s*= \{([^}]*)\}", text):
            names.update(re.findall(r':\s*"(\w+)"', table))
    return names


def _loaded() -> list[str]:
    """Returns the scripts index.html loads from the static tree, in document order."""
    return SCRIPTS.findall(INDEX.read_text())


def test_stock_and_project_catalogs_do_not_share_a_name():
    """A project figure never shadows a stock figure of the same name."""
    assert sorted(_stock_figures() & _project_figures()) == []


def test_every_named_figure_is_in_a_catalog():
    """Every figure the pages and views name is drawn by the stock bundle or the project catalog."""
    known = _stock_figures() | _project_figures()

    assert sorted(_named_figures() - known) == []


def test_every_catalog_figure_is_shown_somewhere():
    """Every vendored and project figure is named by at least one page or view."""
    named = _named_figures()

    assert sorted((_stock_figures() | _project_figures()) - named) == []


def test_every_named_orb_state_exists():
    """Every orb state the pages and views name is one the element draws."""
    assert sorted(_named_states() - _orb_states()) == []


def test_engines_load_before_their_elements():
    """The vendored engines load ahead of the elements built on them, and the elements ahead of the views."""
    loaded = _loaded()

    assert loaded.index("vendor/thinking-orbs-engine.js") < loaded.index("js/thinking_orb.js")
    assert loaded.index("vendor/hairline-kernel.js") < loaded.index("vendor/hairline-figures.js")
    assert loaded.index("vendor/hairline-figures.js") < loaded.index("js/hairline_project.js")
    assert loaded.index("js/hairline_project.js") < loaded.index("js/hairline_figure.js")

    for element in ("js/thinking_orb.js", "js/hairline_figure.js"):
        assert loaded.index(element) < loaded.index("js/home.js")
        assert loaded.index(element) < loaded.index("js/app.js")
