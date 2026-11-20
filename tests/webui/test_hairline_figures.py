from __future__ import annotations

import re

from tests.webui.conftest import WEBUI_ROOT


STATIC  = WEBUI_ROOT / "static"
JS_DIR  = STATIC / "js"
INDEX   = STATIC / "index.html"
STOCK   = STATIC / "vendor" / "hairline-figures.js"
PROJECT = JS_DIR / "hairline_project.js"
SCRIPTS = re.compile(r'<script src="/static/((?:js|vendor)/[\w.-]+\.js)"')


def _stock_figures() -> set[str]:
    body = re.search(r"window\.HairlineStock = \{([^}]*)\}", STOCK.read_text()).group(1)
    return {name.strip() for name in body.split(",")}


def _project_figures() -> set[str]:
    return set(re.findall(r"^    (\w+):\s+ProjectFigures\.entry\(", PROJECT.read_text(), re.M))


def _view_sources() -> dict[str, str]:
    skip  = {PROJECT.name, "hairline_figure.js"}
    texts = {path.name: path.read_text() for path in JS_DIR.glob("*.js") if path.name not in skip}
    texts[INDEX.name] = INDEX.read_text()
    return texts


def _named_figures() -> set[str]:
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


def _loaded() -> list[str]:
    return SCRIPTS.findall(INDEX.read_text())


def test_stock_and_project_catalogs_do_not_share_a_name():
    assert sorted(_stock_figures() & _project_figures()) == []


def test_every_named_figure_is_in_a_catalog():
    known = _stock_figures() | _project_figures()

    assert sorted(_named_figures() - known) == []


def test_every_catalog_figure_is_shown_somewhere():
    named = _named_figures()

    assert sorted((_stock_figures() | _project_figures()) - named) == []


def test_engines_load_before_their_elements():
    loaded = _loaded()

    assert loaded.index("vendor/hairline-kernel.js") < loaded.index("vendor/hairline-figures.js")
    assert loaded.index("vendor/hairline-figures.js") < loaded.index("js/hairline_project.js")
    assert loaded.index("js/hairline_project.js") < loaded.index("js/hairline_figure.js")

    assert loaded.index("js/hairline_figure.js") < loaded.index("js/home.js")
    assert loaded.index("js/hairline_figure.js") < loaded.index("js/app.js")
