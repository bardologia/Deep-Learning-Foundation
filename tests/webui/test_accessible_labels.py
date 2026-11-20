
from __future__ import annotations

import re
from html.parser import HTMLParser

from tests.webui.conftest import WEBUI_ROOT


JS_DIR = WEBUI_ROOT / "static" / "js"
INDEX  = WEBUI_ROOT / "static" / "index.html"


class PlaceholderInputAudit(HTMLParser):

    def __init__(self) -> None:
        super().__init__()
        self.label_depth = 0
        self.labelled    = set()
        self.candidates  = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        named = dict(attrs)
        if tag == "label":
            self.label_depth += 1
            if named.get("for"):
                self.labelled.add(named["for"])
        if tag == "input" and "placeholder" in named and "aria-label" not in named and not self.label_depth:
            self.candidates.append(named.get("id", ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "label":
            self.label_depth -= 1

    def unnamed(self, html: str) -> list:
        self.feed(html)
        return sorted(ident for ident in self.candidates if ident not in self.labelled)


def test_page_shell_placeholder_inputs_have_an_accessible_name():
    assert PlaceholderInputAudit().unnamed(INDEX.read_text()) == []


def test_markup_filter_boxes_carry_an_aria_label():
    unnamed = []
    for path in sorted(JS_DIR.glob("*.js")):
        for tag in re.findall(r"<input\b[^>]*>", path.read_text()):
            if ('type="search"' in tag or re.search(r'class="[^"]*filter', tag)) and "aria-label=" not in tag:
                unnamed.append((path.name, tag))

    assert unnamed == []


def test_built_filter_boxes_carry_an_aria_label():
    unnamed = []
    for path in sorted(JS_DIR.glob("*.js")):
        text = path.read_text()
        for name in set(re.findall(r"^\s*((?:this\.)?(?:search|filter))\.placeholder\s*=", text, re.M)):
            if f'{name}.setAttribute("aria-label",' not in text:
                unnamed.append((path.name, name))

    assert unnamed == []
