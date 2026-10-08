"""Tests that the console's placeholder-only inputs carry an accessible name.

A placeholder disappears once the user types and is not a reliable accessible
name, so every text input in the page shell needs an ``aria-label``, an
enclosing ``<label>`` or a ``<label for>``, and every filter or search box the
front-end modules build needs an ``aria-label`` of its own.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from tests.webui.conftest import WEBUI_ROOT


JS_DIR = WEBUI_ROOT / "static" / "js"
INDEX  = WEBUI_ROOT / "static" / "index.html"


class PlaceholderInputAudit(HTMLParser):
    """Collects the placeholder inputs of a page that have no accessible name.

    Attributes:
        label_depth : Number of ``<label>`` elements currently open around the parser position.
        labelled    : Element ids some ``<label for>`` names.
        candidates  : Ids of placeholder inputs with neither an ``aria-label`` nor an enclosing label.
    """

    def __init__(self) -> None:
        """Starts with no open label, no labelled ids and no candidates."""
        super().__init__()
        self.label_depth = 0
        self.labelled    = set()
        self.candidates  = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Tracks open labels, the ids they name, and placeholder inputs outside any label.

        Args:
            tag   : Lower-cased tag name.
            attrs : Attribute name and value pairs of the tag.
        """
        named = dict(attrs)
        if tag == "label":
            self.label_depth += 1
            if named.get("for"):
                self.labelled.add(named["for"])
        if tag == "input" and "placeholder" in named and "aria-label" not in named and not self.label_depth:
            self.candidates.append(named.get("id", ""))

    def handle_endtag(self, tag: str) -> None:
        """Closes the innermost open label.

        Args:
            tag : Lower-cased tag name.
        """
        if tag == "label":
            self.label_depth -= 1

    def unnamed(self, html: str) -> list:
        """Returns the ids of placeholder inputs no label or aria-label names.

        Args:
            html : Page markup to audit.

        Returns:
            Sorted ids of the unnamed inputs, an empty string for an input without id.
        """
        self.feed(html)
        return sorted(ident for ident in self.candidates if ident not in self.labelled)


def test_page_shell_placeholder_inputs_have_an_accessible_name():
    """Every placeholder input in index.html is named by an aria-label, an enclosing label or a label for."""
    assert PlaceholderInputAudit().unnamed(INDEX.read_text()) == []


def test_markup_filter_boxes_carry_an_aria_label():
    """Search and filter inputs written as markup in the modules carry an aria-label."""
    unnamed = []
    for path in sorted(JS_DIR.glob("*.js")):
        for tag in re.findall(r"<input\b[^>]*>", path.read_text()):
            if ('type="search"' in tag or re.search(r'class="[^"]*filter', tag)) and "aria-label=" not in tag:
                unnamed.append((path.name, tag))

    assert unnamed == []


def test_built_filter_boxes_carry_an_aria_label():
    """Search and filter inputs built through the DOM get an aria-label next to their placeholder."""
    unnamed = []
    for path in sorted(JS_DIR.glob("*.js")):
        text = path.read_text()
        for name in set(re.findall(r"^\s*((?:this\.)?(?:search|filter))\.placeholder\s*=", text, re.M)):
            if f'{name}.setAttribute("aria-label",' not in text:
                unnamed.append((path.name, name))

    assert unnamed == []
