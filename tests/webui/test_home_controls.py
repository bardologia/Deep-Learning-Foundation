"""Tests for the Home board's control cards as StatusBoard builds them in home.js.

The middle control column stacks the notify card over a kill switches card that pairs the
nuke with kill front end, each under its own figure, and the watchdog card no longer
carries kill front end. The nuke reaches this user's processes on this machine only, so
none of its texts promise to cancel jobs anywhere else. Firing the nuke and killing the
front end hand their state to the figures that draw them.
"""

from __future__ import annotations

import re

from tests.webui.conftest import WEBUI_ROOT


HOME = WEBUI_ROOT / "static" / "js" / "home.js"


def _card(label: str) -> str:
    """Returns the markup of the Home board section with the given aria label.

    Args:
        label : The aria-label of the section, e.g. "Kill switches".

    Returns:
        The template text from the section's opening tag to its closing tag.
    """
    text  = HOME.read_text()
    start = text.index(f'aria-label="{label}"')
    return text[start:text.index("</section>", start)]


def _method(name: str) -> str:
    """Returns the source of one StatusBoard method, from its signature to the next method."""
    text  = HOME.read_text()
    start = text.index(f"  {name}() {{")
    end   = re.compile(r"^  \w+\(.*\) \{$", re.M).search(text, start + 1)
    return text[start:end.start()]


def test_kill_switches_card_pairs_the_nuke_and_kill_front_end_under_their_figures():
    """The kill switches card holds both buttons, each in its own cell under its figure."""
    card  = _card("Kill switches")
    cells = card.split('<div class="kill__cell">')[1:]

    assert len(cells) == 2
    assert 'HairlineFigure.html("nuke"' in cells[0] and 'id="sb-nuke"' in cells[0]
    assert 'HairlineFigure.html("guillotine"' in cells[1] and 'id="sb-kill-ui"' in cells[1]
    assert cells[0].index("HairlineFigure.html") < cells[0].index('id="sb-nuke"')
    assert cells[1].index("HairlineFigure.html") < cells[1].index('id="sb-kill-ui"')


def test_kill_front_end_left_the_watchdog_card_and_the_column_stacks_notify_over_the_kill_switches():
    """Kill front end is only on the kill switches card, which follows notify inside one stack."""
    text = HOME.read_text()

    assert 'id="sb-kill-ui"' not in _card("Resource watchdog")
    assert text.count('id="sb-kill-ui"') == 1

    stack = text.split('<div class="ctl__stack">')[1]
    assert stack.index('aria-label="Job notifications"') < stack.index('aria-label="Kill switches"')


def test_nuke_texts_promise_only_local_kills():
    """The nuke's hint, button title and confirmation speak of this user's processes and cancel nothing remote."""
    card    = _card("Kill switches").split('<div class="kill__cell">')[1]
    confirm = re.search(r'window\.confirm\("(NUKE:[^"]*)"\)', _method("_wireNuke")).group(1)
    texts   = re.findall(r'title="([^"]*)"', card) + re.findall(r">([^<>]+)</p>", card) + [confirm]

    assert len(texts) == 4
    for text in texts:
        assert "cancel" not in text.lower(), text
        assert "your user" in text or "of yours" in text, text


def test_nuke_and_kill_front_end_drive_their_figures():
    """Firing the nuke sets the nuke figure firing for the request's span, and a stopped front end leaves the guillotine down."""
    nuke = _method("_wireNuke")
    kill = _method("_wireKillUi")

    assert 'document.getElementById("sb-nuke-figure")' in nuke
    assert "figure.state = { firing: on };" in nuke
    assert nuke.index("fire(true);") < nuke.index('Api.post("/api/system/nuke")') < nuke.index("fire(false);")

    assert 'document.getElementById("sb-kill-figure")' in kill
    assert kill.index("this._serverDown = true;") < kill.index("figure.state = { down: true };")
