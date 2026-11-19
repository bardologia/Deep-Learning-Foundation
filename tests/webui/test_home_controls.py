
from __future__ import annotations

import re

from tests.webui.conftest import WEBUI_ROOT


HOME = WEBUI_ROOT / "static" / "js" / "home.js"


def _card(label: str) -> str:
    text  = HOME.read_text()
    start = text.index(f'aria-label="{label}"')
    return text[start:text.index("</section>", start)]


def _method(name: str) -> str:
    text  = HOME.read_text()
    start = text.index(f"  {name}() {{")
    end   = re.compile(r"^  \w+\(.*\) \{$", re.M).search(text, start + 1)
    return text[start:end.start()]


def test_kill_switches_card_pairs_the_nuke_and_kill_front_end_under_their_figures():
    card  = _card("Kill switches")
    cells = card.split('<div class="kill__cell">')[1:]

    assert len(cells) == 2
    assert 'HairlineFigure.html("nuke"' in cells[0] and 'id="sb-nuke"' in cells[0]
    assert 'HairlineFigure.html("guillotine"' in cells[1] and 'id="sb-kill-ui"' in cells[1]
    assert cells[0].index("HairlineFigure.html") < cells[0].index('id="sb-nuke"')
    assert cells[1].index("HairlineFigure.html") < cells[1].index('id="sb-kill-ui"')


def test_kill_front_end_left_the_watchdog_card_and_the_column_stacks_notify_over_the_kill_switches():
    text = HOME.read_text()

    assert 'id="sb-kill-ui"' not in _card("Resource watchdog")
    assert text.count('id="sb-kill-ui"') == 1

    stack = text.split('<div class="ctl__stack">')[1]
    assert stack.index('aria-label="Job notifications"') < stack.index('aria-label="Kill switches"')


def test_nuke_texts_promise_only_local_kills():
    card    = _card("Kill switches").split('<div class="kill__cell">')[1]
    confirm = re.search(r'window\.confirm\("(NUKE:[^"]*)"\)', _method("_wireNuke")).group(1)
    texts   = re.findall(r'title="([^"]*)"', card) + re.findall(r">([^<>]+)</p>", card) + [confirm]

    assert len(texts) == 4
    for text in texts:
        assert "cancel" not in text.lower(), text
        assert "your user" in text or "of yours" in text, text


def test_nuke_and_kill_front_end_drive_their_figures():
    nuke = _method("_wireNuke")
    kill = _method("_wireKillUi")

    assert 'document.getElementById("sb-nuke-figure")' in nuke
    assert "figure.state = { firing: on };" in nuke
    assert nuke.index("fire(true);") < nuke.index('Api.post("/api/system/nuke")') < nuke.index("fire(false);")

    assert 'document.getElementById("sb-kill-figure")' in kill
    assert kill.index("this._serverDown = true;") < kill.index("figure.state = { down: true };")
