"""Tests for the Home board's neighbour impact card and active users table, run against the real home.js.

This module runs ``home.js`` under node with a stub DOM through
``webui/tools/visual/impact_checks.js``. The card reads the contention monitor's
per-user slice payload: each meter shows the worst neighbour's stall on its
resource, each share line names that neighbour, the stats row counts the stalled
neighbours, the figure weighs this user's own slice stall against the worst
neighbour's, and missing kernel sources or a console outside the user slice are
said in the lede. The active users table gains a cpu/ram/disk stall column read
from the same slices.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


HARNESS  = WEBUI_ROOT / "tools" / "visual" / "impact_checks.js"
STYLES   = WEBUI_ROOT / "static" / "css" / "styles.css"
NEEDS_JS = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run home.js")


def _check(scenario: str) -> dict:
    """Runs one harness scenario and returns what it observed.

    Args:
        scenario : Name of the ImpactChecks method to run.

    Returns:
        The JSON object the scenario printed.
    """
    result = subprocess.run(["node", str(HARNESS), scenario], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@NEEDS_JS
def test_users_table_has_a_stall_column_on_every_row_and_in_the_grid():
    """The head, every row and the CSS grid template all carry the nine columns, the stall column third."""
    observed = _check("users")
    template = re.search(r"\.utable__row \{[^}]*grid-template-columns: ([^;]*);", STYLES.read_text()).group(1)

    assert observed["head"] == ["user", "cpu%", "stall c/m/d", "ram", "ram%", "gpu mem", "gpus", "procs", "ssh"]
    assert observed["columns"] == [9, 9, 9, 9]
    assert len(re.findall(r"minmax\(", template)) == 9


@NEEDS_JS
def test_users_table_reads_each_users_stall_from_their_slice():
    """A user's cpu/mem/io stall is joined by uid, graded by its worst resource, and a user without a slice reads --."""
    observed = _check("users")

    assert observed["stall"] == [
        {"text": "1/0/2", "level": ""},
        {"text": "3/25/0", "level": "is-hot"},
        {"text": "6/0/1", "level": "is-mid"},
        {"text": "--", "level": ""},
    ]


@NEEDS_JS
def test_impact_meters_and_share_lines_follow_the_worst_neighbour():
    """Each meter reads the worst neighbour's stall and its share line names that neighbour beside your share."""
    mine = _check("impact")["mine"]

    assert mine["meters"] == {"mem": 33.4, "io": 0, "cpu": 12}
    assert mine["shares"]["mem"]["html"] == "your share <b>62</b>% &middot; worst alice <b>33</b>%"
    assert mine["shares"]["cpu"]["html"] == "your share <b>30</b>% &middot; worst bob <b>12</b>%"
    assert mine["stalled"] == "<b>2</b> &middot; of 2"


@NEEDS_JS
def test_impact_share_is_dominant_only_on_a_contended_resource_the_backend_calls_dominant():
    """The dominant mark follows the backend's dominant flag on a resource under alarm, never a fixed share."""
    observed = _check("impact")

    assert {kind: share["dominant"] for kind, share in observed["mine"]["shares"].items()} == {"mem": True, "io": False, "cpu": False}
    assert not any(share["dominant"] for share in observed["notMine"]["shares"].values())


@NEEDS_JS
def test_impact_figure_weighs_your_own_slice_stall_against_the_worst_neighbour():
    """The figure gets your slice's worst stall and the worst neighbour stall as fractions, lit while you are blamed."""
    observed = _check("impact")

    assert observed["mine"]["figure"] == {"mine": 0.08, "others": 0.334, "alarm": True}
    assert observed["notMine"]["figure"] == {"mine": 0.08, "others": 0.334, "alarm": False}
    assert observed["outside"]["figure"] == {"mine": 0.08, "others": 0, "alarm": False}


@NEEDS_JS
def test_impact_verdicts_and_ledes_name_the_cause():
    """Blamed, not blamed and quiet samples each get their own verdict and lede."""
    observed = _check("impact")

    assert observed["mine"]["verdict"] == "you are the cause"
    assert observed["mine"]["lede"] == "Other users' tasks are stalling on the resources flagged below, and you are their dominant consumer."
    assert observed["notMine"]["verdict"] == "busy, not you"
    assert observed["notMine"]["lede"] == "Other users are stalling, but you are not the dominant consumer. The largest RAM holder is alice (40 GB, 9 procs)."


@NEEDS_JS
def test_impact_says_when_the_kernel_sources_are_missing():
    """Without /proc/pressure or the user slices the card turns off and the lede names what is missing."""
    missing = _check("impact")["missing"]

    assert missing["verdict"] == "unavailable"
    assert missing["light"] == "impact__light is-off"
    assert missing["lede"] == "Cannot measure neighbour stall: the kernel exposes no /proc/pressure; the host has no per-user cgroup slices."
    assert missing["shares"]["io"]["html"] == "your share <b>0</b>% &middot; no neighbour"


@NEEDS_JS
def test_impact_says_when_the_console_runs_outside_your_user_slice():
    """A quiet sample from a console outside the user slice warns that its jobs are not charged to you."""
    outside = _check("impact")["outside"]

    assert outside["verdict"] == "all clear"
    assert outside["lede"] == "No neighbour is stalling. This console runs outside your user slice, so jobs launched here are not charged to you."
