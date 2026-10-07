"""Tests for the progress and duration formatting and the toast helper in api.js.

Each test executes api.js under node in a bare context and reads the result
back as JSON: a zero total, a missing finish time and a non-finite duration
format without NaN or a TypeError, and a toast shown before the page holds a
#toast element does nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webui.conftest import WEBUI_ROOT


API_JS = WEBUI_ROOT / "static" / "js" / "api.js"


def _run(tmp_path, expression: str, document: str = "{}"):
    """Evaluates an expression against api.js under node and returns its JSON-decoded value.

    Args:
        tmp_path   : Directory that receives the generated node script.
        expression : JavaScript expression evaluated in the api.js context.
        document   : JavaScript source of the document object the context exposes.

    Returns:
        The value of the expression after a JSON round trip.
    """
    script = tmp_path / "run_api.cjs"
    script.write_text(
        "const vm = require('node:vm');\n"
        "const fs = require('node:fs');\n"
        f"const source = fs.readFileSync({str(API_JS)!r}, 'utf8');\n"
        f"const context = vm.createContext({{ window: {{}}, document: {document} }});\n"
        "vm.runInContext(source, context);\n"
        f"process.stdout.write(JSON.stringify(vm.runInContext({expression!r}, context)));\n",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(script)], check=True, capture_output=True, text=True, timeout=60)

    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to execute api.js")
def test_progress_bits_with_a_zero_total_report_zero_percent(tmp_path):
    """A progress record with no items reports 0% instead of NaN%."""
    bits = _run(tmp_path, "Format.progressBits({ done: 0, failed: 0, total: 0 })")

    assert bits == ["0/0", "0%"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to execute api.js")
def test_progress_bits_without_a_finish_time_keep_the_eta(tmp_path):
    """An ETA with no finish time is still shown and the clock time is left out."""
    bits = _run(tmp_path, "Format.progressBits({ done: 5, failed: 0, total: 10, eta_s: 60, finish_at: null })")

    assert bits == ["5/10", "50%", "ETA 1m"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to execute api.js")
def test_duration_formats_missing_and_non_finite_seconds_as_zero(tmp_path):
    """A null or NaN duration formats as 0s, and an hour keeps its usual layout."""
    durations = _run(tmp_path, "[Format.duration(null), Format.duration(NaN), Format.duration(3600)]")

    assert durations == ["0s", "0s", "1h 00m"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to execute api.js")
def test_toast_without_its_element_does_nothing(tmp_path):
    """Showing a toast when the page has no #toast element returns without throwing."""
    shown = _run(tmp_path, "Toast.show('test message', 'error') === undefined", document="{ getElementById: () => null }")

    assert shown is True
