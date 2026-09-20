"""Round-trip test between a decision's candidate values and the field that edits them.

A sequence block declares the configuration axes it puts under test, each with
the candidate values the experiment compares, and the block editor edits that
list as one comma-separated field. Configuration values render as Python
literals, so a list or dict candidate carries commas of its own: splitting the
field on every comma tore ``[32, 32], [64, 32]`` into four candidates, offered
the fragments as verdicts, and stamped a fragment into every block downstream of
the verdict. The field is parsed by ``PythonLiteral.split`` instead, which breaks
only at the commas outside brackets, braces and quotes.

The parsing lives in the browser, so this module runs that real front-end helper
under node, against the strings the Python renderer produces for container
values.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from webui.config_value_render import ConfigValueRenderer

from tests.webui.conftest import WEBUI_ROOT


HARNESS = WEBUI_ROOT / "tools" / "visual" / "literal_split.js"

CONTAINER_VALUES = [
    [32, 32],
    [40, 40],
    (64, 32),
    ["CH01", "CH02", "CH03", "CH04"],
    {"features": [476, 476, 476], "dropout": 0.15, "params_trunk_wd": 0.0001},
    {"kernel_size": 3, "activation": "leaky_relu"},
]


def _split(fields: list, tmp_path: Path) -> list:
    """Returns each candidates field split into candidates by the browser's own helper.

    Args:
        fields   : Candidate fields as the block editor renders them, one comma-separated string per decision.
        tmp_path : Directory the harness payload is written into.

    Returns:
        One candidate list per field, in the order the fields were given.
    """
    payload = tmp_path / "fields.json"
    payload.write_text(json.dumps(fields), encoding="utf-8")

    result = subprocess.run(["node", str(HARNESS), str(payload)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the front-end literal helper")
def test_rendered_containers_stay_one_candidate_each(tmp_path):
    """List, tuple and dict values rendered by the launch form survive the field as one candidate apiece."""
    options = [ConfigValueRenderer.render(value) for value in CONTAINER_VALUES]

    assert _split([", ".join(options)], tmp_path) == [options]


def test_the_candidates_field_is_parsed_by_the_shared_helper():
    """The block editor reads the candidates field through the literal helper, never through a bare comma split."""
    source = (WEBUI_ROOT / "static" / "js" / "sequences.js").read_text()

    assert "decision.options = PythonLiteral.split(candidates.value);" in source
    assert 'candidates.value.split(",")' not in source
