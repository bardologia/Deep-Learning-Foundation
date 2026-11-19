
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
    payload = tmp_path / "fields.json"
    payload.write_text(json.dumps(fields), encoding="utf-8")

    result = subprocess.run(["node", str(HARNESS), str(payload)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the front-end literal helper")
def test_rendered_containers_stay_one_candidate_each(tmp_path):
    options = [ConfigValueRenderer.render(value) for value in CONTAINER_VALUES]

    assert _split([", ".join(options)], tmp_path) == [options]


def test_the_candidates_field_is_parsed_by_the_shared_helper():
    source = (WEBUI_ROOT / "static" / "js" / "sequences.js").read_text()

    assert "decision.options = PythonLiteral.split(candidates.value);" in source
    assert 'candidates.value.split(",")' not in source
