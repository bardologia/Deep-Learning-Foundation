
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline_library import PipelineLibrary

from tests.webui.conftest import WEBUI_ROOT


HARNESS    = WEBUI_ROOT / "tools" / "visual" / "stage_chips.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the pipeline transport")


@pytest.fixture(scope="module")
def chips(tmp_path_factory) -> dict:
    payload = tmp_path_factory.mktemp("stage_chips") / "pipelines.json"
    payload.write_text(json.dumps(PipelineLibrary().collect()), encoding="utf-8")

    return _run(payload)


def _run(payload: Path) -> dict:
    result = subprocess.run(["node", str(HARNESS), str(payload)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_every_animated_pipeline_gets_its_stage_chips(chips):
    stages = {pipeline["key"]: pipeline["stages"] for pipeline in PipelineLibrary().collect()}

    assert set(chips["pipelines"]) == set(chips["scenes"])
    for key, outcome in chips["pipelines"].items():
        assert outcome["chips"] == stages[key], key


def test_the_mapped_stage_never_moves_backwards(chips):
    for key, outcome in chips["pipelines"].items():
        stage_of = outcome["stageOf"]

        assert all(0 <= stage < len(outcome["chips"]) for stage in stage_of), key
        assert stage_of == sorted(stage_of), key


def test_a_chip_click_lights_its_own_stage_or_the_next_one_shown(chips):
    for key, outcome in chips["pipelines"].items():
        owned = sorted(set(outcome["stageOf"]))

        for click in outcome["clicks"]:
            expected = next((stage for stage in owned if stage >= click["stage"]), owned[-1])

            assert click["active"] == expected, (key, click)
            assert click["done"] == list(range(expected)), (key, click)
            assert outcome["stageOf"].index(expected) == click["chapter"], (key, click)


def test_most_stages_own_a_chapter(chips):
    for key, outcome in chips["pipelines"].items():
        assert len(outcome["chips"]) - len(set(outcome["stageOf"])) <= 1, key


def test_only_animated_pipelines_render_as_playable_rows(chips):
    keys = [pipeline["key"] for pipeline in PipelineLibrary().collect()]

    assert sorted(chips["rows"]) == sorted(keys)
    for key, row in chips["rows"].items():
        animated = key in chips["scenes"]

        assert row == {"playable": animated, "clickable": animated}, key
