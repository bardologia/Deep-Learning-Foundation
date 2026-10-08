"""Tests for the stage chips above a pipeline's process animation.

The modal shows the pipeline's stages from ``PipelineLibrary`` as chips over the
animation, whose chapters come from ``ProcessAnimator.SCENES``. The transport
maps every chapter onto a stage by shared words, falling back to the chapter's
proportional position, and never lets the mapped stage move backwards as the
animation advances. A click on a chip seeks the first chapter of that stage, or
of the next stage that owns one, and the chip of the playing chapter lights up
with every earlier chip marked done.

The mapping lives in the browser, so this module runs the real transport under
node against a stub DOM and a stub animator, over the real stage and chapter
catalogs.
"""

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
    """Returns the scene keys and, per animated pipeline, its chips, chapter-to-stage map and click outcomes."""
    payload = tmp_path_factory.mktemp("stage_chips") / "stages.json"
    payload.write_text(json.dumps({pipeline["key"]: pipeline["stages"] for pipeline in PipelineLibrary().collect()}), encoding="utf-8")

    return _run(payload)


def _run(payload: Path) -> dict:
    """Runs the stage chip harness on a stage payload.

    Args:
        payload : JSON file mapping each pipeline key to its stage names.

    Returns:
        The harness output: ``scenes`` and the per-pipeline ``chips``, ``stageOf`` and ``clicks``.
    """
    result = subprocess.run(["node", str(HARNESS), str(payload)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_every_animated_pipeline_gets_its_stage_chips(chips):
    """Each scene with a pipeline card renders one chip per stage of that card."""
    stages = {pipeline["key"]: pipeline["stages"] for pipeline in PipelineLibrary().collect()}

    assert set(chips["pipelines"]) == set(chips["scenes"])
    for key, outcome in chips["pipelines"].items():
        assert outcome["chips"] == stages[key], key


def test_the_mapped_stage_never_moves_backwards(chips):
    """Every chapter maps onto an existing chip, in non-decreasing stage order."""
    for key, outcome in chips["pipelines"].items():
        stage_of = outcome["stageOf"]

        assert all(0 <= stage < len(outcome["chips"]) for stage in stage_of), key
        assert stage_of == sorted(stage_of), key


def test_a_chip_click_lights_its_own_stage_or_the_next_one_shown(chips):
    """A click lights the clicked chip when its stage owns a chapter, else the next stage that does, with every earlier chip done."""
    for key, outcome in chips["pipelines"].items():
        owned = sorted(set(outcome["stageOf"]))

        for click in outcome["clicks"]:
            expected = next((stage for stage in owned if stage >= click["stage"]), owned[-1])

            assert click["active"] == expected, (key, click)
            assert click["done"] == list(range(expected)), (key, click)
            assert outcome["stageOf"].index(expected) == click["chapter"], (key, click)


def test_most_stages_own_a_chapter(chips):
    """Name matching leaves at most one stage per pipeline without a chapter of its own."""
    for key, outcome in chips["pipelines"].items():
        assert len(outcome["chips"]) - len(set(outcome["stageOf"])) <= 1, key
