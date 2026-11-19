
from __future__ import annotations

import json
import shutil
import subprocess
from enum    import Enum
from pathlib import Path

import pytest

from configuration.synthesis                  import SynthesisConfig
from configuration.training                   import BackboneEntryConfig
from pipelines.backbone.training.launcher     import BackboneTrainingLauncher, TrainScheduler
from pipelines.shared.dataset.dataset_spatial import Layout
from tools.runtime.config_cli                 import ConfigCli
from webui.config_value_render                import ConfigValueRenderer


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS   = REPO_ROOT / "webui" / "tools" / "visual" / "trial_names.js"
KEY       = "train_backbone"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root   = tmp_path_factory.mktemp("dataset")
    layout = root / Layout.LAYOUT_RELPATH
    (root / "meta").mkdir()
    layout.parent.mkdir()
    layout.write_text(json.dumps({"channel_labels": SynthesisConfig().channel_labels}))

    return root


def _sweep_argv(dataset: Path) -> list[str]:
    return ["--paths.dataset_path", str(dataset)]


def _payload(argv: list[str]) -> dict:
    config = ConfigCli(BackboneEntryConfig()).apply(argv)
    leaves = []

    for path, value, section, section_class in ConfigCli.detailed_leaves(config):
        leaves.append({
            "path"          : path,
            "value"         : ConfigValueRenderer.render(value),
            "type"          : "none" if value is None else type(value).__name__,
            "editable"      : ConfigCli.is_editable(value),
            "choices"       : [entry.value for entry in type(value)] if isinstance(value, Enum) else None,
            "section"       : section,
            "section_class" : section_class,
        })

    return {"key": KEY, "leaves": leaves}


def _builder_names(argv: list[str], tmp_path: Path) -> dict:
    payload_path = tmp_path / f"{KEY}.json"
    payload_path.write_text(json.dumps(_payload(argv)))

    result = subprocess.run(["node", str(HARNESS), str(payload_path)], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _planner_names(launcher, argv: list[str], mode: str) -> list[str]:
    scheduler = launcher.build_scheduler([*argv, "--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]"])

    return [run_name for run_name, _ in scheduler.planner().plan()]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the experiment-builder script")
def test_preview_names_match_the_planner_names(dataset, tmp_path):
    argv      = _sweep_argv(dataset)
    previewed = _builder_names(argv, tmp_path)
    launcher  = BackboneTrainingLauncher(entry_script=Path("main/training/train_backbone.py"))

    for mode in TrainScheduler.MODE_SUBDIRS:
        assert mode in previewed, f"the experiment builder offers no '{mode}' mode"

        planned = _planner_names(launcher, argv, mode)
        shown   = previewed[mode]

        assert shown, f"{mode}: the experiment builder previews no run name at all"
        assert len(shown) == len(planned), f"{mode}: previews {len(shown)} run names, plans {len(planned)}"

        for preview, run_name in zip(shown, planned):
            assert preview.endswith(f"_{run_name}"), f"{mode}: preview '{preview}' does not end in the planned run name '{run_name}'"
