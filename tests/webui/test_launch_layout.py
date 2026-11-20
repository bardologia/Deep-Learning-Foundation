
from __future__ import annotations

import ast
import re

from importlib import import_module

import pytest

from launch_layout          import LaunchLayout, LayoutError
from project_paths          import ProjectPaths
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver

from tools.runtime.config_cli import ConfigCli

from configuration.benchmark.general        import BenchmarkConfig
from configuration.comparison               import ComparisonEntryConfig
from configuration.cross_validation.general import CrossValidationConfig
from configuration.diagnostics              import ReportCollectionEntryConfig, ReportMailEntryConfig, TensorboardExportEntryConfig
from configuration.inference                import BackboneInferenceEntryConfig
from configuration.training                 import BackboneEntryConfig
from configuration.tuning.general           import TuningEntryConfig
from pipelines.backbone.training.loss_terms import LossComponentCatalog
from tests.webui.conftest import WEBUI_ROOT

_TRAINING_PAGES = [
    ("train_backbone", BackboneEntryConfig),
    ("benchmark",      BenchmarkConfig),
    ("cross_validate", CrossValidationConfig),
    ("tune",           TuningEntryConfig),
]


_INFERENCE_PAGES = [
    ("infer_backbone", BackboneInferenceEntryConfig),
]


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_training_layout_claims_every_config_field_exactly_once(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


_WIDGET_REFERENCES = ("baseFrom", "splitFrom", "markerFrom", "datasetFrom")


def _widget_references(layout: dict) -> list[tuple[str, str, str]]:
    found = []
    for entry in layout.get("essentials", []):
        if not isinstance(entry, dict):
            continue

        widget = entry.get("widget") or {}
        for name in _WIDGET_REFERENCES:
            if widget.get(name):
                found.append((entry["path"], name, widget[name]))

    return found


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES + _INFERENCE_PAGES)
def test_every_widget_reference_names_a_real_field(key, flow_config):
    paths  = {path for path, _value in ConfigCli._leaves(flow_config())}
    layout = getattr(LaunchLayout, f"{key.upper()}_LAYOUT", None) or LaunchLayout.LAYOUTS[key]

    for field, reference, target in _widget_references(layout):
        assert target in paths, f"{key}: {field} reads {reference} from '{target}', which is not a field of {flow_config.__name__}"


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_secondary_labels_pick_secondaries_from_the_page_dataset(key, flow_config):
    paths = {path for path, _value in ConfigCli._leaves(flow_config())}
    if "paths.secondary_labels" not in paths:
        return

    layout = LaunchLayout().build(key, [{"path": path} for path in paths])
    widget = layout["widgets"]["paths.secondary_labels"]
    assert widget["mode"] == "secondaries"
    assert widget["multi"] is True
    assert widget["datasetFrom"] in paths


@pytest.mark.parametrize("key, flow_config", [("infer_backbone", BackboneInferenceEntryConfig)])
def test_the_target_dataset_pickers_read_the_target_scene(key, flow_config):
    paths   = [path for path, _value in ConfigCli._leaves(flow_config())]
    widgets = LaunchLayout().build(key, [{"path": path} for path in paths])["widgets"]

    assert widgets["inference.target.parameters_path"]["mode"]         == "params"
    assert widgets["inference.target.parameters_path"]["datasetFrom"]  == "inference.target.dataset_path"
    assert widgets["inference.target.secondary_labels"]["mode"]        == "secondaries"
    assert widgets["inference.target.secondary_labels"]["datasetFrom"] == "inference.target.dataset_path"


@pytest.mark.parametrize("key, flow_config", [("train_backbone", BackboneEntryConfig)])
def test_the_training_pages_hide_the_target_dataset_block(key, flow_config):
    paths  = [path for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, [{"path": path} for path in paths])
    hidden = {field for section in layout["sections"] for panel in section["panels"] if panel["kind"] == "hidden" for field in panel["fields"]}

    assert set(LaunchLayout.INFERENCE_TARGET_FIELDS) <= hidden
    assert not any(path.startswith("inference.target.") for path in layout["widgets"])


@pytest.mark.parametrize("key, flow_config", _INFERENCE_PAGES)
def test_inference_layout_claims_every_config_field_exactly_once(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key", sorted(ProjectPaths.SCRIPT_DIRS))
def test_every_script_layout_builds_against_its_entry_config(key):
    entry  = ScriptConfigResolver(ProjectPaths()).entry_config(key)
    config = getattr(import_module(entry["module"]), entry["class"])()
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(config)]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key", sorted(ProjectPaths.SCRIPT_DIRS))
def test_second_valued_fields_render_with_the_seconds_widget(key):
    entry   = ScriptConfigResolver(ProjectPaths()).entry_config(key)
    config  = getattr(import_module(entry["module"]), entry["class"])()
    leaves  = [{"path": path} for path, _value in ConfigCli._leaves(config)]
    layout  = LaunchLayout().build(key, leaves)
    seconds = [leaf["path"] for leaf in leaves if leaf["path"].endswith(("gif_duration", "resources.poll_interval_sec"))]

    for path in seconds:
        assert layout["widgets"][path] == LaunchLayout.NUM_SECONDS, path


def test_sweep_loss_choices_match_the_component_catalog():
    choices = {choice["value"] for choice in LaunchLayout.MULTI_SWEEP_LOSSES["choices"]}

    assert choices == set(LossComponentCatalog.names())


def test_pair_components_in_experiment_builder_match_the_catalog():
    js    = (WEBUI_ROOT / "static" / "js" / "experiment_builder.js").read_text()
    block = js.split("static PAIR_COMPONENTS = [", 1)[1].split("];", 1)[0]
    names = set(re.findall(r'"([a-z0-9_]+)"', block))

    assert names == set(LossComponentCatalog.names())


def test_experiment_panel_claims_every_trial_field_of_its_entry():
    leaves = [path for path, _ in ConfigCli._leaves(BackboneEntryConfig())]
    trials = LaunchLayout.EXPERIMENT_TRIALS

    for path in leaves:
        head = path.split(".")[0]
        if head.endswith("_trials") or path in ("trials_enabled", "trials_mode"):
            assert path in trials, path


def test_follow_infer_map_covers_every_standalone_inference_family():
    js    = (WEBUI_ROOT / "static" / "js" / "launch.js").read_text()
    block = js.split("static FOLLOW_INFER = {", 1)[1].split("};", 1)[0]
    pairs = dict(re.findall(r'([a-z0-9_]+):\s*"([a-z0-9_]+)"', block))

    layouts  = set(LaunchLayout.LAYOUTS)
    expected = {key: key.replace("train_", "infer_") for key in layouts if key.startswith("train_") and key.replace("train_", "infer_") in layouts}

    assert pairs == expected


def test_compare_runs_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ComparisonEntryConfig())]

    LaunchLayout().build("compare_runs", leaves)


def test_export_tensorboard_plots_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(TensorboardExportEntryConfig())]

    LaunchLayout().build("export_tensorboard_plots", leaves)


def test_collect_reports_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ReportCollectionEntryConfig())]

    LaunchLayout().build("collect_reports", leaves)


def test_mail_experiment_reports_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ReportMailEntryConfig())]

    LaunchLayout().build("mail_experiment_reports", leaves)


def test_every_registered_script_is_reachable_from_the_catalog():
    members = {member for group in ScriptCatalog.GROUPS.values() for member, _label in group["members"]}
    pages   = set(ScriptCatalog.META) | members

    assert set(ProjectPaths.SCRIPT_DIRS) == pages
    assert pages <= set(LaunchLayout.LAYOUTS)


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_vram_reservation_gate_present_on_training_pages(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    claims = LaunchLayout()._claims(layout)

    assert "training.reserve_vram"      in claims
    assert "training.vram_keep_free_gb" in claims


def _section(layout, key):
    return next(section for section in layout["sections"] if section["key"] == key)


def test_a_when_condition_with_both_in_and_set_is_rejected():
    engine = LaunchLayout()
    layout = engine._expand("mail_experiment_reports")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ReportMailEntryConfig())]

    _section(layout, "runs")["when"] = {"field": "target", "set": True, "in": ["runs"]}

    with pytest.raises(LayoutError):
        engine._validate("mail_experiment_reports", layout, leaves)


def test_an_arch_panel_reading_an_unknown_model_field_is_rejected():
    engine = LaunchLayout()
    layout = engine._expand("train_backbone")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]

    panel = next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides")
    panel["modelFrom"] = "no_such_field"

    with pytest.raises(LayoutError):
        engine._validate("train_backbone", layout, leaves)


def test_the_backbone_page_carries_an_arch_panel_reading_the_backbone_field():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]
    layout = LaunchLayout().build("train_backbone", leaves)

    panels = [panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides"]

    assert panels == [{"kind": "special", "panel": "arch_overrides", "fields": ["model_overrides"], "modelFrom": "backbone_name"}]


def _arch_panels(key, flow_config) -> list:
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)
    return [panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides"]


def test_the_cross_validation_page_carries_an_arch_panel_reading_the_backbone_field():
    panels = _arch_panels("cross_validate", CrossValidationConfig)

    assert panels == [{"kind": "special", "panel": "arch_overrides", "fields": ["model_overrides"], "modelFrom": "backbone_name"}]


def test_no_layout_constant_is_bound_twice():
    tree  = ast.parse((WEBUI_ROOT / "launch_layout.py").read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LaunchLayout")
    names = [target.id for node in klass.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)]

    duplicates = sorted(name for name in set(names) if names.count(name) > 1)

    assert duplicates == []


def test_loader_worker_fields_keep_their_own_bounds_apart_from_the_tuning_widget():
    leaves  = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]
    train   = LaunchLayout().build("train_backbone", leaves)
    leaves  = [{"path": path} for path, _value in ConfigCli._leaves(TuningEntryConfig())]
    tune    = LaunchLayout().build("tune", leaves)

    for path in ("training.num_workers", "inference.num_workers", "inference.cpu_workers", "inference.gif_workers"):
        assert train["widgets"][path]["max"] == 64
        assert 32 in train["widgets"][path]["presets"]
    assert tune["widgets"]["n_workers"]["max"] == 32
    assert tune["widgets"]["training.num_workers"]["max"] == 64
