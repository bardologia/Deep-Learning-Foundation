"""Tests that every launch page layout matches the config it is supposed to expose.

Each entry point config is flattened into leaves and fed to LaunchLayout, which
refuses any field it does not claim exactly once. The remaining tests pin the legacy
mode preset, the arch and model-card panels, the value gates on sections, widgets and
heads, and the agreement between layout constants and the frontend JavaScript.
"""

from __future__ import annotations

import ast
import re

from importlib import import_module

import pytest

from backbone_model_library import BackboneModelLibrary
from launch_layout          import LaunchLayout, LayoutError
from project_paths          import ProjectPaths
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver

from tools.runtime.config_cli import ConfigCli

from configuration.benchmark.general        import BenchmarkConfig
from configuration.comparison               import ComparisonEntryConfig
from configuration.cross_validation.general import CrossValidationConfig
from configuration.diagnostics              import ReportCollectionEntryConfig, ReportMailEntryConfig, TensorboardExportEntryConfig
from configuration.inference                import BackboneInferenceEntryConfig, DualInferenceEntryConfig, ImageAeInferenceEntryConfig, ProfileAeInferenceEntryConfig
from configuration.training                 import BackboneEntryConfig, DualEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig
from configuration.tuning.general           import TuningEntryConfig
from models.backbone                        import BACKBONE_MODEL_REGISTRY
from pipelines.jepa.training.coupling                import TargetProvider
from pipelines.jepa.training.experiments             import CouplingTrialPlanner, CurveTrialPlanner, JepaCurriculumTrialPlanner, MaskingTrialPlanner
from pipelines.jepa.training.launcher                import JepaTrialScheduler
from pipelines.profile_autoencoder.training.launcher import ProfileAeTrialScheduler
from models.dual.dual_resunet               import DualResUNet
from pipelines.backbone.training.loss_terms import LossComponentCatalog
from tests.webui.conftest import WEBUI_ROOT

_TRAINING_PAGES = [
    ("train_backbone",            BackboneEntryConfig),
    ("train_jepa",                JepaEntryConfig),
    ("train_profile_autoencoder", ProfileAeEntryConfig),
    ("train_image_autoencoder",   ImageAeEntryConfig),
    ("train_dual",                DualEntryConfig),
    ("benchmark",                 BenchmarkConfig),
    ("cross_validate",            CrossValidationConfig),
    ("tune",                      TuningEntryConfig),
]


_INFERENCE_PAGES = [
    ("infer_backbone",            BackboneInferenceEntryConfig),
    ("infer_profile_autoencoder", ProfileAeInferenceEntryConfig),
    ("infer_image_autoencoder",   ImageAeInferenceEntryConfig),
    ("infer_dual",                DualInferenceEntryConfig),
]


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_training_layout_claims_every_config_field_exactly_once(key, flow_config):
    """Every training page layout claims each field of its entry config exactly once."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


_WIDGET_REFERENCES = ("baseFrom", "splitFrom", "markerFrom", "datasetFrom")


def _widget_references(layout: dict) -> list[tuple[str, str, str]]:
    """Returns every (field path, reference name, referenced path) a layout's widgets declare.

    Args:
        layout : Expanded layout of one script page.

    Returns:
        One triple per cross-reference found in an essentials or panel widget.
    """
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
    """A picker that reads its base, split or marker from another field must name a field that exists."""
    paths  = {path for path, _value in ConfigCli._leaves(flow_config())}
    layout = getattr(LaunchLayout, f"{key.upper()}_LAYOUT", None) or LaunchLayout.LAYOUTS[key]

    for field, reference, target in _widget_references(layout):
        assert target in paths, f"{key}: {field} reads {reference} from '{target}', which is not a field of {flow_config.__name__}"


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_secondary_labels_pick_secondaries_from_the_page_dataset(key, flow_config):
    """Every page exposing secondary_labels renders it as the secondaries picker fed by a dataset path the page carries."""
    paths = {path for path, _value in ConfigCli._leaves(flow_config())}
    if "paths.secondary_labels" not in paths:
        return

    layout = LaunchLayout().build(key, [{"path": path} for path in paths])
    widget = layout["widgets"]["paths.secondary_labels"]
    assert widget["mode"] == "secondaries"
    assert widget["multi"] is True
    assert widget["datasetFrom"] in paths


@pytest.mark.parametrize("key, flow_config", [("infer_backbone", BackboneInferenceEntryConfig), ("infer_dual", DualInferenceEntryConfig)])
def test_the_target_dataset_pickers_read_the_target_scene(key, flow_config):
    """The label-run and secondary-channel pickers of the target block list the target dataset, not a dataset of the run."""
    paths   = [path for path, _value in ConfigCli._leaves(flow_config())]
    widgets = LaunchLayout().build(key, [{"path": path} for path in paths])["widgets"]

    assert widgets["inference.target.parameters_path"]["mode"]         == "params"
    assert widgets["inference.target.parameters_path"]["datasetFrom"]  == "inference.target.dataset_path"
    assert widgets["inference.target.secondary_labels"]["mode"]        == "secondaries"
    assert widgets["inference.target.secondary_labels"]["datasetFrom"] == "inference.target.dataset_path"


@pytest.mark.parametrize("key, flow_config", [("train_backbone", BackboneEntryConfig), ("train_jepa", JepaEntryConfig), ("train_dual", DualEntryConfig)])
def test_the_training_pages_hide_the_target_dataset_block(key, flow_config):
    """The inference run after training scores the run's own split, so the training pages keep the target block hidden."""
    paths  = [path for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, [{"path": path} for path in paths])
    hidden = {field for section in layout["sections"] for panel in section["panels"] if panel["kind"] == "hidden" for field in panel["fields"]}

    assert set(LaunchLayout.INFERENCE_TARGET_FIELDS) <= hidden
    assert not any(path.startswith("inference.target.") for path in layout["widgets"])


@pytest.mark.parametrize("key, flow_config", _INFERENCE_PAGES)
def test_inference_layout_claims_every_config_field_exactly_once(key, flow_config):
    """Every inference page layout claims each field of its entry config exactly once."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key", sorted(ProjectPaths.SCRIPT_DIRS))
def test_every_script_layout_builds_against_its_entry_config(key):
    """Every registered script except the dispatch-only ones builds a layout against its resolved entry config."""
    entry  = ScriptConfigResolver(ProjectPaths()).entry_config(key)
    config = getattr(import_module(entry["module"]), entry["class"])()
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(config)]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key", sorted(ProjectPaths.SCRIPT_DIRS))
def test_second_valued_fields_render_with_the_seconds_widget(key):
    """Every GIF duration and resource poll interval renders with the seconds widget, so the None-default duration is a number field and not free text."""
    entry   = ScriptConfigResolver(ProjectPaths()).entry_config(key)
    config  = getattr(import_module(entry["module"]), entry["class"])()
    leaves  = [{"path": path} for path, _value in ConfigCli._leaves(config)]
    layout  = LaunchLayout().build(key, leaves)
    seconds = [leaf["path"] for leaf in leaves if leaf["path"].endswith(("gif_duration", "resources.poll_interval_sec"))]

    for path in seconds:
        assert layout["widgets"][path] == LaunchLayout.NUM_SECONDS, path


def test_sweep_loss_choices_match_the_component_catalog():
    """The multi-sweep loss choices match the loss component catalog names."""
    choices = {choice["value"] for choice in LaunchLayout.MULTI_SWEEP_LOSSES["choices"]}

    assert choices == set(LossComponentCatalog.names())
















def test_dual_trunk_choices_match_the_backbone_registry():
    """The dual trunk options list exactly the registered backbone models."""
    assert set(LaunchLayout.CH_TRUNK["options"]) == set(BACKBONE_MODEL_REGISTRY)


def test_pair_components_in_experiment_builder_match_the_catalog():
    """The pair components hard-coded in the experiment builder script match the loss component catalog."""
    js    = (WEBUI_ROOT / "static" / "js" / "experiment_builder.js").read_text()
    block = js.split("static PAIR_COMPONENTS = [", 1)[1].split("];", 1)[0]
    names = set(re.findall(r'"([a-z0-9_]+)"', block))

    assert names == set(LossComponentCatalog.names())


def _builder_js() -> str:
    return (WEBUI_ROOT / "static" / "js" / "experiment_builder.js").read_text()


def _js_string_list(name: str) -> list[str]:
    block = re.split(rf"static\s+{name}\s+=\s+\[", _builder_js(), maxsplit=1)[1].split("];", 1)[0]
    return re.findall(r'"([a-z0-9_]+)"', block)


def _js_mode_keys() -> set:
    block = _builder_js().split("static MODES = [", 1)[1].split("\n  ];", 1)[0]
    return set(re.findall(r'key:\s*"([a-z0-9_]+)"', block))


@pytest.mark.parametrize(("catalog", "expected"), (
    ("CURVE_KINDS",        CurveTrialPlanner.KINDS),
    ("MASK_STRATEGIES",    MaskingTrialPlanner.STRATEGIES),
    ("MASK_FILLS",         MaskingTrialPlanner.FILLS),
    ("MASK_LOSS_TARGETS",  MaskingTrialPlanner.LOSS_ON),
    ("CURRICULUM_RESETS",  tuple(JepaCurriculumTrialPlanner.RESETS)),
    ("COUPLING_MODES",     CouplingTrialPlanner.MODES),
    ("COUPLING_PROVIDERS", TargetProvider.VALID),
))
def test_experiment_builder_catalogs_match_the_planners(catalog, expected):
    """Each option list hard-coded in the experiment builder matches the planner that validates it."""
    assert _js_string_list(catalog) == list(expected)


@pytest.mark.parametrize(("scheduler", "trials"), (
    (ProfileAeTrialScheduler, LaunchLayout.PROFILE_AE_EXPERIMENT_TRIALS),
    (JepaTrialScheduler,      LaunchLayout.JEPA_EXPERIMENT_TRIALS),
))
def test_experiment_builder_offers_every_scheduler_mode(scheduler, trials):
    """Every trial mode the scheduler runs has a matching mode entry in the experiment builder."""
    missing = sorted(mode for mode in scheduler.MODE_SUBDIRS if mode not in _js_mode_keys())

    assert missing == []
    assert "trials_mode" in trials


@pytest.mark.parametrize(("entry_config", "trials"), (
    (ProfileAeEntryConfig, LaunchLayout.PROFILE_AE_EXPERIMENT_TRIALS),
    (JepaEntryConfig,      LaunchLayout.JEPA_EXPERIMENT_TRIALS),
    (BackboneEntryConfig,  LaunchLayout.EXPERIMENT_TRIALS),
    (DualEntryConfig,      LaunchLayout.DUAL_EXPERIMENT_TRIALS),
))
def test_experiment_panel_claims_every_trial_field_of_its_entry(entry_config, trials):
    """The experiment-builder field list of each trials page covers that entry's trial settings."""
    leaves = [path for path, _ in ConfigCli._leaves(entry_config())]
    hidden = {"ablation_features", "ablation_include_full"}

    for path in leaves:
        head = path.split(".")[0]
        if head in hidden:
            continue
        if head.endswith("_trials") or path in ("trials_enabled", "trials_mode"):
            assert path in trials, path


def test_follow_infer_map_covers_every_standalone_inference_family():
    """The frontend follow-inference map pairs each training layout with its matching inference layout."""
    js    = (WEBUI_ROOT / "static" / "js" / "launch.js").read_text()
    block = js.split("static FOLLOW_INFER = {", 1)[1].split("};", 1)[0]
    pairs = dict(re.findall(r'([a-z0-9_]+):\s*"([a-z0-9_]+)"', block))

    layouts  = set(LaunchLayout.LAYOUTS)
    expected = {key: key.replace("train_", "infer_") for key in layouts if key.startswith("train_") and key.replace("train_", "infer_") in layouts}

    assert pairs == expected


def test_compare_runs_layout_claims_every_config_field_exactly_once():
    """The run comparison layout claims each field of its entry config exactly once."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ComparisonEntryConfig())]

    LaunchLayout().build("compare_runs", leaves)


def test_export_tensorboard_plots_layout_claims_every_config_field_exactly_once():
    """The tensorboard export layout claims each field of its entry config exactly once."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(TensorboardExportEntryConfig())]

    LaunchLayout().build("export_tensorboard_plots", leaves)


def test_collect_reports_layout_claims_every_config_field_exactly_once():
    """The report collection layout claims each field of its entry config exactly once."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ReportCollectionEntryConfig())]

    LaunchLayout().build("collect_reports", leaves)


def test_mail_experiment_reports_layout_claims_every_config_field_exactly_once():
    """The report mailing layout claims each field of its entry config exactly once."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ReportMailEntryConfig())]

    LaunchLayout().build("mail_experiment_reports", leaves)


def test_every_registered_script_is_reachable_from_the_catalog():
    """Every registered script except the dispatch-only ones appears in the catalog and has a layout."""
    members = {member for group in ScriptCatalog.GROUPS.values() for member, _label in group["members"]}
    pages   = set(ScriptCatalog.META) | members

    assert set(ProjectPaths.SCRIPT_DIRS) == pages
    assert pages <= set(LaunchLayout.LAYOUTS)


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_vram_reservation_gate_present_on_training_pages(key, flow_config):
    """Every training page exposes the VRAM reservation flag and the free-memory margin."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    claims = LaunchLayout()._claims(layout)

    assert "training.reserve_vram"      in claims
    assert "training.vram_keep_free_gb" in claims


def _section(layout, key):
    """Returns the layout section carrying the given key."""
    return next(section for section in layout["sections"] if section["key"] == key)


def test_jepa_loss_sections_swap_on_the_profile_autoencoder_run():
    """The JEPA page shows the embedding loss section only with a profile autoencoder run and the parameter loss section only without one."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]
    layout = LaunchLayout().build("train_jepa", leaves)

    assert _section(layout, "loss-embedding")["when"] == {"field": "profile_autoencoder_run", "set": True}
    assert _section(layout, "loss-param")["when"]     == {"field": "profile_autoencoder_run", "set": False}


@pytest.mark.parametrize("key, flow_config", [("benchmark", BenchmarkConfig), ("cross_validate", CrossValidationConfig), ("tune", TuningEntryConfig)])
def test_shared_jepa_loss_sections_gate_on_type_and_profile_run(key, flow_config):
    """Sweep pages gate the JEPA loss sections on both the JEPA training type and the presence of a profile autoencoder run."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    assert _section(layout, "jepa-embedding-loss")["when"] == [{"field": "training_type", "in": ["jepa"]}, {"field": "jepa.profile_autoencoder_run", "set": True}]
    assert _section(layout, "jepa-param-loss")["when"]     == [{"field": "training_type", "in": ["jepa"]}, {"field": "jepa.profile_autoencoder_run", "set": False}]


def test_jepa_finetune_rates_are_value_gated_on_the_mode():
    """The JEPA page gates the finetune rates on a selected profile run and on each autoencoder being in finetune mode."""
    engine = LaunchLayout()
    layout = engine._expand("train_jepa")

    conditions = engine._gate_conditions(layout)

    assert {"field": "profile_autoencoder_run", "set": True} in conditions
    assert {"field": "profile_autoencoder_mode", "in": ["finetune"]} in conditions
    assert {"field": "image_autoencoder_mode", "in": ["finetune"]} in conditions


def test_a_when_condition_with_both_in_and_set_is_rejected():
    """A section condition carrying both in and set fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_jepa")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]

    _section(layout, "loss-embedding")["when"] = {"field": "profile_autoencoder_run", "set": True, "in": ["x"]}

    with pytest.raises(LayoutError):
        engine._validate("train_jepa", layout, leaves)


def test_a_value_gate_on_an_unknown_field_is_rejected():
    """A field gate naming a config field that does not exist fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_jepa")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]

    panel = next(panel for panel in _section(layout, "model")["panels"] if panel.get("title") == "Autoencoders")
    gate  = next(entry for group in panel["groups"] for entry in group["fields"] if "gateOn" in entry)
    gate["gateOn"]["field"] = "no_such_field"

    with pytest.raises(LayoutError):
        engine._validate("train_jepa", layout, leaves)


def test_an_essentials_value_gate_on_an_unknown_field_is_rejected():
    """An essentials gate naming a config field that does not exist fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_jepa")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]

    layout["essentials"][0] = {"gateOn": {"field": "no_such_field", "set": True}, "fields": [layout["essentials"][0]]}

    with pytest.raises(LayoutError):
        engine._validate("train_jepa", layout, leaves)


def test_an_essentials_value_gate_needs_exactly_one_of_in_or_set():
    """An essentials gate carrying both in and set fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_jepa")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]

    layout["essentials"][0] = {"gateOn": {"field": "profile_autoencoder_run", "set": True, "in": ["x"]}, "fields": [layout["essentials"][0]]}

    with pytest.raises(LayoutError):
        engine._validate("train_jepa", layout, leaves)


@pytest.mark.parametrize("key, flow_config", [("train_profile_autoencoder", ProfileAeEntryConfig), ("train_image_autoencoder", ImageAeEntryConfig)])
def test_the_ae_pages_carry_an_arch_panel_reading_the_model_field(key, flow_config):
    """Both autoencoder pages expose one architecture override panel driven by the model name field."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    panels = [panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides"]

    assert panels == [{"kind": "special", "panel": "arch_overrides", "fields": ["model_overrides"], "modelFrom": "ae_model_name"}]


def test_an_arch_panel_reading_an_unknown_model_field_is_rejected():
    """An architecture panel reading a model field that does not exist fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_profile_autoencoder")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ProfileAeEntryConfig())]

    panel = next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides")
    panel["modelFrom"] = "no_such_field"

    with pytest.raises(LayoutError):
        engine._validate("train_profile_autoencoder", layout, leaves)


def test_the_backbone_page_carries_an_arch_panel_reading_the_backbone_field():
    """The backbone page exposes one architecture override panel driven by the backbone name field."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]
    layout = LaunchLayout().build("train_backbone", leaves)

    panels = [panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides"]

    assert panels == [{"kind": "special", "panel": "arch_overrides", "fields": ["model_overrides"], "modelFrom": "backbone_name"}]


def _arch_panels(key, flow_config) -> list:
    """Returns the architecture override panels of one built page, in layout order."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)
    return [panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides"]


def test_the_cross_validation_page_switches_its_arch_panel_on_the_training_type():
    """The cross-validation model overrides follow the training type, reading the backbone, JEPA or autoencoder model field."""
    panels   = _arch_panels("cross_validate", CrossValidationConfig)
    by_field = {panel["fields"][0]: panel for panel in panels}

    switch = by_field["model_overrides"]["modelBy"]
    assert switch["field"] == "training_type"
    assert switch["models"] == {"backbone": "backbone_name", "jepa": "backbone_name", "profile_autoencoder": "autoencoder.ae_model_name", "image_autoencoder": "image_autoencoder.ae_model_name"}
    assert "modelFrom" not in by_field["model_overrides"]

    assert by_field["dual.params_overrides"]["modelFrom"]    == "dual.params_backbone"
    assert by_field["dual.existence_overrides"]["modelFrom"] == "dual.existence_backbone"
    assert by_field["dual.params_overrides"]["exclude"] == LaunchLayout.DUAL_TRUNK_EXCLUDE


def test_the_jepa_page_carries_an_arch_panel_reading_the_backbone_field():
    """The JEPA page exposes its model overrides through the architecture panel of the backbone."""
    panels = _arch_panels("train_jepa", JepaEntryConfig)

    assert panels == [{"kind": "special", "panel": "arch_overrides", "fields": ["model_overrides"], "modelFrom": "backbone_name"}]


def test_an_arch_panel_switching_on_an_unknown_field_is_rejected():
    """A model switch naming a field or a model path that does not exist fails validation."""
    engine = LaunchLayout()
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(CrossValidationConfig())]

    for broken in ({"field": "no_such_field", "models": {"backbone": "backbone_name"}}, {"field": "training_type", "models": {"backbone": "no_such_model"}}):
        layout = engine._expand("cross_validate")
        panel  = next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("fields") == ["model_overrides"])
        panel["modelBy"] = broken

        with pytest.raises(LayoutError):
            engine._validate("cross_validate", layout, leaves)


def test_an_arch_panel_with_both_model_sources_is_rejected():
    """A panel declaring both a fixed model field and a switch fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("cross_validate")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(CrossValidationConfig())]

    panel = next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("fields") == ["model_overrides"])
    panel["modelFrom"] = "backbone_name"

    with pytest.raises(LayoutError):
        engine._validate("cross_validate", layout, leaves)


def test_the_dual_page_carries_one_arch_panel_per_trunk():
    """The dual page exposes a titled architecture panel per trunk, each with its own override field, model field and exclusion list."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(DualEntryConfig())]
    layout = LaunchLayout().build("train_dual", leaves)

    panels = [panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides"]

    assert [panel["fields"]    for panel in panels] == [["params_overrides"], ["existence_overrides"]]
    assert [panel["modelFrom"] for panel in panels] == ["params_backbone", "existence_backbone"]
    assert all(panel["exclude"] == LaunchLayout.DUAL_TRUNK_EXCLUDE for panel in panels)
    assert all(panel["title"] for panel in panels)


def test_the_dual_trunk_panels_exclude_the_reserved_and_unused_trunk_fields():
    """The dual trunk exclusion list covers every reserved trunk field plus the per-group learning rate and weight decay patterns."""
    reserved = set(DualResUNet.RESERVED_TRUNK_FIELDS) - set(BackboneModelLibrary.ARCH_EXCLUDED)
    explicit = {name for name in LaunchLayout.DUAL_TRUNK_EXCLUDE if not name.startswith("*")}

    assert reserved <= explicit
    assert "*_lr" in LaunchLayout.DUAL_TRUNK_EXCLUDE
    assert "*_wd" in LaunchLayout.DUAL_TRUNK_EXCLUDE


def test_an_exclude_list_on_a_non_arch_panel_is_rejected():
    """An exclusion list on a panel that is not an architecture panel fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_backbone")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]

    panel = next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "model_card")
    panel["exclude"] = ["dropout"]

    with pytest.raises(LayoutError):
        engine._validate("train_backbone", layout, leaves)


def test_an_empty_exclude_list_is_rejected():
    """An empty exclusion list on an architecture panel fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_dual")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(DualEntryConfig())]

    panel = next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "arch_overrides")
    panel["exclude"] = []

    with pytest.raises(LayoutError):
        engine._validate("train_dual", layout, leaves)


def _model_card(layout):
    """Returns the model card panel of a layout."""
    return next(panel for section in layout["sections"] for panel in section["panels"] if panel.get("panel") == "model_card")


def test_the_jepa_model_card_locks_the_head_to_conv_with_a_profile_ae():
    """The JEPA model card restricts the head to conv, with a hint, whenever a profile autoencoder run is selected."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]
    layout = LaunchLayout().build("train_jepa", leaves)
    gate   = _model_card(layout)["headGate"]

    assert gate["when"] == {"field": "profile_autoencoder_run", "set": True}
    assert gate["only"] == ["conv"]
    assert gate["hint"]


def test_the_cross_validation_model_card_locks_the_head_only_on_the_jepa_type():
    """The cross-validation model card restricts the head to conv only for the JEPA type with a profile autoencoder run."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(CrossValidationConfig())]
    layout = LaunchLayout().build("cross_validate", leaves)
    gate   = _model_card(layout)["headGate"]

    assert gate["when"] == [{"field": "training_type", "in": ["jepa"]}, {"field": "jepa.profile_autoencoder_run", "set": True}]
    assert gate["only"] == ["conv"]


def test_the_cross_validation_tabs_cover_every_training_family():
    """The cross-validation type tabs list every training family, gate the family sections on the selected type and leave the inference section ungated."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(CrossValidationConfig())]
    layout = LaunchLayout().build("cross_validate", leaves)

    assert [option[0] for option in layout["type_tab"]["options"]] == ["backbone", "dual", "profile_autoencoder", "image_autoencoder", "jepa"]

    assert _section(layout, "dual")["when"]              == {"field": "training_type", "in": ["dual"]}
    assert _section(layout, "image-autoencoder")["when"] == {"field": "training_type", "in": ["image_autoencoder"]}
    assert "when" not in _section(layout, "inference")


def test_a_head_gate_on_an_unknown_field_is_rejected():
    """A head gate naming a config field that does not exist fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("train_jepa")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(JepaEntryConfig())]

    _model_card(layout)["headGate"]["when"] = {"field": "no_such_field", "set": True}

    with pytest.raises(LayoutError):
        engine._validate("train_jepa", layout, leaves)


def test_the_benchmark_sweep_axes_only_show_for_the_backbone_type():
    """The benchmark head and loss-component sweep axes are claimed and gated on the backbone training type."""
    engine = LaunchLayout()
    layout = engine._expand("benchmark")

    conditions = engine._gate_conditions(layout)

    assert conditions.count({"field": "training_type", "in": ["backbone"]}) == 2
    for path in ("heads", "sweep_loss_components"):
        assert path in engine._claims(layout)


def test_the_tune_heads_widget_locks_to_conv_with_a_profile_ae():
    """The tuning heads widget restricts the choices to conv, with a hint, for the JEPA type with a profile autoencoder run."""
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(TuningEntryConfig())]
    layout = LaunchLayout().build("tune", leaves)
    gate   = layout["widgets"]["heads"]["choiceGate"]

    assert gate["when"] == [{"field": "training_type", "in": ["jepa"]}, {"field": "jepa.profile_autoencoder_run", "set": True}]
    assert gate["only"] == ["conv"]
    assert gate["hint"]


def test_a_choice_gate_on_an_unknown_field_is_rejected():
    """A widget choice gate naming a config field that does not exist fails validation."""
    engine = LaunchLayout()
    layout = engine._expand("tune")
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(TuningEntryConfig())]

    layout["widgets"]["heads"] = {**layout["widgets"]["heads"], "choiceGate": {"when": {"field": "no_such_field", "set": True}, "only": ["conv"], "hint": "x"}}

    with pytest.raises(LayoutError):
        engine._validate("tune", layout, leaves)


def test_no_layout_constant_is_bound_twice():
    """Every class-level constant of LaunchLayout is assigned once, so no widget silently shadows an earlier one."""
    tree  = ast.parse((WEBUI_ROOT / "launch_layout.py").read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LaunchLayout")
    names = [target.id for node in klass.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)]

    duplicates = sorted(name for name in set(names) if names.count(name) > 1)

    assert duplicates == []


def test_loader_worker_fields_keep_their_own_bounds_apart_from_the_tuning_widget():
    """The loader worker fields render with the 64-worker widget while the tuning n_workers field keeps its 32-worker widget."""
    leaves  = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]
    train   = LaunchLayout().build("train_backbone", leaves)
    leaves  = [{"path": path} for path, _value in ConfigCli._leaves(TuningEntryConfig())]
    tune    = LaunchLayout().build("tune", leaves)

    for path in ("training.num_workers", "inference.num_workers", "inference.cpu_workers", "inference.gif_workers"):
        assert train["widgets"][path]["max"] == 64
        assert 32 in train["widgets"][path]["presets"]
    assert tune["widgets"]["n_workers"]["max"] == 32
    assert tune["widgets"]["training.num_workers"]["max"] == 64
