"""Tests that every launch page layout matches the config it is supposed to expose.

Each entry point config is flattened into leaves and fed to LaunchLayout, which
refuses any field it does not claim exactly once. The remaining tests pin the legacy
mode preset, the arch and model-card panels, the value gates on sections, widgets and
heads, and the agreement between layout constants and the frontend JavaScript.
"""

from __future__ import annotations

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
from pipelines.backbone.training.launcher            import TrainScheduler
from pipelines.jepa.training.coupling                import TargetProvider
from pipelines.jepa.training.experiments             import CouplingTrialPlanner, CurveTrialPlanner, JepaCurriculumTrialPlanner, MaskingTrialPlanner
from pipelines.jepa.training.launcher                import JepaTrialScheduler
from pipelines.profile_autoencoder.training.launcher import ProfileAeTrialScheduler
from models.dual.dual_resunet               import DualResUNet
from pipelines.backbone.training.loss_terms import LOSS_TERMS, LossComponentCatalog

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

    layout["widgets"]["heads"]["choiceGate"] = {"when": {"field": "no_such_field", "set": True}, "only": ["conv"], "hint": "x"}

    with pytest.raises(LayoutError):
        engine._validate("tune", layout, leaves)
