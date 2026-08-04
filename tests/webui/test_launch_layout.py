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
from configuration.diagnostics              import ReportCollectionEntryConfig, TensorboardExportEntryConfig
from configuration.inference                import BackboneInferenceEntryConfig, DualInferenceEntryConfig, ImageAeInferenceEntryConfig, ProfileAeInferenceEntryConfig, UnrolledInferenceEntryConfig
from configuration.param_extraction         import ChannelOrder, InjectExternalParamsEntryConfig
from configuration.patch_sweep.general      import PatchSweepConfig
from configuration.training                 import BackboneEntryConfig, DualEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig, UnrolledEntryConfig
from configuration.tuning.general           import TuningEntryConfig
from models.backbone                        import BACKBONE_MODEL_REGISTRY
from pipelines.backbone.training.loss_terms import LOSS_TERMS, LossComponentCatalog

from tests.webui.conftest import WEBUI_ROOT

_DISPATCH_ONLY = {"generate_tomogram", "generate_interferograms"}

_TRAINING_PAGES = [
    ("train_backbone",            BackboneEntryConfig),
    ("train_jepa",                JepaEntryConfig),
    ("train_profile_autoencoder", ProfileAeEntryConfig),
    ("train_image_autoencoder",   ImageAeEntryConfig),
    ("train_unrolled",            UnrolledEntryConfig),
    ("train_dual",                DualEntryConfig),
    ("benchmark",                 BenchmarkConfig),
    ("cross_validate",            CrossValidationConfig),
    ("sweep_patches",             PatchSweepConfig),
    ("tune",                      TuningEntryConfig),
]


_INFERENCE_PAGES = [
    ("infer_backbone",            BackboneInferenceEntryConfig),
    ("infer_profile_autoencoder", ProfileAeInferenceEntryConfig),
    ("infer_image_autoencoder",   ImageAeInferenceEntryConfig),
    ("infer_unrolled",            UnrolledInferenceEntryConfig),
    ("infer_dual",                DualInferenceEntryConfig),
]


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_training_layout_claims_every_config_field_exactly_once(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key, flow_config", _INFERENCE_PAGES)
def test_inference_layout_claims_every_config_field_exactly_once(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key", sorted(key for key in ProjectPaths.SCRIPT_DIRS if key not in _DISPATCH_ONLY))
def test_every_script_layout_builds_against_its_entry_config(key):
    entry  = ScriptConfigResolver(ProjectPaths()).entry_config(key)
    config = getattr(import_module(entry["module"]), entry["class"])()
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(config)]

    LaunchLayout().build(key, leaves)


def test_sweep_loss_choices_match_the_component_catalog():
    choices = {choice["value"] for choice in LaunchLayout.MULTI_SWEEP_LOSSES["choices"]}

    assert choices == set(LossComponentCatalog.names())


def test_legacy_mode_preset_pins_every_loss_term_flag():
    preset = LaunchLayout.LEGACY_MODE["preset"]

    for term in LOSS_TERMS:
        expected = "True" if term.name == "param_legacy" else "False"
        assert preset[f"curriculum.complete.{term.use_flag}"] == expected

    assert preset["curriculum.complete.param_matching"] == "sorted_gt"
    assert preset["curriculum.enabled"] == "False"


def test_legacy_mode_preset_pins_the_legacy_normalization():
    preset = LaunchLayout.LEGACY_MODE["preset"]

    assert preset["input.use_primary"]                   == "False"
    assert preset["input.use_secondaries"]               == "True"
    assert preset["input.secondaries_representation"]    == "mag_only"
    assert preset["input.use_interferograms"]            == "True"
    assert preset["input.interferograms_representation"] == "angle_only"
    assert preset["input.use_dem"]                       == "False"

    assert preset["normalization.pass_mag"]   == "fixed_log1p"
    assert preset["normalization.pass_phase"] == "fixed_angle_01"
    assert preset["normalization.ifg_mag"]    == "fixed_log1p"
    assert preset["normalization.ifg_phase"]  == "fixed_angle_01"
    assert preset["normalization.out_amp"]    == "fixed_bounds"
    assert preset["normalization.out_mu"]     == "fixed_bounds"
    assert preset["normalization.out_sigma"]  == "fixed_bounds"

    assert preset["normalization.clamp_output"] == "False"
    assert "normalization.clamp_leaky_slope" not in preset

    assert preset["curriculum.complete.amp_zero_thr"] == "0.001"
    assert preset["inference.render_amp_floor"]       == "0.001"


def test_legacy_mode_preset_pins_the_legacy_optimization():
    preset = LaunchLayout.LEGACY_MODE["preset"]

    assert preset["training.warmup_enabled"]      == "False"
    assert preset["training.clip_mode"]           == "disabled"
    assert preset["training.scheduler_type"]      == "constant"
    assert preset["training.scale_lr_with_batch"] == "False"
    assert preset["training.epochs"]              == "300"
    assert preset["training.patch_size"]          == "(64, 64)"
    assert preset["training.patch_stride"]        == "(32, 32)"

    overrides = ast.literal_eval(preset["model_overrides"])
    assert overrides["all_groups_lr"] == 1e-5
    assert overrides["all_groups_wd"] == 0.0


def test_legacy_mode_preset_pins_the_legacy_architecture():
    preset = LaunchLayout.LEGACY_MODE["preset"]

    assert preset["backbone_name"] == "unet"
    assert preset["backbone_head"] == "conv"

    overrides = ast.literal_eval(preset["model_overrides"])
    assert overrides["features"]          == [64, 128, 256, 512]
    assert overrides["bottleneck_factor"] == 2
    assert overrides["dropout"]           == 0.0
    assert overrides["normalization"]     == "none"
    assert overrides["conv_bias"]         is True


def test_legacy_mode_ships_with_the_backbone_training_layout():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]
    layout = LaunchLayout().build("train_backbone", leaves)

    assert layout["legacy"] == LaunchLayout.LEGACY_MODE
    assert "curriculum.complete.use_param_legacy" in layout["legacy"]["expose"]
    assert layout["legacy"]["sections"] == []
    assert layout["legacy"]["preset"]["backbone_name"] == "unet"


def test_legacy_mode_with_unknown_paths_is_rejected():
    engine = LaunchLayout()
    layout = engine._expand("train_backbone")
    layout["legacy"]["preset"]["curriculum.complete.use_param_l2"] = "False"
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(BackboneEntryConfig())]

    with pytest.raises(LayoutError):
        engine._validate("train_backbone", layout, leaves)


def test_dual_trunk_choices_match_the_backbone_registry():
    assert set(LaunchLayout.CH_TRUNK["options"]) == set(BACKBONE_MODEL_REGISTRY)


def test_pair_components_in_experiment_builder_match_the_catalog():
    js    = (WEBUI_ROOT / "static" / "js" / "launch_widgets.js").read_text()
    block = js.split("static PAIR_COMPONENTS = [", 1)[1].split("];", 1)[0]
    names = set(re.findall(r'"([a-z0-9_]+)"', block))

    assert names == set(LossComponentCatalog.names())


def test_follow_infer_map_covers_every_standalone_inference_family():
    js    = (WEBUI_ROOT / "static" / "js" / "launch.js").read_text()
    block = js.split("static FOLLOW_INFER = {", 1)[1].split("};", 1)[0]
    pairs = dict(re.findall(r'([a-z0-9_]+):\s*"([a-z0-9_]+)"', block))

    layouts  = set(LaunchLayout.LAYOUTS)
    expected = {key: key.replace("train_", "infer_") for key in layouts if key.startswith("train_") and key.replace("train_", "infer_") in layouts}

    assert pairs == expected


def test_inject_external_params_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(InjectExternalParamsEntryConfig())]

    LaunchLayout().build("inject_external_params", leaves)


def test_channel_order_choices_match_every_permutation():
    for order in LaunchLayout.CH_CHANNEL_ORDER["options"]:
        ChannelOrder.positions(order)

    assert len(LaunchLayout.CH_CHANNEL_ORDER["options"]) == 6


def test_compare_runs_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ComparisonEntryConfig())]

    LaunchLayout().build("compare_runs", leaves)


def test_export_tensorboard_plots_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(TensorboardExportEntryConfig())]

    LaunchLayout().build("export_tensorboard_plots", leaves)


def test_collect_reports_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ReportCollectionEntryConfig())]

    LaunchLayout().build("collect_reports", leaves)


def test_every_registered_script_is_reachable_from_the_catalog():
    members = {member for group in ScriptCatalog.GROUPS.values() for member, _label in group["members"]}
    pages   = set(ScriptCatalog.META) | members

    assert set(ProjectPaths.SCRIPT_DIRS) - _DISPATCH_ONLY == pages
    assert pages <= set(LaunchLayout.LAYOUTS)


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_vram_reservation_gate_present_on_training_pages(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    claims = LaunchLayout()._claims(layout)

    assert "training.reserve_vram"      in claims
    assert "training.vram_keep_free_gb" in claims
