"""Tests for the backbone training trial planners and the ablation catalog.

Covers every planner in `pipelines.backbone.training.experiments`: the run names and
configuration overrides each one emits, the validation errors raised for malformed trial
definitions, and the round trip of the emitted override paths through `ConfigCli` onto a
`BackboneEntryConfig`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.training                  import BackboneEntryConfig, CurriculumInheritance, default_curriculum
from configuration.training.backbone         import AugmentationTrialsConfig, ClampTrialsConfig, HeadMatchingTrialsConfig, LogEnergyTrialsConfig, PairTrialsConfig, PatchTrialsConfig, ReachTrialsConfig, SecondaryTrialsConfig
from configuration.training.general.run      import TrainingQueueConfig
from configuration.training.general.trials   import TrialCatalog
from configuration.training.general.ablation import AblationCatalog
from pipelines.backbone.training.experiments import (
    AblationTrialPlanner,
    ArchitecturePatchStep,
    AugmentationTrialPlanner,
    ClampTrialPlanner,
    ContextTrialPlanner,
    CurriculumTrialPlanner,
    HeadMatchingTrialPlanner,
    InputTrialPlanner,
    LogEnergyTrialPlanner,
    PatchSizeTrialPlanner,
    PairLossTrialPlanner,
    ReachTrialPlanner,
    SecondaryTrialPlanner,
    SlotPresenceTrialPlanner,
    WarmupTrialPlanner,
)
from pipelines.backbone.training.launcher    import TrainScheduler
from tools.runtime.config_cli                import ConfigCli


CANDIDATES = ["CH01", "CH02", "CH03", "CH04", "CH05", "CH06"]


def _dataset(root: Path, channel_labels) -> Path:
    """Writes a dataset root whose data/dataset.json records the given channel labels, reference first, and returns it."""
    (root / "data").mkdir(parents=True)
    (root / "data" / "dataset.json").write_text(json.dumps({"channel_labels": channel_labels}))

    return root


def test_curriculum_planner_crosses_warmup_and_complete():
    """Verifies the curriculum planner emits one run per warmup/complete loss pair."""
    planner = CurriculumTrialPlanner(
        {"w1": {"use_param_l1": True}, "w2": {"use_l1_curve": True}},
        {"c1": {"use_mse_curve": True}},
    )

    plans = planner.plan()

    assert len(plans) == 2
    assert planner.summary() == {"Warmup losses": 2, "Complete losses": 1}

    run_name, overrides = plans[0]
    assert run_name == "w-w1_c-c1"
    assert overrides["curriculum.enabled"]                is True
    assert overrides["curriculum.warmup.use_param_l1"]    is True
    assert overrides["curriculum.complete.use_mse_curve"] is True


def test_warmup_planner_disables_curriculum():
    """Verifies the warmup planner folds each warmup loss into a curriculum-disabled complete stage."""
    planner = WarmupTrialPlanner({"a": {"use_param_l1": True}, "b": {"use_mse_curve": True}})

    plans = planner.plan()

    assert len(plans) == 2
    assert all(overrides["curriculum.enabled"] is False for _, overrides in plans)
    assert plans[0][0] == "nc-a"
    assert plans[0][1]["curriculum.complete.use_param_l1"] is True


def test_presence_planner_disables_curriculum():
    """Verifies the slot-presence planner writes its flags to the complete stage with the curriculum off."""
    planner = SlotPresenceTrialPlanner(
        {"AB": {"use_active_normalization": True, "presence_balance": True}},
    )

    plans = planner.plan()

    assert len(plans) == 1
    assert [name for name, _ in plans] == ["pr-AB"]
    assert planner.summary()["Total runs"] == 1

    ov = dict(plans)["pr-AB"]

    assert ov["curriculum.enabled"] is False
    assert "curriculum.complete.param_match" not in ov
    assert ov["curriculum.complete.use_active_normalization"] is True
    assert ov["curriculum.complete.presence_balance"]         is True


def test_presence_planner_default_matrix():
    """Verifies the catalog presence trials span the four active-normalization and balance combinations."""
    planner = SlotPresenceTrialPlanner(TrialCatalog.presence_trials())

    plans = dict(planner.plan())

    assert list(plans) == ["pr-none", "pr-A", "pr-B", "pr-AB"]

    expected = {
        "pr-none" : (False, False),
        "pr-A"    : (True,  False),
        "pr-B"    : (False, True),
        "pr-AB"   : (True,  True),
    }
    for name, (active_norm, balance) in expected.items():
        assert plans[name]["curriculum.enabled"]                           is False
        assert plans[name]["curriculum.complete.use_active_normalization"] is active_norm
        assert plans[name]["curriculum.complete.presence_balance"]         is balance


def test_presence_planner_rejects_missing_flag():
    """Verifies a presence trial missing one of the two required flags is rejected."""
    with pytest.raises(ValueError, match="must set"):
        SlotPresenceTrialPlanner({"A": {"use_active_normalization": True}})


def test_presence_planner_rejects_unknown_key():
    """Verifies a presence trial carrying a key outside the two presence flags is rejected."""
    with pytest.raises(ValueError, match="unknown keys"):
        SlotPresenceTrialPlanner({"F": {"use_active_normalization": True, "presence_balance": False, "amp_focal_gamma": 2.0}})


def test_presence_planner_rejects_non_boolean_flag():
    """Verifies a presence flag given a non-boolean value is rejected."""
    with pytest.raises(ValueError, match="booleans"):
        SlotPresenceTrialPlanner({"A": {"use_active_normalization": 1.0, "presence_balance": False}})


def test_presence_planner_rejects_empty_trials():
    """Verifies an empty presence trial mapping is rejected."""
    with pytest.raises(ValueError, match="at least one"):
        SlotPresenceTrialPlanner({})


def make_patch_planner(training: TrainingQueueConfig | None = None, **overrides) -> PatchSizeTrialPlanner:
    """Builds a patch planner over the sweep-narrowed three-level unet with config overrides.

    Args:
        training    : Training queue config supplying the reference patch and batch sizes; the entry defaults when omitted.
        **overrides : Fields set on the patch trials config.

    Returns:
        Planner wired to the entry defaults with a step-8 pyramid.
    """
    trials = PatchTrialsConfig(**overrides)
    return PatchSizeTrialPlanner(trials, "unet", "conv", {"features": [64, 128, 256]}, training or TrainingQueueConfig())


def test_patch_planner_derives_the_step_from_the_backbone_pyramid():
    """Verifies a three-level pyramid yields a step of 8 and the default grid spans it."""
    planner = make_patch_planner(maximum=(16, 16))

    assert planner.patch_step()  == 8
    assert planner.patch_sizes() == ([8, 16], [8, 16])


def test_patch_planner_explicit_step_overrides_the_architecture():
    """Verifies an explicit step replaces the architecture-derived one on both axes."""
    planner = make_patch_planner(step=32, maximum=(96, 64))

    assert planner.patch_sizes() == ([32, 64, 96], [32, 64])


def test_patch_planner_emits_shape_stride_and_budget():
    """Verifies each trial pins the patch geometry, its stride, and the pixel-budgeted batch sizes."""
    planner  = make_patch_planner(maximum=(16, 16))
    training = TrainingQueueConfig()

    plans = planner.plan()
    assert [name for name, _ in plans] == ["p-008x008", "p-008x016", "p-016x008", "p-016x016"]

    name, overrides = plans[0]
    budget          = training.batch_size * training.patch_size[0] * training.patch_size[1]

    assert overrides["training.patch_size"]              == (8, 8)
    assert overrides["training.patch_stride"]            == (4, 4)
    assert overrides["training.batch_size"]              == budget // (8 * 8)
    assert overrides["pretrain.find_batch_size"]         is False

    reference = training.lr_reference_batch_size * training.patch_size[0] * training.patch_size[1]
    assert overrides["training.lr_reference_batch_size"] == reference // (8 * 8)


def test_patch_planner_keeps_the_lr_scale_constant_under_the_budget():
    """Verifies batch and learning-rate reference rescale together, holding the scale fixed."""
    planner    = make_patch_planner(maximum=(128, 64))
    training   = TrainingQueueConfig()
    configured = training.batch_size / training.lr_reference_batch_size

    for _name, overrides in planner.plan():
        assert overrides["training.batch_size"] / overrides["training.lr_reference_batch_size"] == pytest.approx(configured)


def test_patch_planner_keeps_the_lr_scale_constant_with_a_distinct_reference_batch():
    """Verifies the batch and its learning-rate reference keep their ratio when the two differ."""
    training = TrainingQueueConfig(batch_size=128, lr_reference_batch_size=256)
    planner  = make_patch_planner(training, maximum=(128, 64))

    for _name, overrides in planner.plan():
        assert overrides["training.batch_size"] * 2 == overrides["training.lr_reference_batch_size"]


def test_patch_planner_rejects_a_patch_that_breaks_the_pixel_budget():
    """Verifies a shape too large to hold one sample under the budget is rejected loudly."""
    training = TrainingQueueConfig(batch_size=2, patch_size=(8, 8))
    planner  = make_patch_planner(training, maximum=(16, 16))

    with pytest.raises(ValueError, match="not a single sample fits"):
        planner.plan()


def test_patch_planner_rejects_a_reference_batch_that_cannot_rescale_in_lockstep():
    """Verifies a learning-rate reference that the shared scale sends off the integers is rejected."""
    training = TrainingQueueConfig(batch_size=128, lr_reference_batch_size=192)
    planner  = make_patch_planner(training, minimum=(24, 8), maximum=(24, 16))

    with pytest.raises(ValueError, match="lr_reference_batch_size"):
        planner.plan()


def test_patch_planner_leaves_the_batch_alone_without_the_budget():
    """Verifies disabling the pixel budget emits no batch or LR-reference overrides."""
    planner = make_patch_planner(maximum=(16, 16), constant_pixel_budget=False)

    for _name, overrides in planner.plan():
        assert "training.batch_size"              not in overrides
        assert "training.lr_reference_batch_size" not in overrides
        assert overrides["pretrain.find_batch_size"] is False


def test_patch_planner_summary_names_the_pinned_probe_and_the_per_trial_batches():
    """Verifies the summary states the pinned batch-size probe and lists each trial's batch size and LR reference under the pixel budget."""
    planner = make_patch_planner(maximum=(16, 16))
    summary = planner.summary()
    batches = ", ".join(f"{name} {overrides['training.batch_size']} / {overrides['training.lr_reference_batch_size']}" for name, overrides in planner.plan())

    assert summary["Batch-size probe"]          == "off in every trial (pretrain.find_batch_size pinned to false)"
    assert summary["Batch size / LR reference"] == batches
    assert summary["Batch size / LR reference"].startswith("p-008x008 ")


def test_patch_planner_summary_reports_one_batch_without_the_budget():
    """Verifies the summary names the single configured batch size and LR reference when the pixel budget is off."""
    training = TrainingQueueConfig(batch_size=32, lr_reference_batch_size=64)
    planner  = make_patch_planner(training, maximum=(16, 16), constant_pixel_budget=False)

    assert planner.summary()["Batch size / LR reference"] == "32 / 64 in every trial (constant pixel budget off)"


def test_patch_planner_rejects_an_inadmissible_minimum():
    """Verifies a minimum that is not a multiple of the step is rejected."""
    planner = make_patch_planner(minimum=(20, 0))

    with pytest.raises(ValueError, match="row=20 is not a multiple"):
        planner.patch_sizes()


def test_patch_planner_rejects_a_single_shape_grid():
    """Verifies a grid holding one patch shape is rejected as nothing to sweep."""
    planner = make_patch_planner(maximum=(8, 8))

    with pytest.raises(ValueError, match="at least 2"):
        planner.patch_sizes()


def test_patch_planner_rejects_bad_stride_ratio():
    """Verifies the patch planner rejects a stride ratio outside the unit interval."""
    with pytest.raises(ValueError, match="stride_ratio"):
        make_patch_planner(stride_ratio=1.5)


def test_patch_step_rejects_backbones_outside_the_verified_unet_family():
    """Verifies a non-UNet backbone cannot derive a patch step and must be given one."""
    with pytest.raises(ValueError, match="patch_trials.step"):
        ArchitecturePatchStep("pixel_mlp", "conv", {}).resolve()


def test_secondary_consecutive_blocks():
    """Verifies the consecutive secondary strategy slides a fixed-width block over the candidate labels."""
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    assert len(plans) == len(CANDIDATES) - 1

    name, overrides = plans[0]
    assert overrides["paths.secondary_labels"] == ("CH01", "CH02")
    assert name.startswith("sec-consecutive-t00")


def test_secondary_spaced_picks_strided_labels():
    """Verifies the spaced secondary strategy picks labels separated by the configured spacing."""
    trials  = SecondaryTrialsConfig(strategy="spaced", n_secondaries=2, block_step=1, spacing=2)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    assert plans[0][1]["paths.secondary_labels"] == ("CH01", "CH03")


def test_secondary_uniform_produces_distinct_trials():
    """Verifies the uniform secondary strategy draws the requested number of distinct label selections."""
    trials  = SecondaryTrialsConfig(strategy="uniform", n_secondaries=2, n_trials=4, seed=0)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    selections = [overrides["paths.secondary_labels"] for _, overrides in plans]

    assert len(plans) == 4
    assert len(set(selections)) == 4


def test_secondary_exhaustive_enumerates_all_combinations_of_exact_size():
    """Verifies the exhaustive strategy enumerates every ordered combination of the requested size."""
    trials  = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=2)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans      = planner.plan()
    selections = [overrides["paths.secondary_labels"] for _, overrides in plans]

    assert len(plans) == 15
    assert len(set(selections)) == len(selections)
    assert all(len(selection) == 2 for selection in selections)
    assert selections[0]  == ("CH01", "CH02")
    assert selections[-1] == ("CH05", "CH06")
    assert planner.summary()["Total runs"] == 15


def test_secondary_exhaustive_full_stack_yields_the_single_full_selection():
    """Verifies the exhaustive strategy collapses to one trial when every candidate is selected."""
    trials  = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=len(CANDIDATES))
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    assert len(plans) == 1
    assert plans[-1][1]["paths.secondary_labels"] == tuple(CANDIDATES)


def test_secondary_exhaustive_pairs_one_fixed_channel_with_every_other_candidate():
    """Verifies a single fixed channel plus one swept secondary yields one run per remaining candidate, each holding both."""
    trials  = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=1, fixed=("CH03",))
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans      = planner.plan()
    selections = [overrides["paths.secondary_labels"] for _, overrides in plans]

    assert len(plans) == len(CANDIDATES) - 1
    assert selections[0] == ("CH01", "CH03")
    assert selections[2] == ("CH03", "CH04")
    assert all("CH03" in selection and len(selection) == 2 for selection in selections)
    assert planner.summary()["Total runs"] == len(CANDIDATES) - 1


def test_secondary_fixed_channels_leave_the_swept_pool():
    """Verifies the swept strategies never draw a fixed channel again, so every trial keeps the requested swept count."""
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1, fixed=("CH01", "CH06"))
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    selections = [overrides["paths.secondary_labels"] for _, overrides in planner.plan()]

    assert planner.pool == ["CH02", "CH03", "CH04", "CH05"]
    assert len(selections) == 3
    assert selections[0] == ("CH01", "CH02", "CH03", "CH06")
    assert all(len(selection) == 4 for selection in selections)


def test_secondary_rejects_a_fixed_channel_outside_the_candidates():
    """Verifies a fixed label the dataset does not carry is refused instead of silently dropped."""
    trials = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=1, fixed=("CH99",))

    with pytest.raises(ValueError, match="CH99"):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_rejects_more_swept_channels_than_the_pool_left_by_the_fixed_ones():
    """Verifies the swept count is bounded by the candidates that remain after the fixed channels are held out."""
    trials = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=len(CANDIDATES), fixed=("CH01",))

    with pytest.raises(ValueError, match="n_secondaries"):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_gaussian_requires_mean_and_sigma():
    """Verifies the gaussian secondary strategy without mean and sigma is rejected."""
    trials = SecondaryTrialsConfig(strategy="gaussian", n_secondaries=2, n_trials=2)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_rejects_unknown_strategy():
    """Verifies an unrecognised secondary selection strategy is rejected."""
    trials = SecondaryTrialsConfig(strategy="bogus", n_secondaries=2)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_rejects_too_many_secondaries():
    """Verifies asking for more secondaries than there are candidates is rejected."""
    trials = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=99)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_rejects_repeated_candidates():
    """Verifies a candidate list naming one channel twice is refused before any subset could pair a channel with itself."""
    trials = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=2)

    with pytest.raises(ValueError, match="CH01.*more than once"):
        SecondaryTrialPlanner(trials, ["CH01", "CH01", "CH02", "CH02", "CH03"])


def test_secondary_exhaustive_never_pairs_a_channel_with_itself():
    """Verifies every exhaustive subset holds distinct channels and the same pair never recurs in another order."""
    trials  = SecondaryTrialsConfig(strategy="exhaustive", n_secondaries=3)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    selections = [overrides["paths.secondary_labels"] for _, overrides in planner.plan()]

    assert all(len(set(selection)) == 3 for selection in selections)
    assert len({frozenset(selection) for selection in selections}) == len(selections) == 20


def test_input_planner_default_catalog_covers_the_stack_grid():
    """Verifies the catalog input trials span the amplitude, derived channel, track-scope and aux field grid."""
    planner = InputTrialPlanner(TrialCatalog.input_trials(), CANDIDATES)

    plans = dict(planner.plan())

    bases = ["amp-allsec-noderived", "noamp-allsec-derived", "amp-allsec-derived", "amp-redsec-derived", "amp-redsec-noderived", "noamp-redsec-derived"]
    assert list(plans) == [f"in-{base}-{aux}" for base in bases for aux in ("noaux", "aux")]

    amp_all = plans["in-amp-allsec-noderived-noaux"]
    assert amp_all["input.use_primary"]        is True
    assert amp_all["input.use_secondaries"]    is True
    assert amp_all["input.use_derived"] is False
    assert amp_all["input.use_aux"]            is False
    assert amp_all["paths.secondary_labels"]   == tuple(CANDIDATES)

    derived_all = plans["in-noamp-allsec-derived-noaux"]
    assert derived_all["input.use_primary"]     is False
    assert derived_all["input.use_secondaries"] is False
    assert derived_all["input.use_derived"]     is True
    assert derived_all["input.use_aux"]         is False
    assert derived_all["paths.secondary_labels"] == tuple(CANDIDATES)

    full_all = plans["in-amp-allsec-derived-aux"]
    assert full_all["input.use_primary"]        is True
    assert full_all["input.use_secondaries"]    is True
    assert full_all["input.use_derived"] is True
    assert full_all["input.use_aux"]            is True
    assert full_all["paths.secondary_labels"]   == tuple(CANDIDATES)

    reduced = plans["in-noamp-redsec-derived-aux"]
    assert reduced["input.use_primary"]        is False
    assert reduced["input.use_secondaries"]    is False
    assert reduced["input.use_derived"] is True
    assert reduced["input.use_aux"]            is True


def test_input_planner_default_catalog_pairs_every_base_with_and_without_aux():
    """Verifies each aux pair differs only in `input.use_aux`."""
    plans = dict(InputTrialPlanner(TrialCatalog.input_trials(), CANDIDATES).plan())

    for base in TrialCatalog.INPUT_TRIAL_BASES:
        noaux = dict(plans[f"in-{base}-noaux"])
        aux   = dict(plans[f"in-{base}-aux"])

        assert noaux.pop("input.use_aux") is False
        assert aux.pop("input.use_aux")   is True
        assert noaux == aux


def test_input_planner_scopes_tracks_per_variant():
    """Verifies only 'all'-scoped input variants pin the secondary label list."""
    trials  = {"a": {"tracks": "all", "use_derived": False}, "b": {"tracks": "reduced", "use_aux": True}}
    planner = InputTrialPlanner(trials, CANDIDATES)

    plans = dict(planner.plan())

    assert set(plans) == {"in-a", "in-b"}
    assert plans["in-a"]["paths.secondary_labels"] == tuple(CANDIDATES)
    assert "paths.secondary_labels" not in plans["in-b"]
    assert "input.tracks"           not in plans["in-a"]
    assert "input.tracks"           not in plans["in-b"]
    assert planner.summary() == {"Input variants": 2, "Tracks": f"1 all ({len(CANDIDATES)} secondaries), 1 reduced (configured selection)"}


def test_input_planner_rejects_unknown_keys():
    """Verifies an input trial carrying a key outside the input config is rejected."""
    with pytest.raises(ValueError):
        InputTrialPlanner({"bad": {"tracks": "all", "use_phase": True}}, CANDIDATES)


def test_input_planner_rejects_missing_or_invalid_track_scope():
    """Verifies an input trial without a valid 'tracks' scope is rejected."""
    with pytest.raises(ValueError):
        InputTrialPlanner({"bad": {"use_primary": True}}, CANDIDATES)

    with pytest.raises(ValueError):
        InputTrialPlanner({"bad": {"tracks": "some", "use_primary": True}}, CANDIDATES)


REGISTRY_NAMES = ("pixel_mlp", "local_cnn", "unet", "resunet")


def test_context_planner_default_walks_the_context_ladder():
    """Verifies the context ladder emits the pixel-MLP rung plus four local_cnn rungs with widening kernels at constant depth."""
    planner = ContextTrialPlanner(TrialCatalog.context_trials(), REGISTRY_NAMES)

    plans = planner.plan()

    assert plans == [
        ("ctx-mlp",   {"backbone_name": "pixel_mlp", "model_overrides": {"features": [1277] * 20}}),
        ("ctx-cnn01", {"backbone_name": "local_cnn", "model_overrides": {"features": [1277] * 10, "block_kernels": [1] * 10}}),
        ("ctx-cnn09", {"backbone_name": "local_cnn", "model_overrides": {"features": [848]  * 10, "block_kernels": [3] * 2 + [1] * 8}}),
        ("ctx-cnn29", {"backbone_name": "local_cnn", "model_overrides": {"features": [502]  * 10, "block_kernels": [3] * 7 + [1] * 3}}),
        ("ctx-cnn41", {"backbone_name": "local_cnn", "model_overrides": {"features": [426]  * 10, "block_kernels": [3] * 10}}),
    ]
    assert planner.summary() == {
        "Rungs"      : ["mlp:pixel_mlp", "cnn01:local_cnn", "cnn09:local_cnn", "cnn29:local_cnn", "cnn41:local_cnn"],
        "Total runs" : 5,
    }


def test_context_ladder_receptive_fields_bracket_the_label_window():
    """Verifies each local_cnn rung's receptive field matches the pixel count in its label."""
    trials = [trial for trial in TrialCatalog.context_trials() if trial["backbone"] == "local_cnn"]

    fields = [1 + sum(2 * (kernel - 1) for kernel in trial["block_kernels"]) for trial in trials]
    depths = [len(trial["features"]) for trial in trials]
    labels = [trial["label"] for trial in trials]

    assert fields == [1, 9, 29, 41]
    assert depths == [10] * 4
    assert labels == [f"cnn{field:02d}" for field in fields]


def test_context_mlp_rung_mirrors_the_kernel_one_cnn_stack():
    """Verifies the pixel-MLP rung carries cnn01's widths at twenty layers and no kernel schedule."""
    trials = TrialCatalog.context_trials()
    mlp    = next(trial for trial in trials if trial["backbone"] == "pixel_mlp")
    cnn01  = next(trial for trial in trials if trial["label"] == "cnn01")

    assert mlp["label"] == "mlp"
    assert mlp["features"] == [cnn01["features"][0]] * 20
    assert "block_kernels" not in mlp


def test_context_default_rungs_are_size_matched_at_constant_depth():
    """Verifies every context rung matches the U-Net parameter count within 0.5% at 20 convolutions."""
    import torch.nn as nn

    from models import get_backbone

    unet, _ = get_backbone("unet", in_channels=9, out_channels=6)
    target  = sum(p.numel() for p in unet.parameters())

    for trial in TrialCatalog.context_trials():
        overrides = {key: value for key, value in trial.items() if key not in ("label", "backbone")}
        model, _  = get_backbone(trial["backbone"], in_channels=9, out_channels=6, **overrides)

        params = sum(p.numel() for p in model.parameters())
        convs  = sum(1 for m in model.trunk.modules() if isinstance(m, nn.Conv2d))

        assert abs(params - target) / target < 0.005, trial["label"]
        assert convs == 20, trial["label"]


def test_context_planner_rejects_a_rung_without_a_label():
    """Verifies a context rung missing its label key is rejected."""
    with pytest.raises(ValueError, match="missing required keys"):
        ContextTrialPlanner([{"backbone": "unet"}], REGISTRY_NAMES)


def test_context_planner_rejects_duplicate_labels():
    """Verifies two context rungs sharing a label are rejected."""
    trials = [{"label": "a", "backbone": "unet"}, {"label": "a", "backbone": "pixel_mlp"}]

    with pytest.raises(ValueError, match="labels must be unique"):
        ContextTrialPlanner(trials, REGISTRY_NAMES)


def test_context_default_backbones_are_registered():
    """Verifies every backbone named by the catalog context trials exists in the backbone registry."""
    from models import BACKBONE_CONFIG_REGISTRY

    ContextTrialPlanner(TrialCatalog.context_trials(), tuple(BACKBONE_CONFIG_REGISTRY))


def test_context_planner_rejects_empty_and_unknown_backbones():
    """Verifies an empty ladder and an unregistered rung backbone are both rejected."""
    with pytest.raises(ValueError, match="at least one rung"):
        ContextTrialPlanner([], REGISTRY_NAMES)

    with pytest.raises(ValueError, match="Unknown context_trials backbones"):
        ContextTrialPlanner([{"label": "a", "backbone": "voxel_gnn"}], REGISTRY_NAMES)


def _reach_config(**overrides) -> ReachTrialsConfig:
    """Returns a reach trials config with two size-matched local_cnn rungs unless overridden."""
    rungs = [
        {"label" : "a", "backbone" : "local_cnn", "features" : [8] * 2},
        {"label" : "b", "backbone" : "local_cnn", "features" : [8] * 2},
    ]

    return ReachTrialsConfig(rungs=overrides.pop("rungs", rungs), **overrides)


def test_reach_planner_pins_every_arm_to_the_same_patch():
    """Verifies every reach arm inherits the same patch size and stride."""
    planner = ReachTrialPlanner(_reach_config(), REGISTRY_NAMES, "conv", 9, 6)

    plans = planner.plan()

    assert plans == [
        ("reach-a", {"backbone_name": "local_cnn", "training.patch_size": (32, 32), "training.patch_stride": (16, 16), "model_overrides": {"features": [8] * 2}}),
        ("reach-b", {"backbone_name": "local_cnn", "training.patch_size": (32, 32), "training.patch_stride": (16, 16), "model_overrides": {"features": [8] * 2}}),
    ]


def test_reach_default_arms_are_size_matched_and_reach_the_whole_patch():
    """Verifies the default reach arms match in parameter count and cover the full patch extent."""
    from models import BACKBONE_CONFIG_REGISTRY

    trials  = ReachTrialsConfig()
    planner = ReachTrialPlanner(trials, tuple(BACKBONE_CONFIG_REGISTRY), "set_pred", 9, 6)

    counts    = planner.parameter_counts
    deviation = (max(counts.values()) - min(counts.values())) / min(counts.values())

    assert set(counts) == {"cnn33", "unet"}
    assert deviation <= trials.match_tolerance

    local = next(rung for rung in trials.rungs if rung["backbone"] == "local_cnn")
    field = 1 + 4 * len(local["features"])

    assert field == 33
    assert all(field >= size for size in trials.patch_size)


def test_reach_planner_counts_the_arms_at_the_configured_gaussian_count():
    """Verifies reach parameter counts grow with the number of output Gaussian channels."""
    rungs = [
        {"label" : "a", "backbone" : "local_cnn", "features" : [8] * 2},
        {"label" : "b", "backbone" : "local_cnn", "features" : [8] * 2},
    ]

    two_slots  = ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "set_pred", 9, 6)
    five_slots = ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "set_pred", 9, 15)

    assert five_slots.parameter_counts["a"] > two_slots.parameter_counts["a"]


def test_reach_planner_rejects_arms_that_are_not_size_matched():
    """Verifies reach arms whose parameter counts differ beyond tolerance are rejected."""
    rungs = [
        {"label" : "narrow", "backbone" : "local_cnn", "features" : [8]  * 2},
        {"label" : "wide",   "backbone" : "local_cnn", "features" : [64] * 2},
    ]

    with pytest.raises(ValueError, match="in parameter count"):
        ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "conv", 9, 6)


def test_reach_planner_rejects_a_lone_arm_and_unknown_backbones():
    """Verifies a single-arm ladder and an unregistered rung backbone are both rejected."""
    with pytest.raises(ValueError, match="at least two architectures"):
        ReachTrialPlanner(_reach_config(rungs=[{"label": "a", "backbone": "unet"}]), REGISTRY_NAMES, "conv", 9, 6)

    rungs = [{"label": "a", "backbone": "voxel_gnn"}, {"label": "b", "backbone": "unet"}]
    with pytest.raises(ValueError, match="Unknown reach_trials backbones"):
        ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "conv", 9, 6)


def test_reach_planner_rejects_duplicate_labels():
    """Verifies two reach rungs sharing a label are rejected."""
    rungs = [{"label": "a", "backbone": "unet"}, {"label": "a", "backbone": "local_cnn"}]

    with pytest.raises(ValueError, match="labels must be unique"):
        ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "conv", 9, 6)


def test_reach_planner_rejects_an_input_stack_mismatch():
    """Verifies a reach input channel count inconsistent with the rung first layers is rejected."""
    with pytest.raises(ValueError, match="count the wrong first layer"):
        ReachTrialPlanner(_reach_config(), REGISTRY_NAMES, "conv", 61, 6)


def test_reach_and_context_planners_reject_bare_string_rungs():
    """Verifies both ladder planners reject the pre-rung format of bare backbone-name strings."""
    with pytest.raises(ValueError, match="must be dicts"):
        ReachTrialPlanner(_reach_config(rungs=["local_cnn", "unet"]), REGISTRY_NAMES, "conv", 9, 6)

    with pytest.raises(ValueError, match="pre-rung format"):
        ContextTrialPlanner(["pixel_mlp", "local_cnn", "unet"], REGISTRY_NAMES)


HEAD_NAMES     = ("conv", "multihead", "per_gaussian", "set_pred")
MATCHING_NAMES = ("hungarian", "sorted_gt")


def test_head_matching_planner_default_crosses_heads_and_matchings():
    """Verifies the head-matching planner crosses every head with every matching on one backbone."""
    planner = HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)

    plans = planner.plan()

    assert plans == [
        ("hm-conv-sorted_gt",     {"backbone_name": "unet", "backbone_head": "conv",     "curriculum.warmup.param_matching": "sorted_gt", "curriculum.complete.param_matching": "sorted_gt"}),
        ("hm-conv-hungarian",     {"backbone_name": "unet", "backbone_head": "conv",     "curriculum.warmup.param_matching": "hungarian", "curriculum.complete.param_matching": "hungarian"}),
        ("hm-set_pred-sorted_gt", {"backbone_name": "unet", "backbone_head": "set_pred", "curriculum.warmup.param_matching": "sorted_gt", "curriculum.complete.param_matching": "sorted_gt"}),
        ("hm-set_pred-hungarian", {"backbone_name": "unet", "backbone_head": "set_pred", "curriculum.warmup.param_matching": "hungarian", "curriculum.complete.param_matching": "hungarian"}),
    ]
    assert planner.summary() == {"Backbone": "unet", "Heads": ["conv", "set_pred"], "Matchings": ["sorted_gt", "hungarian"], "Total runs": 4}


def test_head_matching_planner_rejects_unknown_axes():
    """Verifies an unregistered backbone, head or matching name is rejected."""
    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(), "voxel_gnn", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)

    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(heads=["conv", "cone"]), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)

    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(matchings=["sorted_gt", "greedy"]), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)


def test_head_matching_planner_rejects_empty_and_duplicate_axes():
    """Verifies empty and duplicated head or matching lists are rejected."""
    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(heads=[]), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)

    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(matchings=[]), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)

    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(heads=["conv", "conv"]), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)

    with pytest.raises(ValueError):
        HeadMatchingTrialPlanner(HeadMatchingTrialsConfig(matchings=["hungarian", "hungarian"]), "unet", REGISTRY_NAMES, HEAD_NAMES, MATCHING_NAMES)


def test_augmentation_planner_default_walks_the_ladder():
    """Verifies the augmentation ladder adds hflip, then vflip, then one noise rung per strength."""
    planner = AugmentationTrialPlanner(AugmentationTrialsConfig())

    plans = dict(planner.plan())

    assert list(plans) == ["aug-0-none", "aug-1-hflip", "aug-2-vflip", "aug-3-noise0.01", "aug-3-noise0.03", "aug-3-noise0.1"]
    assert planner.summary()["Total runs"] == 6

    assert plans["aug-0-none"] == {
        "augmentation.p_flip_h" : 0.0,
        "augmentation.p_flip_v" : 0.0,
        "augmentation.p_rot90"  : 0.0,
        "augmentation.p_noise"  : 0.0,
    }
    assert plans["aug-1-hflip"] == {
        "augmentation.p_flip_h" : 0.5,
        "augmentation.p_flip_v" : 0.0,
        "augmentation.p_rot90"  : 0.0,
        "augmentation.p_noise"  : 0.0,
    }
    assert plans["aug-2-vflip"] == {
        "augmentation.p_flip_h" : 0.5,
        "augmentation.p_flip_v" : 0.5,
        "augmentation.p_rot90"  : 0.0,
        "augmentation.p_noise"  : 0.0,
    }
    assert plans["aug-3-noise0.03"] == {
        "augmentation.p_flip_h"  : 0.5,
        "augmentation.p_flip_v"  : 0.5,
        "augmentation.p_rot90"   : 0.0,
        "augmentation.p_noise"   : 0.5,
        "augmentation.noise_std" : 0.03,
    }


def test_augmentation_planner_uses_configured_probabilities():
    """Verifies the configured flip and noise probabilities land on the rungs that use them."""
    planner = AugmentationTrialPlanner(AugmentationTrialsConfig(flip_probability=0.3, noise_probability=0.8, noise_stds=[0.05]))

    plans = dict(planner.plan())

    assert list(plans) == ["aug-0-none", "aug-1-hflip", "aug-2-vflip", "aug-3-noise0.05"]
    assert plans["aug-1-hflip"]["augmentation.p_flip_h"] == 0.3
    assert plans["aug-2-vflip"]["augmentation.p_flip_v"] == 0.3
    assert plans["aug-3-noise0.05"]["augmentation.p_noise"] == 0.8
    assert plans["aug-3-noise0.05"]["augmentation.noise_std"] == 0.05


def test_augmentation_planner_paths_are_entry_config_leaves():
    """Verifies every augmentation override path resolves to a leaf of the backbone entry config."""
    config = BackboneEntryConfig()

    for _, overrides in AugmentationTrialPlanner(AugmentationTrialsConfig()).plan():
        for path, value in overrides.items():
            section, leaf = path.split(".")
            assert hasattr(getattr(config, section), leaf), path


def test_augmentation_planner_rejects_bad_probabilities():
    """Verifies out-of-range flip and noise probabilities are rejected."""
    with pytest.raises(ValueError, match="flip_probability"):
        AugmentationTrialPlanner(AugmentationTrialsConfig(flip_probability=0.0))

    with pytest.raises(ValueError, match="flip_probability"):
        AugmentationTrialPlanner(AugmentationTrialsConfig(flip_probability=1.5))

    with pytest.raises(ValueError, match="noise_probability"):
        AugmentationTrialPlanner(AugmentationTrialsConfig(noise_probability=0.0))


def test_augmentation_planner_rejects_bad_noise_stds():
    """Verifies empty, non-positive and duplicated noise strengths are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        AugmentationTrialPlanner(AugmentationTrialsConfig(noise_stds=[]))

    with pytest.raises(ValueError, match="positive"):
        AugmentationTrialPlanner(AugmentationTrialsConfig(noise_stds=[0.01, 0.0]))

    with pytest.raises(ValueError, match="unique"):
        AugmentationTrialPlanner(AugmentationTrialsConfig(noise_stds=[0.01, 0.01]))


def test_clamp_planner_default_walks_the_ladder():
    """Verifies the clamp ladder plans the disabled, hard and default leaky rungs."""
    planner = ClampTrialPlanner(ClampTrialsConfig())

    plans = dict(planner.plan())

    assert list(plans) == ["clamp-0-none", "clamp-1-hard", "clamp-2-leaky-s0.1"]
    assert planner.summary()["Total runs"] == 3

    assert plans["clamp-0-none"] == {
        "normalization.clamp_output"            : False,
        "normalization.clamp_leaky_slope"       : 0.0,
        "normalization.param_clamp_leaky_slope" : 0.0,
    }
    assert plans["clamp-1-hard"] == {
        "normalization.clamp_output"            : True,
        "normalization.clamp_leaky_slope"       : 0.0,
        "normalization.param_clamp_leaky_slope" : 0.0,
    }
    assert plans["clamp-2-leaky-s0.1"] == {
        "normalization.clamp_output"            : True,
        "normalization.clamp_leaky_slope"       : 0.1,
        "normalization.param_clamp_leaky_slope" : 0.1,
    }


def test_clamp_planner_plans_one_leaky_rung_per_slope_pair():
    """Verifies extra slope pairs each add a leaky rung with their own slopes."""
    planner = ClampTrialPlanner(ClampTrialsConfig(leaky_slopes=[0.1, 0.3], param_leaky_slopes=[0.1, 0.05]))

    plans = dict(planner.plan())

    assert list(plans) == ["clamp-0-none", "clamp-1-hard", "clamp-2-leaky-s0.1", "clamp-2-leaky-s0.3"]
    assert planner.summary()["Total runs"] == 4

    assert plans["clamp-2-leaky-s0.3"] == {
        "normalization.clamp_output"            : True,
        "normalization.clamp_leaky_slope"       : 0.3,
        "normalization.param_clamp_leaky_slope" : 0.05,
    }


def test_clamp_planner_paths_are_entry_config_leaves():
    """Verifies every clamp override path resolves to a leaf of the backbone entry config."""
    config = BackboneEntryConfig()

    for _, overrides in ClampTrialPlanner(ClampTrialsConfig()).plan():
        for path, value in overrides.items():
            section, leaf = path.split(".")
            assert hasattr(getattr(config, section), leaf), path


def test_clamp_planner_rejects_bad_slope_lists():
    """Verifies empty, mismatched, non-positive and duplicated slope lists are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        ClampTrialPlanner(ClampTrialsConfig(leaky_slopes=[], param_leaky_slopes=[]))

    with pytest.raises(ValueError, match="pair up by position"):
        ClampTrialPlanner(ClampTrialsConfig(leaky_slopes=[0.1, 0.3], param_leaky_slopes=[0.1]))

    with pytest.raises(ValueError, match="positive"):
        ClampTrialPlanner(ClampTrialsConfig(leaky_slopes=[0.0], param_leaky_slopes=[0.1]))

    with pytest.raises(ValueError, match="positive"):
        ClampTrialPlanner(ClampTrialsConfig(leaky_slopes=[0.1], param_leaky_slopes=[-0.1]))

    with pytest.raises(ValueError, match="unique"):
        ClampTrialPlanner(ClampTrialsConfig(leaky_slopes=[0.1, 0.1], param_leaky_slopes=[0.1, 0.2]))


def test_input_planner_rejects_empty_trials():
    """Verifies an empty input trial mapping is rejected."""
    with pytest.raises(ValueError):
        InputTrialPlanner({}, CANDIDATES)


























def test_pair_planner_crosses_components_and_weights():
    """Verifies the pair planner emits a curriculum-disabled run per partner term and weight."""
    trials  = PairTrialsConfig(base_component="param_l1", components=["cosine_curve", "moments"], weights=[0.01, 0.05], include_baseline=False)
    planner = PairLossTrialPlanner(trials)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert len(plans) == 4
    assert names == [
        "pair-cosine_curve-w0.01",
        "pair-cosine_curve-w0.05",
        "pair-moments-w0.01",
        "pair-moments-w0.05",
    ]
    assert planner.summary()["Total runs"] == 4
    assert planner.summary()["Base"]       == "param_l1 @ 1"

    overrides = dict(plans)["pair-cosine_curve-w0.05"]
    assert overrides["curriculum.enabled"]                      is False
    assert overrides["curriculum.inherit"]                      is False
    assert overrides["curriculum.complete.use_param_l1"]        is True
    assert overrides["curriculum.complete.weight_param_l1"]     == 1.0
    assert overrides["curriculum.complete.use_cosine_curve"]    is True
    assert overrides["curriculum.complete.weight_cosine_curve"] == 0.05


def test_pair_planner_neutralizes_every_other_term():
    """Verifies only the base and the tested partner term stay enabled in a pair run."""
    trials  = PairTrialsConfig(base_component="param_l1", components=["moments"], weights=[0.1], include_baseline=False)
    planner = PairLossTrialPlanner(trials)

    overrides = planner.plan()[0][1]
    enabled   = [path for path, value in overrides.items() if path.startswith("curriculum.complete.use_") and value is True]

    assert sorted(enabled) == ["curriculum.complete.use_moments", "curriculum.complete.use_param_l1"]
    assert overrides["curriculum.complete.use_cosine_curve"]       is False
    assert overrides["curriculum.complete.weight_cosine_curve"]    == 0.0
    assert overrides["curriculum.complete.use_total_power"]        is False
    assert overrides["curriculum.complete.weight_total_power"]     == 0.0


def test_pair_planner_uses_catalog_flags_for_irregular_names():
    """Verifies a pair component whose catalog flag name differs from its own is mapped correctly."""
    trials  = PairTrialsConfig(base_component="param_l1", components=["total_power_relerr"], weights=[0.1], include_baseline=False)
    planner = PairLossTrialPlanner(trials)

    name, overrides = planner.plan()[0]

    assert name == "pair-total_power_relerr-w0.1"
    assert overrides["curriculum.complete.use_total_power"]    is True
    assert overrides["curriculum.complete.weight_total_power"] == 0.1


def test_pair_planner_prepends_base_only_baseline():
    """Verifies the pair planner prepends a run with only the base term at its base weight."""
    trials  = PairTrialsConfig(base_component="cosine_curve", base_weight=0.5, components=["param_l1"], weights=[0.01], include_baseline=True)
    planner = PairLossTrialPlanner(trials)

    plans = planner.plan()

    assert [name for name, _ in plans] == ["pair-baseline", "pair-param_l1-w0.01"]

    baseline = dict(plans)["pair-baseline"]
    enabled  = [path for path, value in baseline.items() if path.startswith("curriculum.complete.use_") and value is True]

    assert enabled == ["curriculum.complete.use_cosine_curve"]
    assert baseline["curriculum.complete.weight_cosine_curve"] == 0.5


def test_pair_planner_default_config_targets_param_l1_partners():
    """Verifies the default pair sweep pairs param_l1 with nine partner terms plus a baseline."""
    planner = PairLossTrialPlanner(PairTrialsConfig())

    plans = planner.plan()

    assert len(plans) == 10
    assert "pair-baseline" in dict(plans)
    assert "pair-moments-w0.05" in dict(plans)


def test_pair_planner_rejects_unknown_base_component():
    """Verifies an unregistered pair base component is rejected."""
    with pytest.raises(ValueError, match="base_component"):
        PairLossTrialPlanner(PairTrialsConfig(base_component="bogus", components=["cosine_curve"], weights=[0.1]))


def test_pair_planner_rejects_non_positive_base_weight():
    """Verifies a zero or negative pair base weight is rejected."""
    with pytest.raises(ValueError, match="base_weight"):
        PairLossTrialPlanner(PairTrialsConfig(base_weight=0.0, components=["cosine_curve"], weights=[0.1]))


def test_pair_planner_rejects_empty_components():
    """Verifies an empty pair partner list is rejected."""
    with pytest.raises(ValueError, match="at least one"):
        PairLossTrialPlanner(PairTrialsConfig(components=[], weights=[0.1]))


def test_pair_planner_rejects_unknown_component():
    """Verifies an unregistered pair partner component is rejected."""
    with pytest.raises(ValueError, match="bogus"):
        PairLossTrialPlanner(PairTrialsConfig(components=["bogus"], weights=[0.1]))


def test_pair_planner_rejects_base_repeated_as_candidate():
    """Verifies listing the base component among the partners is rejected."""
    with pytest.raises(ValueError, match="repeat"):
        PairLossTrialPlanner(PairTrialsConfig(base_component="param_l1", components=["param_l1", "cosine_curve"], weights=[0.1]))


def test_pair_planner_rejects_duplicate_components():
    """Verifies a repeated pair partner component is rejected."""
    with pytest.raises(ValueError, match="unique"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments", "moments"], weights=[0.1]))


def test_pair_planner_rejects_bad_weights():
    """Verifies empty, non-positive and repeated pair weight lists are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments"], weights=[]))

    with pytest.raises(ValueError, match="positive"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments"], weights=[0.1, -0.1]))

    with pytest.raises(ValueError, match="unique"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments"], weights=[0.1, 0.1]))


def test_pair_plan_round_trips_through_config_cli():
    """Verifies pair overrides applied through ConfigCli land on the single initial curriculum stage."""
    trials  = PairTrialsConfig(base_component="param_l1", components=["cosine_curve"], weights=[0.05], include_baseline=True)
    planner = PairLossTrialPlanner(trials)

    def _apply(overrides: dict):
        cli   = ConfigCli(BackboneEntryConfig())
        trial = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])
        CurriculumInheritance(trial.curriculum, default_curriculum(), cli.overrides).apply()
        return trial

    plans = dict(planner.plan())

    baseline = _apply({**plans["pair-baseline"], "run_name": "pair-baseline", "logdir": "/tmp/pair"})
    assert baseline.curriculum.enabled                           is False
    assert baseline.curriculum.initial_stage.use_param_l1        is True
    assert baseline.curriculum.initial_stage.use_cosine_curve    is False
    assert baseline.curriculum.initial_stage.use_moments         is False

    tested = _apply({**plans["pair-cosine_curve-w0.05"], "run_name": "pair-cosine_curve-w0.05", "logdir": "/tmp/pair"})
    assert tested.curriculum.enabled                            is False
    assert tested.curriculum.initial_stage.use_param_l1         is True
    assert tested.curriculum.initial_stage.weight_param_l1      == 1.0
    assert tested.curriculum.initial_stage.use_cosine_curve     is True
    assert tested.curriculum.initial_stage.weight_cosine_curve  == 0.05
    assert tested.curriculum.initial_stage.use_moments          is False


def test_pair_planner_paths_are_entry_config_leaves():
    """Verifies every pair override path resolves to a leaf of the backbone entry config."""
    planner = PairLossTrialPlanner(PairTrialsConfig())
    leaves  = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    unknown = [path for _, overrides in planner.plan() for path in overrides if path not in leaves]

    assert unknown == []


def test_log_energy_planner_sweeps_weights_with_baseline():
    """Verifies the log-energy planner sets the term in both stages and prepends a zero-weight baseline."""
    trials  = LogEnergyTrialsConfig(weights=[0.05, 0.5], include_baseline=True)
    planner = LogEnergyTrialPlanner(trials)

    plans = planner.plan()

    assert [name for name, _ in plans] == ["logE-baseline", "logE-w0.05", "logE-w0.5"]
    assert planner.summary()["Total runs"] == 3

    tested = dict(plans)["logE-w0.5"]
    assert tested["curriculum.inherit"] is False
    for stage in ("warmup", "complete"):
        assert tested[f"curriculum.{stage}.use_log_energy_curve"]    is True
        assert tested[f"curriculum.{stage}.weight_log_energy_curve"] == 0.5

    baseline = dict(plans)["logE-baseline"]
    for stage in ("warmup", "complete"):
        assert baseline[f"curriculum.{stage}.use_log_energy_curve"]    is False
        assert baseline[f"curriculum.{stage}.weight_log_energy_curve"] == 0.0
    assert "curriculum.complete.use_moments" not in baseline


def test_log_energy_planner_replace_covariance_strips_it_from_every_arm():
    """Verifies replace_covariance disables the covariance term in every stage of every arm."""
    trials  = LogEnergyTrialsConfig(weights=[0.05], replace_covariance=True, include_baseline=True)
    planner = LogEnergyTrialPlanner(trials)

    for _name, overrides in planner.plan():
        for stage in ("warmup", "complete"):
            assert overrides[f"curriculum.{stage}.use_moments"]    is False
            assert overrides[f"curriculum.{stage}.weight_moments"] == 0.0


def test_log_energy_planner_rejects_bad_weights():
    """Verifies empty, non-positive and repeated log-energy weight lists are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        LogEnergyTrialPlanner(LogEnergyTrialsConfig(weights=[]))

    with pytest.raises(ValueError, match="positive"):
        LogEnergyTrialPlanner(LogEnergyTrialsConfig(weights=[0.1, 0.0]))

    with pytest.raises(ValueError, match="unique"):
        LogEnergyTrialPlanner(LogEnergyTrialsConfig(weights=[0.1, 0.1]))


def test_log_energy_plan_round_trips_through_config_cli():
    """Verifies log-energy overrides applied through ConfigCli keep the surrounding full-model terms."""
    trials  = LogEnergyTrialsConfig(weights=[0.25], include_baseline=True)
    planner = LogEnergyTrialPlanner(trials)

    def _apply(overrides: dict):
        cli   = ConfigCli(BackboneEntryConfig())
        trial = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])
        CurriculumInheritance(trial.curriculum, default_curriculum(), cli.overrides).apply()
        return trial

    plans = dict(planner.plan())

    baseline = _apply({**plans["logE-baseline"], "run_name": "logE-baseline", "logdir": "/tmp/logE"})
    assert baseline.curriculum.enabled                       is True
    assert baseline.curriculum.warmup.use_log_energy_curve   is False
    assert baseline.curriculum.complete.use_log_energy_curve is False
    assert baseline.curriculum.complete.use_cosine_curve     is True
    assert baseline.curriculum.complete.weight_cosine_curve  == AblationCatalog.COSINE_WEIGHT

    tested = _apply({**plans["logE-w0.25"], "run_name": "logE-w0.25", "logdir": "/tmp/logE"})
    assert tested.curriculum.warmup.use_log_energy_curve      is True
    assert tested.curriculum.warmup.weight_log_energy_curve   == 0.25
    assert tested.curriculum.complete.use_log_energy_curve    is True
    assert tested.curriculum.complete.weight_log_energy_curve == 0.25


def test_log_energy_planner_paths_are_entry_config_leaves():
    """Verifies every log-energy override path resolves to a leaf of the backbone entry config."""
    planner = LogEnergyTrialPlanner(LogEnergyTrialsConfig(replace_covariance=True))
    leaves  = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    unknown = [path for _, overrides in planner.plan() for path in overrides if path not in leaves]

    assert unknown == []


ABL_FEATURES = [
    {"label": "active_norm", "enable": {"curriculum.warmup.use_active_normalization": True}, "degrade": {"curriculum.warmup.use_active_normalization": False}},
    {"label": "focal",       "enable": {"curriculum.warmup.amp_focal_gamma": 2.0},           "degrade": {"curriculum.warmup.amp_focal_gamma": 0.0}},
    {"label": "balance",     "enable": {"curriculum.warmup.presence_balance": True},         "degrade": {"curriculum.warmup.presence_balance": False}},
]


def test_ablation_planner_cumulative_full_to_baseline():
    """Verifies the ablation planner degrades one feature per rung from full down to baseline."""
    planner = AblationTrialPlanner(ABL_FEATURES, include_full=True)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert names == [
        "abl-0-full",
        "abl-1-no_active_norm",
        "abl-2-no_focal",
        "abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 4

    full = dict(plans)["abl-0-full"]
    assert full["curriculum.warmup.use_active_normalization"] is True
    assert full["curriculum.warmup.amp_focal_gamma"]          == 2.0
    assert full["curriculum.warmup.presence_balance"]         is True

    step1 = dict(plans)["abl-1-no_active_norm"]
    assert step1["curriculum.warmup.use_active_normalization"] is False
    assert step1["curriculum.warmup.amp_focal_gamma"]          == 2.0
    assert step1["curriculum.warmup.presence_balance"]         is True

    baseline = dict(plans)["abl-3-baseline"]
    assert baseline["curriculum.warmup.use_active_normalization"] is False
    assert baseline["curriculum.warmup.amp_focal_gamma"]          == 0.0
    assert baseline["curriculum.warmup.presence_balance"]         is False


def test_ablation_planner_without_full_run():
    """Verifies the full-model rung is omitted when include_full is off."""
    planner = AblationTrialPlanner(ABL_FEATURES, include_full=False)

    plans = planner.plan()

    assert [name for name, _ in plans] == [
        "abl-1-no_active_norm",
        "abl-2-no_focal",
        "abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 3


def test_ablation_planner_rejects_empty_features():
    """Verifies an empty ablation feature list is rejected."""
    with pytest.raises(ValueError):
        AblationTrialPlanner([], include_full=True)


def test_ablation_planner_rejects_feature_without_degrade():
    """Verifies an ablation feature lacking a degrade block is rejected."""
    with pytest.raises(ValueError, match="degrade"):
        AblationTrialPlanner([{"label": "x", "enable": {"curriculum.enabled": True}}], include_full=True)


def test_ablation_planner_rejects_feature_without_enable():
    """Verifies an ablation feature lacking an enable block is rejected."""
    with pytest.raises(ValueError, match="enable"):
        AblationTrialPlanner([{"label": "x", "degrade": {"curriculum.enabled": False}}], include_full=True)


def test_ablation_planner_rejects_duplicate_labels():
    """Verifies two ablation features sharing a label are rejected."""
    features = [
        {"label": "same", "enable": {"curriculum.warmup.use_cosine_curve": True},         "degrade": {"curriculum.warmup.use_cosine_curve": False}},
        {"label": "same", "enable": {"curriculum.warmup.use_active_normalization": True}, "degrade": {"curriculum.warmup.use_active_normalization": False}},
    ]

    with pytest.raises(ValueError, match="unique"):
        AblationTrialPlanner(features, include_full=True)


def test_ablation_planner_allows_complete_terms_with_disabled_curriculum():
    """Verifies complete-stage overrides survive a rung that switches the curriculum off."""
    features = [
        {"label": "curriculum", "enable": {"curriculum.enabled": True},                          "degrade": {"curriculum.enabled": False}},
        {"label": "moments",    "enable": {"curriculum.complete.use_moments": True},             "degrade": {"curriculum.complete.use_moments": False}},
    ]

    planner = AblationTrialPlanner(features, include_full=True)

    names = [name for name, _ in planner.plan()]
    assert names == ["abl-0-full", "abl-1-no_curriculum", "abl-2-baseline"]

    no_curriculum = dict(planner.plan())["abl-1-no_curriculum"]
    assert no_curriculum["curriculum.enabled"]                       is False
    assert no_curriculum["curriculum.complete.use_moments"]          is True


def test_ablation_catalog_default_is_the_standard_set():
    """Verifies the default ablation feature order and that out_mu is excluded."""
    features = AblationCatalog.default_features()
    labels   = [feature["label"] for feature in features]

    assert labels == [
        "loss_curriculum",
        "cosine_curve", "architecture_param_loss", "augmentation",
        "active_norm", "lr_per_group", "lr_warmup",
        "out_sigma", "out_amp", "derived_phase", "chan_mag",
        "output_clamp",
    ]
    assert "out_mu" not in labels
    for feature in features:
        assert set(feature) >= {"label", "enable", "degrade"}


def test_ablation_catalog_standard_categories_present():
    """Verifies the catalog's normalization, loss, architecture and learning-rate features carry the expected enable and degrade payloads."""
    catalog = AblationCatalog.as_dict()

    assert catalog["out_amp"]["degrade"]["normalization.out_amp"]      == "zscore"
    assert catalog["chan_mag"]["degrade"]["normalization.chan_mag"]    == "zscore_log1p"
    assert catalog["derived_phase"]["degrade"]["normalization.derived_phase"]  == "zscore"
    assert catalog["augmentation"]["degrade"]["augmentation.p_flip_h"] == 0.0
    assert "augmentation.p_noise" not in catalog["augmentation"]["enable"]
    assert "out_mu"      not in catalog
    assert "total_power" not in catalog
    assert "moments"     not in catalog

    loss_curriculum = catalog["loss_curriculum"]
    assert loss_curriculum["enable"]["curriculum.enabled"] is True
    assert loss_curriculum["degrade"]                      == {"curriculum.enabled": False}

    cosine = catalog["cosine_curve"]
    assert cosine["enable"]["curriculum.complete.use_cosine_curve"]  is True
    assert cosine["degrade"]["curriculum.complete.use_cosine_curve"] is False
    assert "curriculum.warmup.use_cosine_curve" not in cosine["enable"]

    assert "class_imbalance"  not in catalog
    assert "predict_presence" not in catalog

    architecture = catalog["architecture_param_loss"]
    assert architecture["enable"]["backbone_name"]                         == "resunet"
    assert architecture["enable"]["curriculum.complete.use_param_l1"]      is True
    assert architecture["enable"]["curriculum.complete.use_param_mse"]     is False
    assert architecture["degrade"]["backbone_name"]                        == "unet"
    assert architecture["degrade"]["curriculum.complete.use_param_l1"]     is False
    assert architecture["degrade"]["curriculum.complete.use_param_mse"]    is True
    assert architecture["degrade"]["curriculum.complete.weight_param_mse"] == 1.0

    active_norm = catalog["active_norm"]
    assert active_norm["enable"]["curriculum.complete.use_active_normalization"]  is True
    assert active_norm["degrade"]["curriculum.complete.use_active_normalization"] is False

    assert "warmup_loss" not in catalog
    assert "curriculum"  not in catalog

    lr_warmup = catalog["lr_warmup"]
    assert lr_warmup["enable"]["training.warmup_enabled"]  is True
    assert lr_warmup["degrade"]["training.warmup_enabled"] is False

    lr_per_group = catalog["lr_per_group"]
    assert lr_per_group["enable"]["model_overrides"]["output_head_lr"] == 1e-3
    assert lr_per_group["enable"]["model_overrides"]["encoder_lr"]     == 3e-4
    assert set(lr_per_group["degrade"]["model_overrides"].values())    == {3e-4}
    assert set(lr_per_group["degrade"]["model_overrides"])             == set(lr_per_group["enable"]["model_overrides"])


def test_ablation_catalog_as_dict_covers_loss_terms():
    """Verifies the retired loss ablations are gone and smoothness_tv is still catalogued."""
    catalog = AblationCatalog.as_dict()

    assert "curve_loss_mse_to_l1" not in catalog
    assert "spectral_coherence"   not in catalog
    assert "ssim"                 not in catalog
    assert "smoothness_tv"        in catalog


def test_ablation_catalog_covers_normalization_and_clamp():
    """Verifies the normalization-strategy and output-clamp ablations carry both sides."""
    catalog = AblationCatalog.as_dict()

    norm = catalog["normalization"]
    assert norm["enable"]["normalization.input_strategy"]   == "per_slot"
    assert norm["degrade"]["normalization.input_strategy"]  == "zscore"
    assert norm["degrade"]["normalization.output_strategy"] == "zscore"

    clamp = catalog["output_clamp"]
    assert clamp["enable"]["normalization.clamp_output"]  is True
    assert clamp["degrade"]["normalization.clamp_output"] is False


def test_input_from_dataset_uses_full_stack(tmp_path):
    """Verifies the input planner built from a dataset pins every secondary label the dataset records."""
    planner = InputTrialPlanner.from_dataset(TrialCatalog.input_trials(), _dataset(tmp_path, ["CH00", "S1", "S2", "S3", "S4"]))

    plans = dict(planner.plan())

    assert len(plans) == 12
    assert planner.candidates == ["S1", "S2", "S3", "S4"]

    overrides = plans["in-amp-allsec-noderived-noaux"]
    assert overrides["input.use_derived"] is False
    assert overrides["paths.secondary_labels"] == ("S1", "S2", "S3", "S4")

    assert "paths.secondary_labels" not in plans["in-amp-redsec-derived-aux"]


def test_secondary_from_dataset_loads_candidates(tmp_path):
    """Verifies the secondary planner built from a dataset samples from every secondary label the dataset records."""
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner.from_dataset(trials, _dataset(tmp_path, ["CH00", "S1", "S2", "S3", "S4"]))

    plans = planner.plan()

    assert planner.candidates == ["S1", "S2", "S3", "S4"]
    assert len(plans) >= 1
    assert all("paths.secondary_labels" in overrides for _, overrides in plans)


@pytest.mark.parametrize("planner_class, trials", (
    (InputTrialPlanner,     TrialCatalog.input_trials()),
    (SecondaryTrialPlanner, SecondaryTrialsConfig(strategy="consecutive", n_secondaries=1, block_step=1)),
))
def test_planners_from_dataset_refuse_a_dataset_without_secondaries(tmp_path, planner_class, trials):
    """Verifies both planners refuse a missing layout, a layout without channel labels and a reference-only stack."""
    with pytest.raises(FileNotFoundError, match="dataset.json does not exist"):
        planner_class.from_dataset(trials, tmp_path / "missing")

    with pytest.raises(ValueError, match="records no channel labels"):
        planner_class.from_dataset(trials, _dataset(tmp_path / "unlabelled", None))

    with pytest.raises(ValueError, match="only its reference channel"):
        planner_class.from_dataset(trials, _dataset(tmp_path / "reference", ["CH00"]))


@pytest.mark.parametrize("mode", ("input", "secondary"))
def test_scheduler_planners_take_candidates_from_the_dataset_not_the_configured_selection(tmp_path, mode):
    """Verifies the input and secondary sweeps enumerate every dataset secondary when the configured selection is narrower, so tracks=all differs from tracks=reduced."""
    config                        = BackboneEntryConfig()
    config.logdir                 = tmp_path / "runs"
    config.trials_mode            = mode
    config.paths.dataset_path     = _dataset(tmp_path / "dataset", ["CH00", *CANDIDATES])
    config.paths.secondary_labels = ("CH02",)

    planner = TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py")).planner()

    assert planner.candidates == CANDIDATES

    if mode == "input":
        plans = dict(planner.plan())
        assert plans["in-amp-allsec-noderived-noaux"]["paths.secondary_labels"] == tuple(CANDIDATES)
        assert "paths.secondary_labels" not in plans["in-amp-redsec-noderived-noaux"]


def test_ablation_catalog_paths_are_entry_config_leaves():
    """Verifies every catalog enable and degrade path resolves to a leaf of the backbone entry config."""
    leaves  = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}
    unknown = [
        (feature["label"], side, path)
        for feature in AblationCatalog.features()
        for side in ("enable", "degrade")
        for path in feature.get(side, {})
        if path not in leaves
    ]

    assert unknown == []


def test_ablation_default_plan_round_trips_through_config_cli():
    """Verifies the default ablation plan applies cleanly and its full and baseline rungs differ as designed."""
    config  = BackboneEntryConfig()
    planner = AblationTrialPlanner(config.ablation_features, config.ablation_include_full)

    plans = planner.plan()
    assert len(plans) == len(config.ablation_features) + 1

    def _apply(overrides: dict):
        cli   = ConfigCli(BackboneEntryConfig())
        trial = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])
        CurriculumInheritance(trial.curriculum, default_curriculum(), cli.overrides).apply()
        return trial

    for run_name, overrides in plans:
        trial = _apply({**overrides, "run_name": run_name, "logdir": "/tmp/abl"})
        assert trial.run_name == run_name

    full     = _apply(dict(plans)["abl-0-full"])
    baseline = _apply(dict(plans)[f"abl-{len(config.ablation_features)}-baseline"])
    assert full.curriculum.enabled                               is True
    assert full.backbone_name                                    == "resunet"
    assert full.curriculum.complete.use_param_l1                 is True
    assert full.curriculum.complete.use_cosine_curve             is True
    assert full.curriculum.warmup.use_param_l1                   is True
    assert full.curriculum.warmup.use_cosine_curve               is True
    assert full.curriculum.warmup.presence_balance               is False
    assert full.training.warmup_enabled                          is True
    assert baseline.curriculum.enabled                           is False
    assert baseline.backbone_name                                == "unet"
    assert baseline.curriculum.complete.use_param_mse            is True
    assert baseline.curriculum.complete.use_param_l1             is False
    assert baseline.curriculum.complete.use_active_normalization is False
    assert baseline.curriculum.warmup.use_param_mse              is True
    assert set(baseline.model_overrides.values())                == {3e-4}
    assert baseline.training.warmup_enabled                      is False
    assert baseline.normalization.clamp_output                   is False
    assert baseline.normalization.out_amp                        == "zscore"
    assert baseline.normalization.derived_phase                      == "zscore"
