
from __future__ import annotations

import pytest

from configuration.training                  import BackboneEntryConfig
from configuration.training.backbone         import AugmentationTrialsConfig, PatchTrialsConfig, ReachTrialsConfig
from configuration.training.general.run      import TrainingQueueConfig
from configuration.training.general.trials   import TrialCatalog
from pipelines.backbone.training.experiments import (
    AugmentationTrialPlanner,
    ContextTrialPlanner,
    PatchSizeTrialPlanner,
    ReachTrialPlanner,
)


def make_patch_planner(training: TrainingQueueConfig | None = None, **overrides) -> PatchSizeTrialPlanner:
    trials = PatchTrialsConfig(**overrides)
    return PatchSizeTrialPlanner(trials, "unet", "conv", {"features": [64, 128, 256]}, training or TrainingQueueConfig())


def test_patch_planner_derives_the_step_from_the_backbone_pyramid():
    planner = make_patch_planner(maximum=(16, 16))

    assert planner.patch_step()  == 8
    assert planner.patch_sizes() == ([8, 16], [8, 16])


def test_patch_planner_explicit_step_overrides_the_architecture():
    planner = make_patch_planner(step=32, maximum=(96, 64))

    assert planner.patch_sizes() == ([32, 64, 96], [32, 64])


def test_patch_planner_emits_shape_stride_and_budget():
    planner  = make_patch_planner(maximum=(16, 16))
    training = TrainingQueueConfig()

    plans = planner.plan()
    assert [name for name, _ in plans] == ["p-008x008", "p-008x016", "p-016x008", "p-016x016"]

    name, overrides = plans[0]
    budget          = training.batch_size * training.patch_size[0] * training.patch_size[1]

    assert overrides["training.patch_size"]              == (8, 8)
    assert overrides["training.patch_stride"]            == (4, 4)
    assert overrides["training.batch_size"]              == budget // (8 * 8)

    reference = training.lr_reference_batch_size * training.patch_size[0] * training.patch_size[1]
    assert overrides["training.lr_reference_batch_size"] == reference // (8 * 8)


def test_patch_planner_keeps_the_lr_scale_constant_under_the_budget():
    planner    = make_patch_planner(maximum=(128, 64))
    training   = TrainingQueueConfig()
    configured = training.batch_size / training.lr_reference_batch_size

    for _name, overrides in planner.plan():
        assert overrides["training.batch_size"] / overrides["training.lr_reference_batch_size"] == pytest.approx(configured)


def test_patch_planner_keeps_the_lr_scale_constant_with_a_distinct_reference_batch():
    training = TrainingQueueConfig(batch_size=128, lr_reference_batch_size=256)
    planner  = make_patch_planner(training, maximum=(128, 64))

    for _name, overrides in planner.plan():
        assert overrides["training.batch_size"] * 2 == overrides["training.lr_reference_batch_size"]


def test_patch_planner_leaves_the_batch_alone_without_the_budget():
    planner = make_patch_planner(maximum=(16, 16), constant_pixel_budget=False)

    for _name, overrides in planner.plan():
        assert "training.batch_size"              not in overrides
        assert "training.lr_reference_batch_size" not in overrides


def test_patch_planner_summary_names_the_per_trial_batches():
    planner = make_patch_planner(maximum=(16, 16))
    summary = planner.summary()
    batches = ", ".join(f"{name} {overrides['training.batch_size']} / {overrides['training.lr_reference_batch_size']}" for name, overrides in planner.plan())

    assert summary["Batch size / LR reference"] == batches
    assert summary["Batch size / LR reference"].startswith("p-008x008 ")


def test_patch_planner_summary_reports_one_batch_without_the_budget():
    training = TrainingQueueConfig(batch_size=32, lr_reference_batch_size=64)
    planner  = make_patch_planner(training, maximum=(16, 16), constant_pixel_budget=False)

    assert planner.summary()["Batch size / LR reference"] == "32 / 64 in every trial (constant pixel budget off)"


REGISTRY_NAMES = ("pixel_mlp", "local_cnn", "unet", "resunet")


def test_context_planner_default_walks_the_context_ladder():
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
    trials = [trial for trial in TrialCatalog.context_trials() if trial["backbone"] == "local_cnn"]

    fields = [1 + sum(2 * (kernel - 1) for kernel in trial["block_kernels"]) for trial in trials]
    depths = [len(trial["features"]) for trial in trials]
    labels = [trial["label"] for trial in trials]

    assert fields == [1, 9, 29, 41]
    assert depths == [10] * 4
    assert labels == [f"cnn{field:02d}" for field in fields]


def test_context_mlp_rung_mirrors_the_kernel_one_cnn_stack():
    trials = TrialCatalog.context_trials()
    mlp    = next(trial for trial in trials if trial["backbone"] == "pixel_mlp")
    cnn01  = next(trial for trial in trials if trial["label"] == "cnn01")

    assert mlp["label"] == "mlp"
    assert mlp["features"] == [cnn01["features"][0]] * 20
    assert "block_kernels" not in mlp


def test_context_default_rungs_are_size_matched_at_constant_depth():
    import torch.nn as nn

    from models import get_backbone

    unet, _ = get_backbone("unet", in_channels=4, out_channels=2)
    target  = sum(p.numel() for p in unet.parameters())

    for trial in TrialCatalog.context_trials():
        overrides = {key: value for key, value in trial.items() if key not in ("label", "backbone")}
        model, _  = get_backbone(trial["backbone"], in_channels=4, out_channels=2, **overrides)

        params = sum(p.numel() for p in model.parameters())
        convs  = sum(1 for m in model.trunk.modules() if isinstance(m, nn.Conv2d))

        assert abs(params - target) / target < 0.005, trial["label"]
        assert convs == 20, trial["label"]


def test_context_default_backbones_are_registered():
    from models import BACKBONE_CONFIG_REGISTRY

    ContextTrialPlanner(TrialCatalog.context_trials(), tuple(BACKBONE_CONFIG_REGISTRY))


def _reach_config(**overrides) -> ReachTrialsConfig:
    rungs = [
        {"label" : "a", "backbone" : "local_cnn", "features" : [8] * 2},
        {"label" : "b", "backbone" : "local_cnn", "features" : [8] * 2},
    ]

    return ReachTrialsConfig(rungs=overrides.pop("rungs", rungs), **overrides)


def test_reach_planner_pins_every_arm_to_the_same_patch():
    planner = ReachTrialPlanner(_reach_config(), REGISTRY_NAMES, "conv", 2)

    plans = planner.plan()

    assert plans == [
        ("reach-a", {"backbone_name": "local_cnn", "training.patch_size": (32, 32), "training.patch_stride": (16, 16), "model_overrides": {"features": [8] * 2}}),
        ("reach-b", {"backbone_name": "local_cnn", "training.patch_size": (32, 32), "training.patch_stride": (16, 16), "model_overrides": {"features": [8] * 2}}),
    ]


def test_reach_default_arms_are_size_matched_and_reach_the_whole_patch():
    from models import BACKBONE_CONFIG_REGISTRY

    trials  = ReachTrialsConfig()
    planner = ReachTrialPlanner(trials, tuple(BACKBONE_CONFIG_REGISTRY), "conv", 2)

    counts    = planner.parameter_counts
    deviation = (max(counts.values()) - min(counts.values())) / min(counts.values())

    assert set(counts) == {"cnn33", "unet"}
    assert deviation <= trials.match_tolerance

    local = next(rung for rung in trials.rungs if rung["backbone"] == "local_cnn")
    field = 1 + 4 * len(local["features"])

    assert field == 33
    assert all(field >= size for size in trials.patch_size)


def test_reach_planner_counts_the_arms_at_the_dataset_target_channel_count():
    rungs = [
        {"label" : "a", "backbone" : "local_cnn", "features" : [8] * 2},
        {"label" : "b", "backbone" : "local_cnn", "features" : [8] * 2},
    ]

    two_targets  = ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "conv", 2)
    five_targets = ReachTrialPlanner(_reach_config(rungs=rungs), REGISTRY_NAMES, "conv", 5)

    assert five_targets.parameter_counts["a"] > two_targets.parameter_counts["a"]


def test_augmentation_planner_default_walks_the_ladder():
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
    planner = AugmentationTrialPlanner(AugmentationTrialsConfig(flip_probability=0.3, noise_probability=0.8, noise_stds=[0.05]))

    plans = dict(planner.plan())

    assert list(plans) == ["aug-0-none", "aug-1-hflip", "aug-2-vflip", "aug-3-noise0.05"]
    assert plans["aug-1-hflip"]["augmentation.p_flip_h"] == 0.3
    assert plans["aug-2-vflip"]["augmentation.p_flip_v"] == 0.3
    assert plans["aug-3-noise0.05"]["augmentation.p_noise"] == 0.8
    assert plans["aug-3-noise0.05"]["augmentation.noise_std"] == 0.05


def test_augmentation_planner_paths_are_entry_config_leaves():
    config = BackboneEntryConfig()

    for _, overrides in AugmentationTrialPlanner(AugmentationTrialsConfig()).plan():
        for path, value in overrides.items():
            section, leaf = path.split(".")
            assert hasattr(getattr(config, section), leaf), path

