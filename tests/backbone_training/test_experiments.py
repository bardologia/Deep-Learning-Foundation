
from __future__ import annotations

import pytest

from configuration.training                  import BackboneEntryConfig
from configuration.training.backbone         import AugmentationTrialsConfig, PatchTrialsConfig
from configuration.training.general.run      import TrainingQueueConfig
from pipelines.backbone.training.experiments import ArchitectureTrialPlanner, AugmentationTrialPlanner, PatchSizeTrialPlanner


def make_patch_planner(training: TrainingQueueConfig | None = None, **overrides) -> PatchSizeTrialPlanner:
    trials = PatchTrialsConfig(**overrides)
    return PatchSizeTrialPlanner(trials, "unet", {"features": [64, 128, 256]}, training or TrainingQueueConfig())


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


def test_architecture_planner_trains_one_run_per_listed_backbone():
    planner = ArchitectureTrialPlanner([{"label": "unet", "backbone": "unet"}, {"label": "wide", "backbone": "local_cnn", "features": [64] * 4}])

    assert planner.plan() == [
        ("arch-unet", {"backbone_name": "unet"}),
        ("arch-wide", {"backbone_name": "local_cnn", "model_overrides": {"features": [64] * 4}}),
    ]
    assert planner.summary() == {"Architectures": ["unet:unet", "wide:local_cnn"], "Total runs": 2}


def test_architecture_default_backbones_are_registered():
    from models import BACKBONE_CONFIG_REGISTRY

    for trial in BackboneEntryConfig().architecture_trials:
        assert trial["backbone"] in BACKBONE_CONFIG_REGISTRY


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

