"""Tests for the profile-autoencoder trial planners and their scheduler dispatch.

Covers the planned run names, the emitted override paths, the validation
errors, the entry-config leaf contract and the ConfigCli round trip of every
profile-autoencoder trial mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.training.profile_autoencoder import CapacityTrialsConfig, EmbeddingDimTrialsConfig, ProfileAeEntryConfig, ProfileAeLossConfig, ProfileAugmentationTrialsConfig, SamplingTrialsConfig, SmoothnessTrialsConfig
from configuration.training.general.trials      import ProfileAeTrialCatalog
from models.profile_autoencoder                 import PROFILE_AE_CONFIG_REGISTRY
from pipelines.profile_autoencoder.training.experiments import ArchTrialPlanner, CapacityTrialPlanner, EmbeddingDimTrialPlanner, ProfileAugmentationTrialPlanner, ProfileLossTrialPlanner, SamplingTrialPlanner, SmoothnessTrialPlanner
from pipelines.profile_autoencoder.training.launcher    import ProfileAeTrainingLauncher, ProfileAeTrialScheduler
from tools.runtime.config_cli                   import ConfigCli


ENTRY = Path("main/training/train_profile_autoencoder.py")


def _leaves() -> set:
    return {path for path, _ in ConfigCli._leaves(ProfileAeEntryConfig())}


def _assert_paths_are_leaves(plans) -> None:
    leaves = _leaves()
    for _, overrides in plans:
        for path in overrides:
            assert path in leaves, path


def test_arch_planner_walks_every_registered_family():
    planner = ArchTrialPlanner(ProfileAeTrialCatalog.arch_trials(), tuple(PROFILE_AE_CONFIG_REGISTRY), {})
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["arch-mlp", "arch-conv1d", "arch-tcn", "arch-gru", "arch-resmlp", "arch-trans1d", "arch-cnn_attn"]
    assert planner.summary()["Total runs"] == 7
    assert plans[0][1] == {"ae_model_name": "mlp_ae", "model_overrides": {}}
    _assert_paths_are_leaves(plans)


def test_arch_planner_forwards_extra_rung_keys_as_model_overrides():
    rungs = [{"label": "wide", "ae_model_name": "mlp_ae", "hidden_dim": 1024}]
    plans = ArchTrialPlanner(rungs, tuple(PROFILE_AE_CONFIG_REGISTRY), {}).plan()

    assert plans == [("arch-wide", {"ae_model_name": "mlp_ae", "model_overrides": {"hidden_dim": 1024}})]


def test_arch_planner_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown autoencoder"):
        ArchTrialPlanner([{"label": "bad", "ae_model_name": "nope_ae"}], tuple(PROFILE_AE_CONFIG_REGISTRY), {})


def test_arch_planner_rejects_entry_level_model_overrides():
    with pytest.raises(ValueError, match="architecture fields of a single family"):
        ArchTrialPlanner(ProfileAeTrialCatalog.arch_trials(), tuple(PROFILE_AE_CONFIG_REGISTRY), {"hidden_dim": 1024})


def test_arch_planner_rejects_duplicate_labels():
    rungs = [{"label": "a", "ae_model_name": "mlp_ae"}, {"label": "a", "ae_model_name": "tcn_ae"}]
    with pytest.raises(ValueError, match="unique"):
        ArchTrialPlanner(rungs, tuple(PROFILE_AE_CONFIG_REGISTRY), {})


def test_embedding_planner_default_walks_the_ladder():
    planner = EmbeddingDimTrialPlanner(EmbeddingDimTrialsConfig(), {})
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["emb-d8", "emb-d16", "emb-d24", "emb-d32", "emb-d48", "emb-d64"]
    assert plans[2][1] == {"model_overrides": {"embedding_dim": 24}}
    _assert_paths_are_leaves(plans)


def test_embedding_planner_rejects_pinned_dimension():
    with pytest.raises(ValueError, match="pins embedding_dim"):
        EmbeddingDimTrialPlanner(EmbeddingDimTrialsConfig(), {"embedding_dim": 24})


def test_embedding_planner_rejects_duplicate_dims():
    with pytest.raises(ValueError, match="unique"):
        EmbeddingDimTrialPlanner(EmbeddingDimTrialsConfig(dims=[16, 16]), {})


def test_capacity_planner_reads_the_family_tunable_choices():
    planner = CapacityTrialPlanner(CapacityTrialsConfig(), "mlp_ae", {}, PROFILE_AE_CONFIG_REGISTRY)
    space   = PROFILE_AE_CONFIG_REGISTRY["mlp_ae"].tunable_arch_params()

    assert planner.widths == space["hidden_dim"]["choices"]
    assert planner.depths == space["depth"]["choices"]
    assert planner.summary()["Total runs"] == len(planner.widths) * len(planner.depths)
    _assert_paths_are_leaves(planner.plan())


def test_capacity_planner_explicit_grid_and_width_field():
    planner = CapacityTrialPlanner(CapacityTrialsConfig(widths=[64, 128], depths=[2]), "tcn_ae", {}, PROFILE_AE_CONFIG_REGISTRY)
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["cap-w64-d2", "cap-w128-d2"]
    assert plans[0][1] == {"model_overrides": {"seq_channels": 64, "depth": 2}}


def test_capacity_planner_rejects_pinned_swept_fields():
    with pytest.raises(ValueError, match="pins"):
        CapacityTrialPlanner(CapacityTrialsConfig(widths=[64], depths=[2]), "mlp_ae", {"hidden_dim": 512}, PROFILE_AE_CONFIG_REGISTRY)


def test_capacity_planner_sweeps_the_attention_width_of_the_cnn_attn_family():
    planner = CapacityTrialPlanner(CapacityTrialsConfig(), "cnn_attn_ae", {}, PROFILE_AE_CONFIG_REGISTRY)
    space   = PROFILE_AE_CONFIG_REGISTRY["cnn_attn_ae"].tunable_arch_params()

    assert planner.width_field == "hidden_dim"
    assert planner.widths      == space["hidden_dim"]["choices"]
    assert planner.plan()[0][1] == {"model_overrides": {"hidden_dim": 128, "depth": 2}}


@pytest.mark.parametrize(("ae_model_name", "list_field"), (("mlp_ae", "hidden_dims"), ("conv1d_ae", "seq_channels_list")))
def test_capacity_planner_rejects_pinned_per_layer_width_lists(ae_model_name, list_field):
    with pytest.raises(ValueError, match=f"pins '{list_field}'"):
        CapacityTrialPlanner(CapacityTrialsConfig(widths=[64, 128], depths=[2]), ae_model_name, {list_field: [64, 64]}, PROFILE_AE_CONFIG_REGISTRY)


def test_capacity_planner_accepts_a_layer_list_field_of_another_family():
    planner = CapacityTrialPlanner(CapacityTrialsConfig(widths=[64], depths=[2]), "tcn_ae", {"seq_kernel_size": 7}, PROFILE_AE_CONFIG_REGISTRY)

    assert planner.layer_list_field is None
    assert planner.plan()[0][1] == {"model_overrides": {"seq_kernel_size": 7, "seq_channels": 64, "depth": 2}}


def test_loss_planner_disables_every_term_before_each_arm():
    plans = ProfileLossTrialPlanner({"l1": {"use_l1_curve": True, "weight_l1_curve": 1.0}}).plan()

    assert plans[0][0] == "loss-l1"
    overrides = plans[0][1]
    assert overrides["ae_loss.use_l1_curve"] is True
    assert overrides["ae_loss.weight_l1_curve"] == 1.0
    assert overrides["ae_loss.use_mse_curve"] is False
    assert overrides["ae_loss.weight_mse_curve"] == 0.0
    assert overrides["ae_loss.use_latent_noise"] is False
    _assert_paths_are_leaves(plans)


def test_loss_planner_default_catalog_arms():
    planner = ProfileLossTrialPlanner(ProfileAeTrialCatalog.loss_trials())

    assert [name for name, _ in planner.plan()] == ["loss-mse", "loss-l1", "loss-huber", "loss-charb", "loss-mse-cos0.5", "loss-mse-lE0.1", "loss-mse-tv0.1"]
    _assert_paths_are_leaves(planner.plan())


def test_loss_planner_rejects_unknown_fields_and_empty_arms():
    with pytest.raises(ValueError, match="unknown ae_loss fields"):
        ProfileLossTrialPlanner({"bad": {"use_fancy_curve": True}})

    with pytest.raises(ValueError, match="enables no loss term"):
        ProfileLossTrialPlanner({"empty": {"huber_delta": 2.0}})


@pytest.mark.parametrize(("use_key", "weight_key"), ProfileLossTrialPlanner.TERMS)
def test_loss_planner_rejects_an_enabled_term_without_its_weight(use_key, weight_key):
    with pytest.raises(ValueError, match=weight_key):
        ProfileLossTrialPlanner({"unweighted": {use_key: True}})


def test_smoothness_planner_default_arms():
    planner = SmoothnessTrialPlanner(SmoothnessTrialsConfig(), ProfileAeLossConfig())
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["smooth-baseline", "smooth-sob0.1", "smooth-sob0.5", "smooth-sob1", "smooth-ln0.02", "smooth-ln0.05", "smooth-ln0.1"]
    assert plans[0][1] == {"ae_loss.use_sobolev": False, "ae_loss.use_latent_noise": False}
    assert plans[1][1] == {"ae_loss.use_sobolev": True, "ae_loss.use_latent_noise": False, "ae_loss.weight_sobolev": 0.1}
    assert plans[4][1] == {"ae_loss.use_sobolev": False, "ae_loss.use_latent_noise": True, "ae_loss.latent_noise_std": 0.02, "ae_loss.weight_latent_noise": 1.0}
    _assert_paths_are_leaves(plans)


def test_smoothness_planner_combined_grid():
    trials = SmoothnessTrialsConfig(sobolev_weights=[0.5], latent_noise_stds=[0.05], combine=True, include_baseline=False)
    plans  = SmoothnessTrialPlanner(trials, ProfileAeLossConfig()).plan()

    assert [name for name, _ in plans] == ["smooth-sob0.5", "smooth-ln0.05", "smooth-sob0.5-ln0.05"]
    assert plans[2][1]["ae_loss.use_sobolev"] is True
    assert plans[2][1]["ae_loss.use_latent_noise"] is True


def test_smoothness_planner_rejects_a_base_loss_carrying_only_the_swept_sobolev_term():
    ae_loss = ProfileAeLossConfig(use_mse_curve=False, use_sobolev=True)

    with pytest.raises(ValueError, match="empty reconstruction loss"):
        SmoothnessTrialPlanner(SmoothnessTrialsConfig(), ae_loss)


def test_smoothness_planner_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        SmoothnessTrialPlanner(SmoothnessTrialsConfig(sobolev_weights=[0.0]), ProfileAeLossConfig())


def test_augmentation_planner_walks_the_ladder():
    planner = ProfileAugmentationTrialPlanner(ProfileAugmentationTrialsConfig())
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["aug-0-none", "aug-1-flip", "aug-2-shift", "aug-3-scale", "aug-4-noise0.005", "aug-4-noise0.01", "aug-4-noise0.03"]
    assert plans[0][1]["profile_augmentation.p_flip"] == 0.0
    assert plans[1][1]["profile_augmentation.p_flip"] == 0.5
    assert plans[1][1]["profile_augmentation.p_shift"] == 0.0
    assert plans[4][1]["profile_augmentation.p_noise"] == 0.5
    assert plans[4][1]["profile_augmentation.noise_std"] == 0.005
    _assert_paths_are_leaves(plans)


def test_augmentation_planner_rejects_out_of_range_probability():
    with pytest.raises(ValueError, match="lie in"):
        ProfileAugmentationTrialPlanner(ProfileAugmentationTrialsConfig(flip_probability=1.5))


def test_sampling_planner_walks_the_grid():
    planner = SamplingTrialPlanner(SamplingTrialsConfig(subsamples=[0.5, 1.0], empty_fracs=[0.0, 0.2]))
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["smp-p0.5-e0", "smp-p0.5-e0.2", "smp-p1-e0", "smp-p1-e0.2"]
    assert plans[0][1] == {"pixel_subsample": 0.5, "keep_empty_frac": 0.0}
    _assert_paths_are_leaves(plans)


def test_sampling_planner_rejects_out_of_range_subsample():
    with pytest.raises(ValueError, match="lie in"):
        SamplingTrialPlanner(SamplingTrialsConfig(subsamples=[0.0, 1.0]))


def test_scheduler_builds_every_mode():
    launcher = ProfileAeTrainingLauncher(entry_script=ENTRY)

    for mode in ProfileAeTrialScheduler.MODE_SUBDIRS:
        scheduler = launcher.build_scheduler(["--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]"])
        assert isinstance(scheduler, ProfileAeTrialScheduler)
        assert scheduler.sweep_name == f"sweep_profile_autoencoder_{mode}"
        assert scheduler.planner().plan()


def test_scheduler_rejects_unknown_mode():
    launcher = ProfileAeTrainingLauncher(entry_script=ENTRY)

    with pytest.raises(ValueError, match="Unknown trials_mode"):
        launcher.build_scheduler(["--trials_enabled", "true", "--trials_mode", "physics"])


def test_launcher_without_trials_builds_the_seed_scheduler():
    scheduler = ProfileAeTrainingLauncher(entry_script=ENTRY).build_scheduler([])

    assert scheduler.sweep_name == "sweep_seeds_profile_autoencoder"


@pytest.mark.parametrize(("argv", "match"), (
    (["--trial", "--ddp", "true"],                     "not available for a --trial unit"),
    (["--trials_enabled", "true", "--ddp", "true"],    "fans its units out one GPU each"),
))
def test_launcher_rejects_data_parallel_trials(argv, match):
    launcher = ProfileAeTrainingLauncher(entry_script=ENTRY)

    with pytest.raises(ValueError, match=match):
        launcher.run(argv)


def test_resume_from_skips_complete_units_and_resumes_the_rest(tmp_path):
    root = tmp_path / "ema"
    root.mkdir()
    (root / "ema-0-off").mkdir()
    (root / "ema-0-off" / "complete.json").write_text(json.dumps({"status": "complete"}))
    (root / "ema-1-raw-d0.999").mkdir()

    launcher  = ProfileAeTrainingLauncher(entry_script=ENTRY)
    scheduler = launcher.build_scheduler(["--trials_enabled", "true", "--trials_mode", "ema", "--seeds", "[0]", "--resume_from", str(root)])

    assert scheduler.runs_root              == root
    assert scheduler.resume_plan.completed  == ["ema-0-off"]
    assert [name for name, _ in scheduler.units()] == ["ema-1-raw-d0.999", "ema-2-warmup-d0.999"]

    argv = scheduler.trial_argv("ema-1-raw-d0.999", {"training.use_ema": True})

    assert "--resume" in argv
    assert "--training.resume" in argv


def test_loss_plan_round_trips_through_config_cli():
    plans = ProfileLossTrialPlanner(ProfileAeTrialCatalog.loss_trials()).plan()
    _, overrides = plans[1]

    cli    = ConfigCli(ProfileAeEntryConfig(), description="test")
    config = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])

    assert config.ae_loss.use_l1_curve is True
    assert config.ae_loss.weight_l1_curve == 1.0
    assert config.ae_loss.use_mse_curve is False


def test_sampling_plan_round_trips_through_config_cli():
    plans = SamplingTrialPlanner(SamplingTrialsConfig()).plan()
    _, overrides = plans[0]

    cli    = ConfigCli(ProfileAeEntryConfig(), description="test")
    config = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])

    assert config.pixel_subsample == overrides["pixel_subsample"]
    assert config.keep_empty_frac == overrides["keep_empty_frac"]


def test_scheduler_fields_cover_every_trials_field():
    trial_fields = [name for name, _ in ConfigCli._leaves(ProfileAeEntryConfig()) if name.split(".")[0].endswith("_trials") or name in ("trials_enabled", "trials_mode")]

    for path in trial_fields:
        assert path.split(".")[0] in ProfileAeTrialScheduler.SCHEDULER_FIELDS, path
