"""Tests for the JEPA trial planners and their scheduler dispatch.

Covers the planned run names, the emitted override paths, the validation
errors, the entry-config leaf contract and the ConfigCli round trip of every
JEPA trial mode, and checks the planned coupling and curve arms against the
constraints the trainer enforces on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.training.jepa           import AeRunTrialsConfig, CouplingTrialsConfig, CurveTrialsConfig, EmbeddingLossConfig, JepaCurriculumTrialsConfig, JepaEntryConfig, MaskingTrialsConfig
from configuration.training.general.trials import JepaTrialCatalog
from pipelines.jepa.training.coupling      import CouplingMode, TargetProvider
from pipelines.jepa.training.experiments   import AeRunTrialPlanner, CouplingTrialPlanner, CurveTrialPlanner, EmbeddingLossTrialPlanner, JepaCurriculumTrialPlanner, MaskingTrialPlanner
from pipelines.jepa.training.launcher      import JepaTrainingLauncher, JepaTrialScheduler
from pipelines.jepa.training.trainer       import Trainer
from tools.runtime.config_cli              import ConfigCli


ENTRY = Path("main/training/train_jepa.py")


def _leaves() -> set:
    return {path for path, _ in ConfigCli._leaves(JepaEntryConfig())}


def _assert_paths_are_leaves(plans) -> None:
    leaves = _leaves()
    for _, overrides in plans:
        for path in overrides:
            assert path in leaves, path


def test_embedding_planner_disables_every_latent_term_before_each_arm():
    plans = EmbeddingLossTrialPlanner({"hub1": {"use_embedding_huber": True, "weight_embedding_huber": 1.0}}).plan()

    assert plans[0][0] == "el-hub1"
    overrides = plans[0][1]
    assert overrides["embedding_loss.use_embedding_huber"] is True
    assert overrides["embedding_loss.use_embedding_mse"] is False
    assert overrides["embedding_loss.weight_embedding_mse"] == 0.0
    assert "embedding_loss.use_curve_recon" not in overrides
    _assert_paths_are_leaves(plans)


def test_embedding_planner_default_catalog_arms():
    planner = EmbeddingLossTrialPlanner(JepaTrialCatalog.embedding_losses())

    assert [name for name, _ in planner.plan()] == ["el-mse1", "el-sl1", "el-hub1", "el-cha1", "el-mse1-cos0.5", "el-mse1-var0.1", "el-mse1-cov0.05", "el-mse1-var0.1-cov0.05"]
    _assert_paths_are_leaves(planner.plan())


def test_embedding_planner_rejects_curve_fields_and_empty_arms():
    with pytest.raises(ValueError, match="non-latent"):
        EmbeddingLossTrialPlanner({"bad": {"use_curve_recon": True}})

    with pytest.raises(ValueError, match="non-latent"):
        EmbeddingLossTrialPlanner({"delta": {"use_embedding_mse": True, "huber_delta": 2.0}})

    with pytest.raises(ValueError, match="unknown embedding_loss fields"):
        EmbeddingLossTrialPlanner({"nope": {"use_embedding_fancy": True}})

    with pytest.raises(ValueError, match="enables no latent term"):
        EmbeddingLossTrialPlanner({"empty": {"smoothl1_beta": 2.0}})


def test_embedding_planner_accepts_latent_shape_fields():
    plans = EmbeddingLossTrialPlanner({"sl1": {"use_embedding_smoothl1": True, "weight_embedding_smoothl1": 1.0, "smoothl1_beta": 0.5}}).plan()

    assert plans[0][1]["embedding_loss.smoothl1_beta"] == 0.5


def test_curve_planner_default_arms():
    planner = CurveTrialPlanner(CurveTrialsConfig(), "stopgrad")
    plans   = planner.plan()

    names = [name for name, _ in plans]
    assert names == ["curve-off", "curve-mse0.25", "curve-mse1", "curve-huber0.25", "curve-huber1", "curve-charbonnier0.25", "curve-charbonnier1", "curve-sob0.1", "curve-sob1"]
    assert plans[0][1] == {"embedding_loss.use_curve_recon": False, "embedding_loss.use_curve_sobolev": False}
    assert plans[1][1]["embedding_loss.curve_kind"] == "mse"
    assert plans[7][1]["embedding_loss.use_curve_sobolev"] is True
    assert plans[7][1]["embedding_loss.weight_curve_sobolev"] == 0.1
    _assert_paths_are_leaves(plans)


def test_curve_planner_rejects_unknown_kind():
    with pytest.raises(ValueError, match="allowed kinds"):
        CurveTrialPlanner(CurveTrialsConfig(kinds=["cosine"]), "stopgrad")


def test_curve_planner_rejects_the_unanchored_baseline_under_a_live_target():
    with pytest.raises(ValueError, match="target_provider 'live' rejects"):
        CurveTrialPlanner(CurveTrialsConfig(), "live")

    plans = CurveTrialPlanner(CurveTrialsConfig(include_baseline=False), "live").plan()

    assert all(overrides["embedding_loss.use_curve_recon"] for _, overrides in plans)


def test_curve_arms_pass_the_trainer_coupling_validation(autoencoder):
    frozen = CouplingMode("frozen", "profile autoencoder")

    for _, overrides in CurveTrialPlanner(CurveTrialsConfig(), "stopgrad").plan():
        cli    = ConfigCli(JepaEntryConfig(), description="test")
        config = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])

        Trainer.validate_coupling(frozen, "stopgrad", config.embedding_loss, autoencoder)


def test_masking_planner_walks_the_grid():
    planner = MaskingTrialPlanner(MaskingTrialsConfig(), "mlp_ae_a")
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["mask-off", "mask-block0.25", "mask-block0.5", "mask-block0.75", "mask-random0.25", "mask-random0.5", "mask-random0.75"]
    assert plans[0][1] == {"masking.enabled": False}
    assert plans[1][1] == {"masking.enabled": True, "masking.strategy": "block", "masking.mask_ratio": 0.25, "masking.fill": "zeros", "masking.loss_on": "masked"}
    _assert_paths_are_leaves(plans)


def test_masking_planner_rejects_bad_ratio_and_fill():
    with pytest.raises(ValueError, match="lie in"):
        MaskingTrialPlanner(MaskingTrialsConfig(ratios=[1.0]), "mlp_ae_a")

    with pytest.raises(ValueError, match="allowed fills"):
        MaskingTrialPlanner(MaskingTrialsConfig(fill="ones"), "mlp_ae_a")


def test_masking_planner_requires_a_coupled_profile_autoencoder():
    with pytest.raises(ValueError, match="coupled profile autoencoder"):
        MaskingTrialPlanner(MaskingTrialsConfig(), None)

    with pytest.raises(ValueError, match="coupled profile autoencoder"):
        MaskingTrialPlanner(MaskingTrialsConfig(), "")


def test_curriculum_planner_walks_the_grid():
    planner = JepaCurriculumTrialPlanner(JepaCurriculumTrialsConfig(swap_epochs=[10], resets=["none", "all"]), 60)
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["cur-off", "cur-e10-none", "cur-e10-all"]
    assert plans[0][1] == {"curriculum.enabled": False}
    assert plans[1][1] == {"curriculum.enabled": True, "curriculum.swap_epoch": 10, "curriculum.reset_lr": False, "curriculum.reset_warmup": False, "curriculum.reset_optimizer": False}
    assert plans[2][1]["curriculum.reset_optimizer"] is True
    _assert_paths_are_leaves(plans)


def test_curriculum_planner_rejects_unknown_reset():
    with pytest.raises(ValueError, match="allowed resets"):
        JepaCurriculumTrialPlanner(JepaCurriculumTrialsConfig(resets=["scheduler"]), 60)


def test_curriculum_planner_rejects_swap_epochs_the_run_never_reaches():
    with pytest.raises(ValueError, match=r"must be below training.epochs=20"):
        JepaCurriculumTrialPlanner(JepaCurriculumTrialsConfig(swap_epochs=[5, 20, 30]), 20)

    plans = JepaCurriculumTrialPlanner(JepaCurriculumTrialsConfig(swap_epochs=[5]), 20).plan()

    assert [name for name, _ in plans] == ["cur-off", "cur-e5-none", "cur-e5-all"]


def test_coupling_planner_default_arms():
    planner = CouplingTrialPlanner(CouplingTrialsConfig(), TargetProvider.VALID, EmbeddingLossConfig())
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["cpl-stopgrad-frozen", "cpl-stopgrad-ft1e-05", "cpl-stopgrad-ft3e-05", "cpl-stopgrad-ft0.0001"]
    assert plans[0][1] == {"profile_autoencoder_mode": "frozen", "target_provider": "stopgrad"}
    assert plans[1][1] == {"profile_autoencoder_mode": "finetune", "target_provider": "stopgrad", "ae_finetune_lr": 1e-5}
    _assert_paths_are_leaves(plans)


def test_coupling_planner_rejects_unknown_provider():
    with pytest.raises(ValueError, match="allowed providers"):
        CouplingTrialPlanner(CouplingTrialsConfig(target_providers=["ema"]), TargetProvider.VALID, EmbeddingLossConfig())


def test_coupling_planner_skips_the_frozen_cell_of_a_live_target():
    planner = CouplingTrialPlanner(CouplingTrialsConfig(target_providers=["stopgrad", "live"], finetune_lrs=[1e-5]), TargetProvider.VALID, EmbeddingLossConfig())
    plans   = planner.plan()

    assert [name for name, _ in plans] == ["cpl-stopgrad-frozen", "cpl-stopgrad-ft1e-05", "cpl-live-ft1e-05"]
    assert planner.summary()["Total runs"] == len(plans)
    _assert_paths_are_leaves(plans)


def test_coupling_planner_rejects_a_live_target_without_a_finetune_arm():
    with pytest.raises(ValueError, match="add 'finetune'"):
        CouplingTrialPlanner(CouplingTrialsConfig(modes=["frozen"], target_providers=["live"]), TargetProvider.VALID, EmbeddingLossConfig())


def test_coupling_planner_rejects_a_live_target_without_a_curve_anchor():
    with pytest.raises(ValueError, match="use_curve_recon"):
        CouplingTrialPlanner(CouplingTrialsConfig(target_providers=["live"]), TargetProvider.VALID, EmbeddingLossConfig(use_curve_recon=False))

    with pytest.raises(ValueError, match="use_curve_recon"):
        CouplingTrialPlanner(CouplingTrialsConfig(target_providers=["live"]), TargetProvider.VALID, EmbeddingLossConfig(weight_curve_recon=0.0))


def test_coupling_arms_pass_the_trainer_coupling_validation(autoencoder):
    embedding_loss = EmbeddingLossConfig()
    plans          = CouplingTrialPlanner(CouplingTrialsConfig(target_providers=["stopgrad", "live"]), TargetProvider.VALID, embedding_loss).plan()

    for _, overrides in plans:
        mode = CouplingMode(overrides["profile_autoencoder_mode"], "profile autoencoder")

        Trainer.validate_coupling(mode, overrides["target_provider"], embedding_loss, autoencoder)


def test_ae_run_planner_names_every_arm_after_its_run():
    planner = AeRunTrialPlanner(AeRunTrialsConfig(runs=["mlp_ae_a", "tcn_ae_b"]))
    plans   = planner.plan()

    assert plans == [("pae-mlp_ae_a", {"profile_autoencoder_run": "mlp_ae_a"}), ("pae-tcn_ae_b", {"profile_autoencoder_run": "tcn_ae_b"})]
    assert planner.summary()["AE runs"]["pae-tcn_ae_b"] == "tcn_ae_b"
    _assert_paths_are_leaves(plans)


def test_ae_run_planner_names_survive_a_reordering():
    runs      = ["mlp_ae_a", "tcn_ae_b", "gru_ae_c"]
    forward   = dict(AeRunTrialPlanner(AeRunTrialsConfig(runs=runs)).plan())
    backward  = dict(AeRunTrialPlanner(AeRunTrialsConfig(runs=list(reversed(runs)))).plan())

    assert forward == backward


def test_ae_run_planner_sanitizes_nested_run_paths():
    plans = AeRunTrialPlanner(AeRunTrialsConfig(runs=["sweep_ae_run/mlp ae@1.5"])).plan()

    assert plans == [("pae-sweep_ae_run_mlp_ae_1.5", {"profile_autoencoder_run": "sweep_ae_run/mlp ae@1.5"})]


def test_ae_run_planner_rejects_empty_duplicated_and_colliding_runs():
    with pytest.raises(ValueError, match="at least one"):
        AeRunTrialPlanner(AeRunTrialsConfig())

    with pytest.raises(ValueError, match="unique"):
        AeRunTrialPlanner(AeRunTrialsConfig(runs=["a", "a"]))

    with pytest.raises(ValueError, match="once sanitized"):
        AeRunTrialPlanner(AeRunTrialsConfig(runs=["ae/run", "ae run"]))


def test_scheduler_builds_every_local_mode():
    launcher = JepaTrainingLauncher(entry_script=ENTRY)

    for mode in JepaTrialScheduler.MODE_SUBDIRS:
        if mode == "input":
            continue

        args = ["--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]", "--profile_autoencoder_run", "mlp_ae_a"]
        if mode == "ae_run":
            args += ["--ae_run_trials.runs", "['runA']"]

        scheduler = launcher.build_scheduler(args)
        assert isinstance(scheduler, JepaTrialScheduler)
        assert scheduler.sweep_name == f"sweep_jepa_{mode}"
        assert scheduler.planner().plan()


def test_scheduler_builds_the_input_mode_from_the_dataset_channels(tmp_path):
    """Verifies the JEPA input sweep enumerates every secondary channel the dataset records, not the narrower configured selection."""
    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "data" / "dataset.json").write_text(json.dumps({"channel_labels": ["CH00", "CH01", "CH02", "CH03"]}))

    args    = ["--trials_enabled", "true", "--trials_mode", "input", "--seeds", "[0]", "--profile_autoencoder_run", "mlp_ae_a", "--paths.dataset_path", str(dataset), "--paths.secondary_labels", "['CH02']"]
    planner = JepaTrainingLauncher(entry_script=ENTRY).build_scheduler(args).planner()
    plans   = dict(planner.plan())

    assert planner.candidates == ["CH01", "CH02", "CH03"]
    assert plans["in-amp-allsec-noderived-noaux"]["paths.secondary_labels"] == ("CH01", "CH02", "CH03")


def test_scheduler_rejects_unknown_mode():
    launcher = JepaTrainingLauncher(entry_script=ENTRY)

    with pytest.raises(ValueError, match="Unknown trials_mode"):
        launcher.build_scheduler(["--trials_enabled", "true", "--trials_mode", "physics"])


def test_launcher_without_trials_builds_the_seed_scheduler():
    scheduler = JepaTrainingLauncher(entry_script=ENTRY).build_scheduler([])

    assert scheduler.sweep_name == "sweep_seeds_jepa"


def test_launcher_rejects_both_inference_flags():
    launcher = JepaTrainingLauncher(entry_script=ENTRY)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        launcher.build_scheduler(["--infer_after", "true", "--infer_at_end", "true"])


@pytest.mark.parametrize(("argv", "match"), (
    (["--trial", "--ddp", "true"],                  "not available for a --trial unit"),
    (["--trials_enabled", "true", "--ddp", "true"], "fans its units out one GPU each"),
))
def test_launcher_rejects_data_parallel_trials(argv, match):
    launcher = JepaTrainingLauncher(entry_script=ENTRY)

    with pytest.raises(ValueError, match=match):
        launcher.run(argv)


def test_embedding_plan_round_trips_through_config_cli():
    plans = EmbeddingLossTrialPlanner(JepaTrialCatalog.embedding_losses()).plan()
    _, overrides = plans[2]

    cli    = ConfigCli(JepaEntryConfig(), description="test")
    config = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])

    assert config.embedding_loss.use_embedding_huber is True
    assert config.embedding_loss.weight_embedding_huber == 1.0
    assert config.embedding_loss.use_embedding_mse is False


def test_masking_plan_round_trips_through_config_cli():
    plans = MaskingTrialPlanner(MaskingTrialsConfig(), "mlp_ae_a").plan()
    _, overrides = plans[1]

    cli    = ConfigCli(JepaEntryConfig(), description="test")
    config = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])

    assert config.masking.enabled is True
    assert config.masking.strategy == "block"
    assert config.masking.mask_ratio == 0.25


def test_scheduler_fields_cover_every_trials_field():
    trial_fields = [name for name, _ in ConfigCli._leaves(JepaEntryConfig()) if name.split(".")[0].endswith("_trials") or name in ("trials_enabled", "trials_mode")]

    for path in trial_fields:
        assert path.split(".")[0] in JepaTrialScheduler.SCHEDULER_FIELDS, path
