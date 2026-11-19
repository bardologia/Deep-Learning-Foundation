
from __future__ import annotations

from configuration.cross_validation import CrossValidationConfig
from job_describer                  import JobDescriber
from tools.runtime.config_cli        import ConfigCli


GENERATE_DATASET_LEAVES = [
    {"path": "synthesis.output_directory", "value": "None"},
    {"path": "synthesis.global_crop",      "value": "[0, 1512, 0, 1024]"},
    {"path": "synthesis.k_max",            "value": "2"},
    {"path": "synthesis.params_run_name",  "value": "None"},
    {"path": "synthesis.seed",             "value": "0"},
]

BACKBONE_LEAVES = [
    {"path": "backbone_name",          "value": "resunet"},
    {"path": "backbone_head",          "value": "conv"},
    {"path": "trials_enabled",         "value": "False"},
    {"path": "trials_mode",            "value": "curriculum"},
    {"path": "run_name",               "value": "None"},
    {"path": "infer_after",            "value": "False"},
    {"path": "paths.dataset_path",     "value": "/data/base_dataset_w20_10"},
    {"path": "paths.parameters_path",  "value": "/data/base_dataset_w20_10/params/params_Ng3_sigonly_k5.npy"},
    {"path": "training.patch_size",    "value": "(64, 32)"},
    {"path": "training.epochs",        "value": "60"},
]

EXTRACT_LEAVES = [
    {"path": "dataset_filter",    "value": "[]"},
    {"path": "fit_k_values",      "value": "[5]"},
    {"path": "fit_lambda_values", "value": "[0.01]"},
    {"path": "fit_modes",         "value": "['sigma', 'sigma_amp', 'sigma_amp_mu']"},
    {"path": "output_suffix",     "value": "None"},
]


class KnownScriptPaths:

    def __init__(self, known: set[str]) -> None:
        self.known = known

    def has_script(self, key: str) -> bool:
        return key in self.known


class StubResolver:

    def __init__(self, leaves_by_key: dict, ok: bool = True) -> None:
        self.leaves_by_key = leaves_by_key
        self.ok            = ok

    def resolve(self, key: str, interpreter: str) -> dict:
        if not self.ok or key not in self.leaves_by_key:
            return {"ok": False, "error": "unavailable"}
        return {"ok": True, "leaves": self.leaves_by_key[key]}


def _describer(leaves_by_key: dict, ok: bool = True) -> JobDescriber:
    return JobDescriber(KnownScriptPaths(set(leaves_by_key)), StubResolver(leaves_by_key, ok))


def test_backbone_defaults_read_as_single_training():
    text = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", {})

    assert text.startswith("single training · resunet-conv")
    assert "dataset base_dataset_w20_10" in text
    assert "params params_Ng3_sigonly_k5.npy" in text
    assert "patch (64, 32)" in text
    assert "epochs 60" in text
    assert "run" not in text.replace("single training", "")


def test_backbone_trials_read_as_experiment():
    overrides = {"trials_enabled": "True", "trials_mode": "presence"}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert text.startswith("presence trials experiment · resunet-conv")


def test_overrides_win_over_defaults():
    overrides = {"backbone_name": "nafnet", "backbone_head": "set_pred", "infer_after": "True"}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert "nafnet-set_pred" in text
    assert "inference after" in text




def test_generate_dataset_shows_scene_shape_and_seed():
    text = _describer({"generate_dataset": GENERATE_DATASET_LEAVES}).describe("generate_dataset", "python", {})

    assert "crop [0, 1512, 0, 1024]" in text
    assert "K 2" in text
    assert "seed 0" in text


def test_generate_dataset_output_override_appears():
    text = _describer({"generate_dataset": GENERATE_DATASET_LEAVES}).describe("generate_dataset", "python", {"synthesis.output_directory": "/data/scenes/synthetic_big"})

    assert "synthetic_big" in text


def test_extract_params_empty_filter_reads_all_datasets():
    text = _describer({"extract_params": EXTRACT_LEAVES}).describe("extract_params", "python", {})

    assert "datasets all datasets" in text
    assert "K [5]" in text
    assert "lambda [0.01]" in text
    assert "modes [sigma, sigma_amp, sigma_amp_mu]" in text


def test_unconsumed_overrides_surface_as_extras():
    overrides = {"training.batch_size": "512", "gpu": "2"}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert "training.batch_size=512" in text
    assert "gpu=2" in text


def test_extras_overflow_is_counted():
    overrides = {f"section.field_{i}": str(i) for i in range(6)}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert "+3 more overrides" in text


def test_resolver_failure_still_describes_from_overrides():
    describer = _describer({"train_backbone": BACKBONE_LEAVES}, ok=False)
    text      = describer.describe("train_backbone", "python", {"backbone_name": "unet", "training.epochs": "5"})

    assert text.startswith("single training · unet")
    assert "epochs 5" in text


def test_unknown_script_describes_overrides_only():
    text = _describer({}).describe("mystery_script", "python", {"alpha": "1", "beta": "two"})

    assert text == "alpha=1 · beta=two"


def test_description_is_capped():
    overrides = {"run_name": "x" * 500}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert len(text) <= JobDescriber.MAX_LENGTH


def test_jepa_mode_follows_selected_autoencoders():
    leaves = [
        {"path": "backbone_name",            "value": "resunet"},
        {"path": "backbone_head",            "value": "conv"},
        {"path": "profile_autoencoder_run",  "value": "None"},
        {"path": "profile_autoencoder_mode", "value": "frozen"},
        {"path": "image_autoencoder_run",    "value": "None"},
        {"path": "image_autoencoder_mode",   "value": "frozen"},
    ]
    describer = _describer({"train_jepa": leaves})

    plain = describer.describe("train_jepa", "python", {})
    both  = describer.describe("train_jepa", "python", {"profile_autoencoder_run": "runs/mlp_ae_001", "image_autoencoder_run": "runs/conv2d_ae_007"})

    assert plain.startswith("backbone ·")
    assert both.startswith("image-AE + backbone + profile-AE")
    assert "profile-AE mlp_ae_001 (frozen)" in both
    assert "image-AE conv2d_ae_007 (frozen)" in both


MAIL_LEAVES = [
    {"path": "target",            "value": "experiments"},
    {"path": "experiments_dir",   "value": "/runs/sequences"},
    {"path": "experiment_filter", "value": "[]"},
    {"path": "runs_dir",          "value": "/runs/backbone"},
    {"path": "run_filter",        "value": "[]"},
    {"path": "datasets_dir",      "value": "/data/datasets"},
    {"path": "dataset_filter",    "value": "[]"},
    {"path": "params_dir",        "value": "/data/datasets"},
    {"path": "param_run_filter",  "value": "[]"},
    {"path": "recipient",         "value": "victor@example.com"},
]


def test_mail_lead_surfaces_only_the_selected_target():
    overrides = {"target": "datasets", "dataset_filter": "['ds_alpha']"}
    text      = _describer({"mail_experiment_reports": MAIL_LEAVES}).describe("mail_experiment_reports", "python", overrides)

    assert text.startswith("dataset results")
    assert "root datasets" in text
    assert "ds_alpha" in text
    assert "to victor@example.com" in text
    assert "sequences" not in text


def test_mail_lead_names_the_selected_runs():
    overrides = {"target": "runs", "run_filter": "['unet_a']"}
    text      = _describer({"mail_experiment_reports": MAIL_LEAVES}).describe("mail_experiment_reports", "python", overrides)

    assert text.startswith("run contents")
    assert "root backbone" in text
    assert "unet_a" in text
    assert "datasets" not in text


def test_mail_lead_reads_every_registered_target():
    from configuration.diagnostics import ReportMailEntryConfig

    assert tuple(JobDescriber.MAIL_TARGETS) == ReportMailEntryConfig.TARGETS


def _recorded_cv_run(root, run_tag: str, n_folds: int, seeds: list):
    config                    = CrossValidationConfig()
    config.run_tag            = run_tag
    config.paths.log_base_dir = root
    config.folds.n_folds      = n_folds
    config.seeds              = seeds

    run_dir = root / run_tag
    ConfigCli.save_resolved(config, run_dir / "pipeline" / "resolved_config.json")

    return run_dir


def test_a_resumed_run_is_described_by_what_it_recorded(tmp_path):
    run_dir = _recorded_cv_run(tmp_path, "cv_recorded", n_folds=3, seeds=[1, 2])

    leaves     = [{"path": "folds.n_folds", "value": "10"}, {"path": "seeds", "value": "[0, 1, 2, 3, 4]"}, {"path": "resume_from", "value": ""}]
    describer  = _describer({"cross_validate": leaves})

    fresh   = describer.describe("cross_validate", "py", {})
    resumed = describer.describe("cross_validate", "py", {"resume_from": str(run_dir)})

    assert "10-fold" in fresh
    assert "3-fold" in resumed
    assert "seeds [1, 2]" in resumed
    assert "resuming cv_recorded" in resumed


def test_a_resume_path_that_records_nothing_keeps_the_launch_defaults(tmp_path):
    leaves    = [{"path": "folds.n_folds", "value": "10"}, {"path": "resume_from", "value": ""}]
    describer = _describer({"cross_validate": leaves})

    line = describer.describe("cross_validate", "py", {"resume_from": str(tmp_path)})

    assert "10-fold" in line
    assert f"resuming {tmp_path.name}" in line


def test_rescoring_is_named_only_when_completed_runs_are_not_skipped():
    leaves    = [{"path": "runs_dir", "value": "/runs"}, {"path": "run_filter", "value": "[]"}, {"path": "skip_completed", "value": "True"}]
    describer = _describer({"infer_backbone": leaves})

    assert "rescoring scored runs" not in describer.describe("infer_backbone", "py", {})
    assert "rescoring scored runs" in describer.describe("infer_backbone", "py", {"skip_completed": "False"})


def test_a_non_default_stage_is_named_and_the_full_stage_is_not():
    leaves    = [{"path": "seeds", "value": "[0]"}, {"path": "stage", "value": "full"}]
    describer = _describer({"cross_validate": leaves})

    assert "stage" not in describer.describe("cross_validate", "py", {})
    assert "stage prepare" in describer.describe("cross_validate", "py", {"stage": "prepare"})
    assert "stage compare" in describer.describe("cross_validate", "py", {"stage": "compare"})
