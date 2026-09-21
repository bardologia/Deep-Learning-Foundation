"""Tests for JobDescriber, which turns a script key and its overrides into a job label.

Covers the lead phrase chosen per entry point, the fields each script contributes,
override precedence over resolved defaults, the recorded configuration a resumed
launch is described by, leftover overrides shown as extras, and the behaviour when
the config resolver cannot answer.
"""

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

DUAL_LEAVES = [
    {"path": "model_name",          "value": "dual_resunet"},
    {"path": "params_backbone",     "value": "unet_skip"},
    {"path": "existence_backbone",  "value": "unet_skip"},
    {"path": "params_input",        "value": "['chan', 'derived']"},
    {"path": "existence_input",     "value": "['chan', 'derived']"},
    {"path": "trials_enabled",      "value": "False"},
    {"path": "trials_mode",         "value": "curriculum"},
    {"path": "run_name",            "value": "None"},
    {"path": "infer_after",         "value": "False"},
    {"path": "paths.dataset_path",  "value": "/data/base_dataset_w20_10"},
    {"path": "training.epochs",     "value": "60"},
]

EXTRACT_LEAVES = [
    {"path": "dataset_filter",    "value": "[]"},
    {"path": "fit_k_values",      "value": "[5]"},
    {"path": "fit_lambda_values", "value": "[0.01]"},
    {"path": "fit_modes",         "value": "['sigma', 'sigma_amp', 'sigma_amp_mu']"},
    {"path": "output_suffix",     "value": "None"},
]


class KnownScriptPaths:
    """Stand-in project paths answering only which script keys exist.

    Attributes:
        known : Script keys the describer should treat as recognised.
    """

    def __init__(self, known: set[str]) -> None:
        """Stores the set of recognised script keys."""
        self.known = known

    def has_script(self, key: str) -> bool:
        """Returns whether the script key is one of the known entry points."""
        return key in self.known


class StubResolver:
    """Stand-in config resolver returning canned default leaves per script key.

    Attributes:
        leaves_by_key : Resolved config leaves keyed by script key.
        ok            : When false, every resolution fails.
    """

    def __init__(self, leaves_by_key: dict, ok: bool = True) -> None:
        """Stores the canned leaves and whether resolution succeeds."""
        self.leaves_by_key = leaves_by_key
        self.ok            = ok

    def resolve(self, key: str, interpreter: str) -> dict:
        """Returns the canned leaves for a script key, or a failure payload when unknown or disabled."""
        if not self.ok or key not in self.leaves_by_key:
            return {"ok": False, "error": "unavailable"}
        return {"ok": True, "leaves": self.leaves_by_key[key]}


def _describer(leaves_by_key: dict, ok: bool = True) -> JobDescriber:
    """Returns a JobDescriber backed by stub paths and resolver over the given leaves."""
    return JobDescriber(KnownScriptPaths(set(leaves_by_key)), StubResolver(leaves_by_key, ok))


def test_backbone_defaults_read_as_single_training():
    """Backbone defaults describe a single training naming architecture, dataset, parameter file, patch size and epochs."""
    text = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", {})

    assert text.startswith("single training · resunet-conv")
    assert "dataset base_dataset_w20_10" in text
    assert "params params_Ng3_sigonly_k5.npy" in text
    assert "patch (64, 32)" in text
    assert "epochs 60" in text
    assert "run" not in text.replace("single training", "")


def test_backbone_trials_read_as_experiment():
    """Enabling trials switches the lead phrase to the trials experiment for the selected mode."""
    overrides = {"trials_enabled": "True", "trials_mode": "presence"}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert text.startswith("presence trials experiment · resunet-conv")


def test_overrides_win_over_defaults():
    """Overridden backbone, head and infer_after values replace the resolved defaults in the description."""
    overrides = {"backbone_name": "nafnet", "backbone_head": "set_pred", "infer_after": "True"}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert "nafnet-set_pred" in text
    assert "inference after" in text


def test_dual_lead_names_the_selected_trunks():
    """A dual run is described by its trunk backbone rather than the dual model name."""
    text = _describer({"train_dual": DUAL_LEAVES}).describe("train_dual", "python", {})

    assert text.startswith("single training · dual unet_skip")
    assert "dual_resunet" not in text


def test_dual_lead_names_both_trunks_when_they_differ():
    """Differing parameter and existence trunks are joined into one dotted trunk pair and not repeated as an extra."""
    overrides = {"existence_backbone": "local_cnn"}
    text      = _describer({"train_dual": DUAL_LEAVES}).describe("train_dual", "python", overrides)

    assert "dual unet_skip.local_cnn" in text
    assert "existence_backbone=" not in text


def test_generate_dataset_shows_scene_shape_and_seed():
    """Dataset synthesis is described by its crop, slot count and seed."""
    text = _describer({"generate_dataset": GENERATE_DATASET_LEAVES}).describe("generate_dataset", "python", {})

    assert "crop [0, 1512, 0, 1024]" in text
    assert "K 2" in text
    assert "seed 0" in text


def test_generate_dataset_output_override_appears():
    """An explicit output directory override is shown as its tail."""
    text = _describer({"generate_dataset": GENERATE_DATASET_LEAVES}).describe("generate_dataset", "python", {"synthesis.output_directory": "/data/scenes/synthetic_big"})

    assert "synthetic_big" in text


def test_extract_params_empty_filter_reads_all_datasets():
    """An empty dataset filter reads as all datasets alongside the K, lambda and mode grids."""
    text = _describer({"extract_params": EXTRACT_LEAVES}).describe("extract_params", "python", {})

    assert "datasets all datasets" in text
    assert "K [5]" in text
    assert "lambda [0.01]" in text
    assert "modes [sigma, sigma_amp, sigma_amp_mu]" in text


def test_unconsumed_overrides_surface_as_extras():
    """Overrides the describer does not fold into named fields appear verbatim as key=value extras."""
    overrides = {"training.batch_size": "512", "gpu": "2"}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert "training.batch_size=512" in text
    assert "gpu=2" in text


def test_extras_overflow_is_counted():
    """Extras beyond the display limit are summarised as a remaining-override count."""
    overrides = {f"section.field_{i}": str(i) for i in range(6)}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert "+3 more overrides" in text


def test_resolver_failure_still_describes_from_overrides():
    """A failed resolution still yields a description built from the overrides alone."""
    describer = _describer({"train_backbone": BACKBONE_LEAVES}, ok=False)
    text      = describer.describe("train_backbone", "python", {"backbone_name": "unet", "training.epochs": "5"})

    assert text.startswith("single training · unet")
    assert "epochs 5" in text


def test_unknown_script_describes_overrides_only():
    """An unrecognised script key is described purely by its overrides."""
    text = _describer({}).describe("mystery_script", "python", {"alpha": "1", "beta": "two"})

    assert text == "alpha=1 · beta=two"


def test_description_is_capped():
    """An oversized override value is truncated so the description stays within the maximum length."""
    overrides = {"run_name": "x" * 500}
    text      = _describer({"train_backbone": BACKBONE_LEAVES}).describe("train_backbone", "python", overrides)

    assert len(text) <= JobDescriber.MAX_LENGTH


def test_jepa_mode_follows_selected_autoencoders():
    """The JEPA lead lists only the components actually selected, naming each autoencoder run and its mode."""
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


def _recorded_cv_run(root, run_tag: str, n_folds: int, seeds: list):
    """Writes a cross-validation run directory holding the recording a resumed launch reads.

    Args:
        root    : Directory the run is created in.
        run_tag : Name of the run directory.
        n_folds : Fold count written into the recording.
        seeds   : Seed list written into the recording.

    Returns:
        Path of the run directory.
    """
    config                    = CrossValidationConfig()
    config.run_tag            = run_tag
    config.paths.log_base_dir = root
    config.folds.n_folds      = n_folds
    config.seeds              = seeds

    run_dir = root / run_tag
    ConfigCli.save_resolved(config, run_dir / "pipeline" / "resolved_config.json")

    return run_dir


def test_a_resumed_run_is_described_by_what_it_recorded(tmp_path):
    """A resume is described by the recorded fold count and seeds, not by the entry point defaults."""
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
    """A resume path without a recording leaves the described fields on the entry point defaults."""
    leaves    = [{"path": "folds.n_folds", "value": "10"}, {"path": "resume_from", "value": ""}]
    describer = _describer({"cross_validate": leaves})

    line = describer.describe("cross_validate", "py", {"resume_from": str(tmp_path)})

    assert "10-fold" in line
    assert f"resuming {tmp_path.name}" in line


def test_rescoring_is_named_only_when_completed_runs_are_not_skipped():
    """The inference line names a rescoring pass only when skip_completed is off."""
    leaves    = [{"path": "runs_dir", "value": "/runs"}, {"path": "run_filter", "value": "[]"}, {"path": "skip_completed", "value": "True"}]
    describer = _describer({"infer_backbone": leaves})

    assert "rescoring scored runs" not in describer.describe("infer_backbone", "py", {})
    assert "rescoring scored runs" in describer.describe("infer_backbone", "py", {"skip_completed": "False"})
