
from __future__ import annotations

import json
import shutil
import subprocess
from enum    import Enum
from pathlib import Path

import pytest

from configuration.synthesis                    import SynthesisConfig
from configuration.training.jepa                import JepaEntryConfig
from configuration.training.profile_autoencoder import ProfileAeEntryConfig
from pipelines.jepa.training.launcher           import JepaTrainingLauncher, JepaTrialScheduler
from pipelines.profile_autoencoder.training.launcher import ProfileAeTrainingLauncher, ProfileAeTrialScheduler
from pipelines.shared.dataset.dataset_spatial   import Layout
from tools.runtime.config_cli                   import ConfigCli
from webui.config_value_render                  import ConfigValueRenderer


REPO_ROOT   = Path(__file__).resolve().parents[2]
HARNESS     = REPO_ROOT / "webui" / "tools" / "visual" / "trial_names.js"

SWEEP_OVERRIDES = {
    "train_profile_autoencoder": [
        "--capacity_trials.widths",              "[64, 128]",
        "--capacity_trials.depths",              "[2, 3]",
        "--smoothness_trials.sobolev_weights",   "[1e-05, 0.5]",
        "--smoothness_trials.latent_noise_stds", "[1e-05, 0.02]",
        "--augmentation_trials.noise_stds",      "[1e-05, 0.01]",
        "--sampling_trials.subsamples",          "[1e-05, 0.5]",
        "--sampling_trials.empty_fracs",         "[0.0, 0.05]",
        "--ema_trials.decays",                   "[1e-05, 0.999]",
        "--stability_trials.warmup_fractions",   "[1e-05, 0.03]",
        "--stability_trials.max_grad_norms",     "[1e-05, 1.0]",
    ],
    "train_jepa": [
        "--profile_autoencoder_run",             "mlp_ae_20260827_1132",
        "--curve_trials.weights",                "[1e-05, 0.25]",
        "--curve_trials.sobolev_weights",        "[1e-05, 1.0]",
        "--masking_trials.ratios",               "[1e-05, 0.5]",
        "--curriculum_trials.swap_epochs",       "[5, 15]",
        "--coupling_trials.target_providers",    "['stopgrad', 'live']",
        "--coupling_trials.finetune_lrs",        "[1e-05, 3e-05]",
        "--ae_run_trials.runs",                  "['mlp_ae_20260827_1132', 'nested/run b']",
        "--augmentation_trials.noise_stds",      "[1e-05, 0.01]",
        "--ema_trials.decays",                   "[1e-05, 0.999]",
        "--stability_trials.warmup_fractions",   "[1e-05, 0.03]",
        "--stability_trials.max_grad_norms",     "[1e-05, 1.0]",
    ],
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root   = tmp_path_factory.mktemp("dataset")
    layout = root / Layout.LAYOUT_RELPATH
    (root / "meta").mkdir()
    layout.parent.mkdir()
    layout.write_text(json.dumps({"channel_labels": SynthesisConfig().channel_labels}))

    return root


def _sweep_argv(key: str, dataset: Path) -> list[str]:
    return SWEEP_OVERRIDES[key] + ["--paths.dataset_path", str(dataset)]


def _payload(entry_config, key: str, argv: list[str]) -> dict:
    config = ConfigCli(entry_config()).apply(argv)
    leaves = []

    for path, value, section, section_class in ConfigCli.detailed_leaves(config):
        leaves.append({
            "path"          : path,
            "value"         : ConfigValueRenderer.render(value),
            "type"          : "none" if value is None else type(value).__name__,
            "editable"      : ConfigCli.is_editable(value),
            "choices"       : [entry.value for entry in type(value)] if isinstance(value, Enum) else None,
            "section"       : section,
            "section_class" : section_class,
        })

    return {"key": key, "leaves": leaves}


def _builder_names(entry_config, key: str, argv: list[str], tmp_path: Path) -> dict:
    payload_path = tmp_path / f"{key}.json"
    payload_path.write_text(json.dumps(_payload(entry_config, key, argv)))

    result = subprocess.run(["node", str(HARNESS), str(payload_path)], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _planner_names(launcher, argv: list[str], mode: str) -> list[str]:
    scheduler = launcher.build_scheduler([*argv, "--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]"])

    return [run_name for run_name, _ in scheduler.planner().plan()]


CASES = [(ProfileAeEntryConfig, "train_profile_autoencoder", ProfileAeTrainingLauncher, ProfileAeTrialScheduler, "main/training/train_profile_autoencoder.py"),
         (JepaEntryConfig,      "train_jepa",                JepaTrainingLauncher,      JepaTrialScheduler,      "main/training/train_jepa.py")]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the experiment-builder script")
@pytest.mark.parametrize(("entry_config", "key", "launcher_class", "scheduler", "entry"), CASES)
def test_preview_names_match_the_planner_names(entry_config, key, launcher_class, scheduler, entry, dataset, tmp_path):
    argv      = _sweep_argv(key, dataset)
    previewed = _builder_names(entry_config, key, argv, tmp_path)
    launcher  = launcher_class(entry_script=Path(entry))

    for mode in scheduler.MODE_SUBDIRS:
        assert mode in previewed, f"the experiment builder offers no '{mode}' mode on {key}"

        planned = _planner_names(launcher, argv, mode)
        shown   = previewed[mode]

        assert shown, f"{mode}: the experiment builder previews no run name at all"
        assert len(shown) == len(planned), f"{mode}: previews {len(shown)} run names, plans {len(planned)}"

        for preview, run_name in zip(shown, planned):
            assert preview.endswith(f"_{run_name}"), f"{mode}: preview '{preview}' does not end in the planned run name '{run_name}'"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the experiment-builder script")
@pytest.mark.parametrize(("entry_config", "key", "launcher_class", "scheduler", "entry"), CASES)
def test_every_scheduler_mode_is_offered(entry_config, key, launcher_class, scheduler, entry, dataset, tmp_path):
    previewed = _builder_names(entry_config, key, _sweep_argv(key, dataset), tmp_path)
    missing   = sorted(mode for mode in scheduler.MODE_SUBDIRS if mode not in previewed)

    assert missing == []
