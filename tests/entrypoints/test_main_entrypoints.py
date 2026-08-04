from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from configuration.inference  import BackboneInferenceEntryConfig
from tools.runtime.config_cli import ConfigCli


_MAIN_DIR = Path(__file__).resolve().parents[2] / "main"
_SUBDIRS  = ("processing", "training", "inference", "experiments", "analysis")

THREAD_KEYS = ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")


def _script_path(name):
    for subdir in _SUBDIRS:
        candidate = _MAIN_DIR / subdir / f"{name}.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No entry script named {name}.py under {_MAIN_DIR}")

DEFER_HEAVY_IMPORTS = (
    "train_backbone",
    "train_jepa",
    "train_profile_autoencoder",
    "train_image_autoencoder",
    "train_unrolled",
    "train_dual",
    "infer_backbone",
    "infer_profile_autoencoder",
    "infer_image_autoencoder",
    "infer_unrolled",
    "infer_dual",
    "generate_interferograms",
    "generate_tomogram",
    "tune",
    "benchmark",
    "cross_validate",
    "sweep_patches",
    "tune_dataloader",
    "pre_process",
    "extract_params",
    "inject_external_params",
    "analyze_preprocessing",
    "analyze_param_extraction",
    "export_tensorboard_plots",
    "collect_reports",
)

CLI_MODULES = (
    "infer_backbone",
    "infer_profile_autoencoder",
    "infer_image_autoencoder",
    "infer_unrolled",
    "infer_dual",
    "analyze_preprocessing",
    "analyze_param_extraction",
    "pre_process",
    "extract_params",
    "inject_external_params",
    "tune",
    "tune_dataloader",
    "benchmark",
    "cross_validate",
    "sweep_patches",
    "compare_runs",
    "compare_trials",
    "compare_param_extraction_trials",
    "compare_preprocessing_trials",
    "export_tensorboard_plots",
    "collect_reports",
)

ENTRY_CONFIGS = {
    "infer_backbone"                  : ("configuration.inference",          "BackboneInferenceEntryConfig"),
    "infer_profile_autoencoder"       : ("configuration.inference",          "ProfileAeInferenceEntryConfig"),
    "infer_image_autoencoder"         : ("configuration.inference",          "ImageAeInferenceEntryConfig"),
    "infer_unrolled"                  : ("configuration.inference",          "UnrolledInferenceEntryConfig"),
    "infer_dual"                      : ("configuration.inference",          "DualInferenceEntryConfig"),
    "pre_process"                     : ("configuration.sar.processing_config",        "PreProcessEntryConfig"),
    "extract_params"                  : ("configuration.param_extraction", "ExtractParamsEntryConfig"),
    "inject_external_params"          : ("configuration.param_extraction", "InjectExternalParamsEntryConfig"),
    "analyze_preprocessing"           : ("configuration.sar.processing_config", "PreprocessInferenceConfig"),
    "analyze_param_extraction"        : ("configuration.param_extraction",      "ParamExtractionInferenceConfig"),
    "tune"                            : ("configuration.tuning",                       "TuningEntryConfig"),
    "tune_dataloader"                 : ("configuration.benchmark.dataloader_tuning",  "DataLoaderTuningEntryConfig"),
    "benchmark"                       : ("configuration.benchmark",                    "BenchmarkConfig"),
    "cross_validate"                  : ("configuration.cross_validation",             "CrossValidationConfig"),
    "sweep_patches"                   : ("configuration.patch_sweep",                  "PatchSweepConfig"),
    "compare_runs"                    : ("configuration.comparison",                   "ComparisonEntryConfig"),
    "compare_trials"                  : ("configuration.comparison",                   "TrialComparisonConfig"),
    "compare_param_extraction_trials" : ("configuration.comparison",             "ParamExtractionComparisonConfig"),
    "compare_preprocessing_trials"    : ("configuration.comparison",             "PreprocessingComparisonConfig"),
    "export_tensorboard_plots"        : ("configuration.diagnostics",                  "TensorboardExportEntryConfig"),
    "collect_reports"                 : ("configuration.diagnostics",                  "ReportCollectionEntryConfig"),
}

ENTRY_SCRIPTS = sorted(script for subdir in _SUBDIRS for script in (_MAIN_DIR / subdir).glob("*.py") if not script.stem.startswith("_"))


@pytest.fixture
def main_on_path(monkeypatch):
    for subdir in _SUBDIRS:
        monkeypatch.syspath_prepend(str(_MAIN_DIR / subdir))
    monkeypatch.syspath_prepend(str(_MAIN_DIR))
    return _MAIN_DIR


@pytest.fixture
def frozen_env(monkeypatch):
    snapshot = dict(os.environ)
    yield
    for key in set(os.environ) - set(snapshot):
        monkeypatch.delenv(key, raising=False)
    for key, value in snapshot.items():
        monkeypatch.setenv(key, value)


def _import_main(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_module_imports_without_error(name, main_on_path, frozen_env):
    module = _import_main(name)

    assert module is not None


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_module_exposes_main_callable(name, main_on_path, frozen_env):
    module = _import_main(name)

    assert callable(module.main)


@pytest.mark.parametrize("script", ENTRY_SCRIPTS, ids=lambda script: script.stem)
def test_script_imports_in_launcher_subprocess(script, tmp_path):
    code   = f"import runpy; runpy.run_path({str(script)!r}, run_name='launcher_import_check')"
    result = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(tmp_path), capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr


def test_import_does_not_set_cuda_visible_devices(main_on_path, frozen_env, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    for name in DEFER_HEAVY_IMPORTS:
        _import_main(name)

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("name", ("train_jepa", "train_profile_autoencoder", "train_image_autoencoder"))
def test_seed_sweep_entries_hand_their_script_to_the_fanout(name, main_on_path, frozen_env, monkeypatch):
    import pipelines.shared.training.training_launcher as training_launcher

    captured = {}

    class _RecordingLauncher:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    module = _import_main(name)

    monkeypatch.setattr(training_launcher, "SeedSweepLauncher", _RecordingLauncher)
    monkeypatch.setattr(module.EnvironmentPinner, "gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [f"{name}.py"])

    module.main()

    assert captured["ran"]          is True
    assert captured["entry_script"] == _script_path(name)


@pytest.mark.parametrize("name", ("train_backbone", "train_jepa", "train_profile_autoencoder", "train_image_autoencoder", "train_unrolled", "train_dual"))
def test_train_main_defers_heavy_imports(name, tmp_path):
    script = _script_path(name)
    code   = f"import sys; sys.path.insert(0, {str(script.parent)!r}); import {name}; print(sorted(module for module in sys.modules if module.split('.')[0] in {{'pipelines', 'models', 'torch'}}))"
    result = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(tmp_path), capture_output=True, text=True, timeout=120)

    assert result.returncode     == 0, result.stderr
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_import_does_not_pin_threads(name, main_on_path, frozen_env, monkeypatch):
    for key in THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)

    _import_main(name)

    assert all(key not in os.environ for key in THREAD_KEYS)


def test_compare_runs_has_no_module_level_side_effects(main_on_path, monkeypatch):
    for key in THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)

    sys.modules.pop("compare_runs", None)
    importlib.import_module("compare_runs")

    assert all(key not in os.environ for key in THREAD_KEYS)


def test_compare_runs_exposes_main_callable(main_on_path, frozen_env):
    module = _import_main("compare_runs")

    assert callable(module.main)


@pytest.mark.parametrize("name", CLI_MODULES)
def test_entry_config_constructs_from_defaults(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    config = config_class()

    assert config is not None


@pytest.mark.parametrize("name", CLI_MODULES)
def test_config_cli_builds_parser_for_entry_config(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    cli = ConfigCli(config_class(), description=f"{name} cli")

    assert cli.parser is not None


@pytest.mark.parametrize("name", CLI_MODULES)
def test_config_cli_help_config_exits_zero(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    cli = ConfigCli(config_class())

    with pytest.raises(SystemExit) as excinfo:
        cli.apply(["--help-config"])

    assert excinfo.value.code == 0


@pytest.mark.parametrize("name", CLI_MODULES)
def test_config_cli_apply_no_args_returns_unmodified_config(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    config = config_class()
    result = ConfigCli(config).apply([])

    assert result is config


def test_config_cli_rejects_unknown_override():
    cli = ConfigCli(BackboneInferenceEntryConfig())

    with pytest.raises(ValueError):
        cli.apply(["--not-a-real-flag", "1"])


def test_infer_worker_requires_run_dir_and_config(main_on_path, frozen_env, monkeypatch):
    module = _import_main("infer_backbone")
    monkeypatch.setattr(sys, "argv", ["infer_backbone.py", "--worker", "--run-dir", "x"])

    with pytest.raises(SystemExit):
        module.main()


def test_tune_worker_requires_model(main_on_path, frozen_env, monkeypatch):
    module = _import_main("tune")
    monkeypatch.setattr(sys, "argv", ["tune.py", "--worker"])

    with pytest.raises(SystemExit):
        module.main()


def test_benchmark_worker_requires_model_and_run_tag(main_on_path, frozen_env, monkeypatch):
    module = _import_main("benchmark")
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--worker", "train"])

    with pytest.raises(SystemExit):
        module.main()


def test_cross_validate_worker_requires_fold(main_on_path, frozen_env, monkeypatch):
    module = _import_main("cross_validate")
    monkeypatch.setattr(sys, "argv", ["cross_validate.py", "--worker", "train"])

    with pytest.raises(SystemExit):
        module.main()


def test_cross_validate_infer_worker_requires_split(main_on_path, frozen_env, monkeypatch):
    module = _import_main("cross_validate")
    monkeypatch.setattr(sys, "argv", ["cross_validate.py", "--worker", "infer", "--fold", "0", "--run-tag", "t"])

    with pytest.raises(SystemExit):
        module.main()
