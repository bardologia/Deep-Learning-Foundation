from __future__ import annotations

import sys
from pathlib import Path


class ProjectPaths:

    PRIORITY        = ["conda:dlr-cu12", "conda:Dune", "conda:nazaria"]
    SCRIPT_PRIORITY = {
        "generate_tomogram"       : ["conda:stetools"],
        "generate_interferograms" : ["conda:stetools"],
    }

    ENTRY_OVERRIDES = {
        "train_backbone": {
            "config_module" : "configuration.training.backbone",
            "config_class"  : "BackboneEntryConfig",
        },
        "train_profile_autoencoder": {
            "config_module" : "configuration.training.profile_autoencoder",
            "config_class"  : "ProfileAeEntryConfig",
        },
        "train_image_autoencoder": {
            "config_module" : "configuration.training.image_autoencoder",
            "config_class"  : "ImageAeEntryConfig",
        },
        "train_jepa": {
            "config_module" : "configuration.training.jepa",
            "config_class"  : "JepaEntryConfig",
        },
        "train_unrolled": {
            "config_module" : "configuration.training.unrolled",
            "config_class"  : "UnrolledEntryConfig",
        },
        "train_dual": {
            "config_module" : "configuration.training.dual",
            "config_class"  : "DualEntryConfig",
        },
        "infer_backbone": {
            "config_module" : "configuration.inference",
            "config_class"  : "BackboneInferenceEntryConfig",
        },
        "infer_profile_autoencoder": {
            "config_module" : "configuration.inference",
            "config_class"  : "ProfileAeInferenceEntryConfig",
        },
        "infer_image_autoencoder": {
            "config_module" : "configuration.inference",
            "config_class"  : "ImageAeInferenceEntryConfig",
        },
        "infer_unrolled": {
            "config_module" : "configuration.inference",
            "config_class"  : "UnrolledInferenceEntryConfig",
        },
        "infer_dual": {
            "config_module" : "configuration.inference",
            "config_class"  : "DualInferenceEntryConfig",
        },
        "cross_validate": {
            "config_module" : "configuration.cross_validation.general",
            "config_class"  : "CrossValidationConfig",
        },
        "sweep_patches": {
            "config_module" : "configuration.patch_sweep.general",
            "config_class"  : "PatchSweepConfig",
        },
    }

    def __init__(self) -> None:
        self.webui_root     = Path(__file__).resolve().parent
        self.repo_root      = self.webui_root.parent
        self.main_dir       = self.repo_root / "main"
        self.scripts_dir    = self.repo_root / "scripts"
        self.config_dir     = self.repo_root / "configuration"
        self.static_dir     = self.webui_root / "static"
        self.logs_dir       = self.repo_root / "logs"
        self.gpu_guard_dir  = self.logs_dir / "gpu_guard"
        self.gpu_pools_dir  = self.logs_dir / "gpu_pools"
        self.saved_runs_dir = self.logs_dir / "saved_runs"

    SCRIPT_DIRS = {
        "pre_process"                     : "processing",
        "extract_params"                  : "processing",
        "inject_external_params"          : "processing",
        "generate_tomogram"               : "processing",
        "generate_interferograms"         : "processing",
        "train_backbone"                  : "training",
        "train_profile_autoencoder"       : "training",
        "train_image_autoencoder"         : "training",
        "train_jepa"                      : "training",
        "train_unrolled"                  : "training",
        "train_dual"                      : "training",
        "infer_backbone"                  : "inference",
        "infer_profile_autoencoder"       : "inference",
        "infer_image_autoencoder"         : "inference",
        "infer_unrolled"                  : "inference",
        "infer_dual"                      : "inference",
        "benchmark"                       : "experiments",
        "cross_validate"                  : "experiments",
        "sweep_patches"                   : "experiments",
        "tune"                            : "experiments",
        "tune_dataloader"                 : "experiments",
        "analyze_preprocessing"           : "analysis",
        "analyze_param_extraction"        : "analysis",
        "compare_trials"                  : "analysis",
        "compare_preprocessing_trials"    : "analysis",
        "compare_param_extraction_trials" : "analysis",
        "compare_runs"                    : "analysis",
        "compare_seeds"                   : "analysis",
        "xray_weights"                    : "analysis",
        "xray_activations"                : "analysis",
        "measure_receptive_field"         : "analysis",
        "attribute_inputs"                : "analysis",
        "capture_attention"               : "analysis",
        "probe_layers"                    : "analysis",
        "compare_representations"         : "analysis",
        "map_loss_landscape"              : "analysis",
        "stress_inputs"                   : "analysis",
        "animate_training"                : "analysis",
        "export_paper_figures"            : "analysis",
        "export_tensorboard_plots"        : "analysis",
        "collect_reports"                 : "analysis",
    }

    def has_script(self, key: str) -> bool:
        return key in self.SCRIPT_DIRS

    def script_entry(self, key: str) -> dict:
        override  = self.ENTRY_OVERRIDES.get(key, {})
        file_stem = override.get("file", key)
        subdir    = self.SCRIPT_DIRS[key]
        rel       = f"main/{subdir}/{file_stem}.py"

        return {
            "path"          : self.main_dir / subdir / f"{file_stem}.py",
            "rel"           : rel,
            "config_module" : override.get("config_module"),
            "config_class"  : override.get("config_class"),
        }

    def discover_interpreters(self) -> list[dict]:
        found   = []
        seen    = set()
        current = str(Path(sys.executable))

        home  = Path.home()
        bases = [home / "miniconda3", home / "anaconda3", home / ".conda"]
        for base in bases:
            base_py = base / "bin" / "python"
            if base_py.exists() and str(base_py) not in seen:
                found.append({"label": "conda:base", "path": str(base_py), "current": str(base_py) == current})
                seen.add(str(base_py))

            envs_dir = base / "envs"
            if envs_dir.is_dir():
                for env in sorted(envs_dir.iterdir()):
                    env_py = env / "bin" / "python"
                    if env_py.exists() and str(env_py) not in seen:
                        found.append({"label": f"conda:{env.name}", "path": str(env_py), "current": str(env_py) == current})
                        seen.add(str(env_py))

        if current not in seen and Path(current).exists():
            found.insert(0, {"label": "current", "path": current, "current": True})

        return found

    def preferred_interpreter(self, interpreters: list[dict], script_key: str = "") -> str:
        priority = self.SCRIPT_PRIORITY.get(script_key, []) + self.PRIORITY
        for wanted in priority:
            for item in interpreters:
                if item["label"] == wanted:
                    return item["path"]
        return interpreters[0]["path"] if interpreters else sys.executable
