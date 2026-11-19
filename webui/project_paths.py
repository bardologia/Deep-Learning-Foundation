
from __future__ import annotations

import sys
from pathlib import Path

from configuration.reference_layout import ReferenceLayout
from tools.runtime.machine_profile  import MachineProfile


class ProjectPaths:

    REQUIRED_DIRS = ("main", "configuration", "tools")
    CONDA_BASES   = ("miniconda3", "anaconda3", ".conda")

    ENTRY_OVERRIDES = {
        "generate_dataset": {
            "config_module" : "configuration.synthesis",
            "config_class"  : "GenerateDatasetEntryConfig",
        },
        "train_backbone": {
            "config_module" : "configuration.training.backbone",
            "config_class"  : "BackboneEntryConfig",
        },
        "infer_backbone": {
            "config_module" : "configuration.inference",
            "config_class"  : "BackboneInferenceEntryConfig",
        },
        "cross_validate": {
            "config_module" : "configuration.cross_validation.general",
            "config_class"  : "CrossValidationConfig",
        },
    }

    def __init__(self, root: Path | None = None, profile: MachineProfile | None = None) -> None:
        self.profile        = MachineProfile.current() if profile is None else profile
        self.webui_root     = Path(__file__).resolve().parent
        self.repo_root      = self.webui_root.parent if root is None else Path(root).resolve()
        self.main_dir       = self.repo_root / "main"
        self.scripts_dir    = self.repo_root / "scripts"
        self.config_dir     = self.repo_root / "configuration"
        self.static_dir     = self.webui_root / "static"
        self.runs_dir       = Path(self.profile.roots["runs"]) if self.profile.roots["runs"] else self.repo_root / ReferenceLayout.RUNS_ROOT
        self.data_root      = Path(self.profile.roots["data"]) if self.profile.roots["data"] else self.repo_root / ReferenceLayout.DATA_ROOT
        self.logs_dir       = self.repo_root / "logs"
        self.gpu_guard_dir  = self.logs_dir / "gpu_guard"
        self.gpu_pools_dir  = self.logs_dir / "gpu_pools"
        self.saved_runs_dir = self.logs_dir / "saved_runs"
        self.templates_dir  = self.logs_dir / "config_templates"
        self.sequences_dir  = self.logs_dir / "sequences"

        self._require_repository()

    def _require_repository(self) -> None:
        missing = [name for name in self.REQUIRED_DIRS if not (self.repo_root / name).is_dir()]

        if missing:
            raise ValueError(f"project root {self.repo_root} is not an ML Template repository: missing {', '.join(missing)}")

    @property
    def priority(self) -> list[str]:
        return [f"conda:{self.profile.env_name}"]

    def roots(self) -> dict:
        return {"datasets": str(self.data_root), "runs": str(self.runs_dir)}

    def profile_signature(self) -> tuple:
        return self.tree_signature([self.profile.path]) if self.profile.path is not None else ()

    def tree_signature(self, paths) -> tuple:
        stamps = []
        for path in paths:
            try:
                stamps.append((path.name, path.stat().st_mtime_ns))
            except OSError:
                continue
        return tuple(stamps)

    def config_signature(self) -> tuple:
        return self.tree_signature(sorted(self.config_dir.rglob("*.py")))

    SCRIPT_DIRS = {
        "generate_dataset"                : "processing",
        "extract_params"                  : "processing",
        "train_backbone"                  : "training",
        "infer_backbone"                  : "inference",
        "benchmark"                       : "experiments",
        "cross_validate"                  : "experiments",
        "tune"                            : "experiments",
        "tune_dataloader"                 : "experiments",
        "analyze_param_extraction"        : "analysis",
        "compare_trials"                  : "analysis",
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
        "export_result_tables"            : "analysis",
        "profile_training"                : "analysis",
        "export_onnx"                     : "analysis",
        "export_tensorboard_plots"        : "analysis",
        "collect_reports"                 : "analysis",
        "mail_experiment_reports"         : "analysis",
    }

    def has_script(self, key: str) -> bool:
        return key in self.SCRIPT_DIRS

    def script_entry(self, key: str) -> dict:
        override = self.ENTRY_OVERRIDES.get(key, {})
        subdir   = self.SCRIPT_DIRS[key]
        rel      = f"main/{subdir}/{key}.py"

        return {
            "path"          : self.main_dir / subdir / f"{key}.py",
            "rel"           : rel,
            "config_module" : override.get("config_module"),
            "config_class"  : override.get("config_class"),
        }

    def discover_interpreters(self) -> list[dict]:
        found   = []
        seen    = set()
        current = str(Path(sys.executable))

        home  = Path.home()
        bases = [home / name for name in self.CONDA_BASES]
        if self.profile.machine["conda_root"]:
            bases.insert(0, Path(self.profile.machine["conda_root"]))
        for base in dict.fromkeys(bases):
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
        for wanted in self.priority:
            for item in interpreters:
                if item["label"] == wanted:
                    return item["path"]
        return interpreters[0]["path"] if interpreters else sys.executable
