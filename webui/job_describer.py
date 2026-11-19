
from __future__ import annotations

from project_paths          import ProjectPaths
from run_config_reader      import ResumeRecording
from script_config_resolver import ScriptConfigResolver


class JobDescriber:

    MAX_LENGTH    = 240
    MAX_EXTRAS    = 3
    MAX_EXTRA_LEN = 40
    UNSET_VALUES  = ("", "None", "none", "null", "[]", "{}", "()")

    LEADS = {
        "train_backbone"            : "_lead_train_backbone",
        "train_jepa"                : "_lead_train_jepa",
        "train_profile_autoencoder" : "_lead_train_profile_autoencoder",
        "benchmark"                 : "_lead_benchmark",
        "cross_validate"            : "_lead_cross_validate",
        "tune"                      : "_lead_tune",
        "tune_dataloader"           : "_lead_tune_dataloader",
        "mail_experiment_reports"   : "_lead_mail_experiment_reports",
    }

    SPECS = {
        "generate_dataset": [
            ("output", "synthesis.output_directory", "opt_tail"),
            ("crop",   "synthesis.global_crop",      "text"),
            ("K",      "synthesis.k_max",            "text"),
            ("params", "synthesis.params_run_name",  "opt"),
            ("seed",   "synthesis.seed",             "text"),
        ],
        "extract_params": [
            ("datasets", "dataset_filter",    "list", "all datasets"),
            ("K",        "fit_k_values",      "list"),
            ("lambda",   "fit_lambda_values", "list"),
            ("modes",    "fit_modes",         "list"),
            ("suffix",   "output_suffix",     "opt"),
        ],
        "train_backbone": [
            ("dataset",          "paths.dataset_path",    "opt_tail"),
            ("params",           "paths.parameters_path", "opt_tail"),
            ("patch",            "training.patch_size",   "text"),
            ("epochs",           "training.epochs",       "text"),
            ("run",              "run_name",              "opt"),
            ("resuming",         "resume_from",           "opt_tail"),
            ("inference after",  "infer_after",           "flag"),
            ("inference at end", "infer_at_end",          "flag"),
        ],
        "train_profile_autoencoder": [
            ("model",    "ae_model_name",         "text"),
            ("params",   "paths.parameters_path", "opt_tail"),
            ("epochs",   "training.epochs",       "text"),
            ("run",      "run_name",              "opt"),
            ("resuming", "resume_from",           "opt_tail"),
        ],
        "train_image_autoencoder": [
            ("model",    "ae_model_name",      "text"),
            ("dataset",  "paths.dataset_path", "opt_tail"),
            ("epochs",   "training.epochs",    "text"),
            ("run",      "run_name",           "opt"),
            ("resuming", "resume_from",        "opt_tail"),
        ],
        "train_jepa": [
            ("dataset",          "paths.dataset_path", "opt_tail"),
            ("epochs",           "training.epochs",    "text"),
            ("run",              "run_name",           "opt"),
            ("resuming",         "resume_from",        "opt_tail"),
            ("inference after",  "infer_after",        "flag"),
            ("inference at end", "infer_at_end",       "flag"),
        ],
        "infer_backbone": [
            ("root",                  "runs_dir",        "opt_tail"),
            ("filter",                "run_filter",      "list", "all runs"),
            ("split",                 "inference.split", "text"),
            ("gpus",                  "gpus",            "list"),
            ("rescoring scored runs", "skip_completed",  "unflag"),
        ],
        "infer_profile_autoencoder": [
            ("root",                  "runs_dir",       "opt_tail"),
            ("filter",                "run_filter",     "list", "all runs"),
            ("gpus",                  "gpus",           "list"),
            ("rescoring scored runs", "skip_completed", "unflag"),
        ],
        "infer_image_autoencoder": [
            ("root",                  "runs_dir",       "opt_tail"),
            ("filter",                "run_filter",     "list", "all runs"),
            ("gpus",                  "gpus",           "list"),
            ("rescoring scored runs", "skip_completed", "unflag"),
        ],
        "benchmark": [
            ("heads",    "heads",                 "list"),
            ("losses",   "sweep_loss_components", "list"),
            ("seeds",    "seeds",                 "list"),
            ("gpus",     "gpus",                  "list"),
            ("tag",      "run_tag",               "opt"),
            ("resuming", "resume_from",           "opt_tail"),
            ("stage",    "stage",                 "unless", "full"),
        ],
        "cross_validate": [
            ("seeds",    "seeds",       "list"),
            ("gpus",     "gpus",        "list"),
            ("tag",      "run_tag",     "opt"),
            ("resuming", "resume_from", "opt_tail"),
            ("stage",    "stage",       "unless", "full"),
        ],
        "tune": [
            ("trials",       "tuning.n_trials", "text"),
            ("epochs/trial", "tuning.n_epochs", "text"),
            ("heads",        "heads",           "list"),
            ("tag",          "run_tag",         "opt"),
            ("resuming",     "resume_from",     "opt_tail"),
            ("stage",        "stage",           "unless", "full"),
            ("workers",      "n_workers",       "unless", "0"),
        ],
        "tune_dataloader": [
            ("model",       "model_name",  "opt"),
            ("batch sizes", "batch_sizes", "list"),
        ],
        "analyze_param_extraction": [
            ("trials", "run_tags",   "list", "all trials"),
            ("root",   "params_dir", "opt_tail"),
        ],
        "analyze_normalization": [
            ("dataset", "dataset_dir",     "opt_tail"),
            ("params",  "parameters_path", "opt_tail"),
        ],
        "analyze_symbolic": [
            ("dataset", "dataset_dir",     "opt_tail"),
            ("params",  "parameters_path", "opt_tail"),
        ],
        "compare_trials": [
            ("runs", "run_tags", "list", "all runs"),
            ("root", "runs_dir", "opt_tail"),
        ],
        "compare_param_extraction_trials": [
            ("trials", "run_tags",   "list", "all trials"),
            ("root",   "params_dir", "opt_tail"),
        ],
        "compare_runs": [
            ("tag",       "run_tag",         "opt"),
            ("reference", "reference_model", "text"),
        ],
        "compare_seeds": [
            ("groups", "group_tags", "list", "all groups"),
            ("root",   "runs_dir",   "opt_tail"),
        ],
        "xray_weights": [
            ("runs",       "run_filter",          "list", "all runs"),
            ("root",       "runs_dir",            "opt_tail"),
            ("checkpoint", "checkpoint_filename", "text"),
        ],
        "measure_receptive_field": [
            ("runs",   "run_filter", "list", "all runs"),
            ("root",   "runs_dir",   "opt_tail"),
            ("window", "window",     "text"),
        ],
        "xray_activations": [
            ("runs",    "run_filter",  "list", "all runs"),
            ("root",    "runs_dir",    "opt_tail"),
            ("batches", "max_batches", "text"),
        ],
        "attribute_inputs": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
        ],
        "capture_attention": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
        ],
        "probe_layers": [
            ("runs",   "run_filter", "list", "all runs"),
            ("root",   "runs_dir",   "opt_tail"),
            ("layers", "max_layers", "text"),
        ],
        "compare_representations": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
        ],
        "map_loss_landscape": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
            ("span", "span",       "text"),
        ],
        "stress_inputs": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
        ],
        "animate_training": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
        ],
        "export_paper_figures": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
            ("into", "output_dir", "opt_tail"),
        ],
        "export_tensorboard_plots": [
            ("runs", "run_filter", "list", "all runs"),
            ("root", "runs_dir",   "opt_tail"),
        ],
        "collect_reports": [
            ("runs", "run_filter",    "list", "all runs"),
            ("root", "runs_dir",      "opt_tail"),
            ("into", "collector_dir", "opt_tail"),
        ],
        "mail_experiment_reports": [
            ("to", "recipient", "opt"),
        ],
    }

    MAIL_TARGETS = {
        "experiments"      : ("experiment reports", "experiments_dir", "experiment_filter", "all experiments"),
        "runs"             : ("run contents",       "runs_dir",        "run_filter",        "all runs"),
        "datasets"         : ("dataset results",    "datasets_dir",    "dataset_filter",    "all datasets"),
        "param_extraction" : ("param run results",  "params_dir",      "param_run_filter",  "all parameter runs"),
    }

    def __init__(self, paths: ProjectPaths, resolver: ScriptConfigResolver) -> None:
        self.paths    = paths
        self.resolver = resolver

    def _values(self, key: str, interpreter: str, overrides: dict) -> dict:
        values = {}

        if self.paths.has_script(key):
            resolved = self.resolver.resolve(key, interpreter)
            if resolved.get("ok"):
                values = {leaf["path"]: str(leaf["value"]) for leaf in resolved["leaves"]}

        values.update(self._recorded(overrides))
        values.update({path: str(value) for path, value in overrides.items()})

        return values

    def _recorded(self, overrides: dict) -> dict:
        resume_from = str(overrides.get("resume_from", "")).strip()
        if not resume_from:
            return {}

        return ResumeRecording.values(resume_from)

    def _lead_train_backbone(self, values: dict, used: set) -> list[str]:
        used.update(("trials_enabled", "trials_mode", "backbone_name", "backbone_head"))

        mode  = f"{values.get('trials_mode', '')} trials experiment".strip() if self._truthy(values.get("trials_enabled", "")) else "single training"
        model = self._join(values, "backbone_name", "backbone_head")
        return [part for part in (mode, model) if part]

    def _lead_train_jepa(self, values: dict, used: set) -> list[str]:
        used.update(("trials_enabled", "trials_mode", "backbone_name", "backbone_head", "profile_autoencoder_run", "profile_autoencoder_mode", "image_autoencoder_run", "image_autoencoder_mode"))

        profile = values.get("profile_autoencoder_run", "")
        image   = values.get("image_autoencoder_run", "")
        stages  = ["image-AE"] if self._is_set(image) else []
        stages += ["backbone"]
        stages += ["profile-AE"] if self._is_set(profile) else []

        mode  = f"{values.get('trials_mode', '')} trials experiment".strip() if self._truthy(values.get("trials_enabled", "")) else ""
        parts = [mode, " + ".join(stages), self._join(values, "backbone_name", "backbone_head")]
        if self._is_set(profile):
            parts.append(f"profile-AE {self._tail(profile)} ({values.get('profile_autoencoder_mode', '')})")
        if self._is_set(image):
            parts.append(f"image-AE {self._tail(image)} ({values.get('image_autoencoder_mode', '')})")
        return [part for part in parts if part]

    def _lead_train_profile_autoencoder(self, values: dict, used: set) -> list[str]:
        used.update(("trials_enabled", "trials_mode", "ae_model_name"))

        mode  = f"{values.get('trials_mode', '')} trials experiment".strip() if self._truthy(values.get("trials_enabled", "")) else "single training"
        model = values.get("ae_model_name", "")
        return [part for part in (mode, model) if part]

    def _lead_benchmark(self, values: dict, used: set) -> list[str]:
        used.add("training_type")
        return [f"{values.get('training_type', '')} benchmark".strip()]

    CV_MODEL_FIELDS = {
        "profile_autoencoder" : ("autoencoder.ae_model_name",),
        "image_autoencoder"   : ("image_autoencoder.ae_model_name",),
    }

    def _lead_cross_validate(self, values: dict, used: set) -> list[str]:
        used.update(("training_type", "folds.n_folds", "backbone_name", "backbone_head"))
        used.update(field for fields in self.CV_MODEL_FIELDS.values() for field in fields)

        training_type = values.get("training_type", "")
        folds         = values.get("folds.n_folds", "")
        lead          = f"{folds}-fold cross-validation, {training_type}".strip(", ") if folds else f"cross-validation, {training_type}".strip(", ")

        return [part for part in (lead, self._cross_validate_model(training_type, values)) if part]

    def _cross_validate_model(self, training_type: str, values: dict) -> str:
        if training_type in ("profile_autoencoder", "image_autoencoder"):
            return values.get(self.CV_MODEL_FIELDS[training_type][0], "")

        return self._join(values, "backbone_name", "backbone_head")

    def _lead_tune(self, values: dict, used: set) -> list[str]:
        used.add("training_type")
        return [f"Optuna search, {values.get('training_type', '')}".strip(", ")]

    def _lead_tune_dataloader(self, values: dict, used: set) -> list[str]:
        used.add("mode")
        return [f"feed tuner, {values.get('mode', '')}".strip(", ")]

    def _lead_mail_experiment_reports(self, values: dict, used: set) -> list[str]:
        used.add("target")
        for _noun, root, run_filter, _fallback in self.MAIL_TARGETS.values():
            used.update((root, run_filter))

        target = values.get("target", "")
        if target not in self.MAIL_TARGETS:
            return []

        noun, root, run_filter, fallback = self.MAIL_TARGETS[target]

        parts = [noun, self._render("root", values.get(root), "opt_tail", None), self._render("", values.get(run_filter), "list", fallback)]

        return [part for part in parts if part]

    def _lead(self, key: str, values: dict, used: set) -> list[str]:
        builder = self.LEADS.get(key)
        if builder is None:
            return []
        return getattr(self, builder)(values, used)

    def _details(self, key: str, values: dict, used: set) -> list[str]:
        parts = []
        for entry in self.SPECS.get(key, []):
            label, path, kind = entry[0], entry[1], entry[2]
            fallback          = entry[3] if len(entry) > 3 else None
            used.add(path)

            value = values.get(path)
            part  = self._render(label, value, kind, fallback)
            if part:
                parts.append(part)
        return parts

    def _render(self, label: str, value: str | None, kind: str, fallback: str | None) -> str | None:
        if kind == "flag":
            return label if value is not None and self._truthy(value) else None

        if kind == "unflag":
            return label if value is not None and not self._truthy(value) else None

        if kind == "unless":
            return None if value is None or value == fallback else self._labelled(label, self._compact(value))

        if value is None or not self._is_set(value):
            if kind == "list" and fallback and value is not None:
                return self._labelled(label, fallback)
            return None

        if kind in ("tail", "opt_tail"):
            return self._labelled(label, self._tail(value))
        if kind == "list":
            return self._labelled(label, self._compact(value))
        return self._labelled(label, self._compact(value))

    def _extras(self, overrides: dict, used: set) -> list[str]:
        pending = [(path, str(value)) for path, value in overrides.items() if path not in used]
        parts   = [f"{path}={self._clip(self._compact(value), self.MAX_EXTRA_LEN)}" for path, value in pending[: self.MAX_EXTRAS]]

        if len(pending) > self.MAX_EXTRAS:
            parts.append(f"+{len(pending) - self.MAX_EXTRAS} more overrides")
        return parts

    def _truthy(self, value: str) -> bool:
        return value.strip().lower() in ("true", "1", "yes", "on")

    def _is_set(self, value: str) -> bool:
        return value.strip() not in self.UNSET_VALUES

    def _compact(self, value: str) -> str:
        return value.replace("'", "").replace('"', "").strip()

    def _tail(self, value: str) -> str:
        return self._compact(value).rstrip("/").rsplit("/", 1)[-1]

    def _join(self, values: dict, first: str, second: str) -> str:
        left  = values.get(first, "")
        right = values.get(second, "")
        return f"{left}-{right}" if left and right else left or right

    def _labelled(self, label: str, value: str) -> str:
        return f"{label} {value}" if label else value

    def _clip(self, text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def describe(self, key: str, interpreter: str, overrides: dict | None) -> str:
        overrides = dict(overrides or {})
        values    = self._values(key, interpreter, overrides)
        used      = set()

        parts  = self._lead(key, values, used)
        parts += self._details(key, values, used)
        parts += self._extras(overrides, used)

        return self._clip(" · ".join(parts), self.MAX_LENGTH)
