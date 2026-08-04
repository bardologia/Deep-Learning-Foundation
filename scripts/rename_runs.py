from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib     import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.dataset                import AugmentationConfig
from configuration.training               import LossConfig, ParamMatching
from models                               import BACKBONE_MODEL_REGISTRY
from models.dual                          import DUAL_CONFIG_REGISTRY
from pipelines.dual.training.pipeline     import TrunkChannelMap
from pipelines.shared.training.run_naming import RunNaming
from tools.monitoring.logger              import Logger


class LegacyNaming:

    MATCHING_ALIASES = {"sort_gt_by_mu": "sorted_gt", "hungarian_active": "hungarian", "hungarian": "hungarian", "sorted_gt": "sorted_gt"}

    @classmethod
    def normalize_matching(cls, raw) -> str | None:
        token = str(raw).split(".")[-1].lower()
        return cls.MATCHING_ALIASES.get(token)

    @staticmethod
    def run_prefix(model: str) -> str:
        return f"run_{model}"

    @staticmethod
    def underscore_stem(model: str, head: str, matching: str) -> str:
        return f"{model}_{head}_{matching}"

    @classmethod
    def underscore_tag(cls, model: str, head: str, matching: str, loss: LossConfig) -> str:
        names = [term.name for term in RunNaming.NAMING_ORDER if getattr(loss, term.use_flag)]
        return f"{cls.underscore_stem(model, head, matching)}_{'-'.join(names)}"

    @staticmethod
    def presence_letters(loss: LossConfig) -> str:
        letters = ("A" if loss.use_active_normalization else "") + ("B" if loss.presence_balance else "") + ("F" if loss.amp_focal_gamma > 0.0 else "")
        return letters or "none"

    @classmethod
    def presence_stem(cls, model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        return "-".join((model, head, RunNaming.gaussians_tag(n_gaussians), RunNaming.augmentation_tag(augmentation), cls.presence_letters(loss)))

    @classmethod
    def presence_tag(cls, model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        weights = "_".join(f"{term.name}_{getattr(loss, term.weight_key):g}" for term in RunNaming.NAMING_ORDER if getattr(loss, term.use_flag))
        return f"{cls.presence_stem(model, head, loss, n_gaussians, augmentation)}-{weights}"

    @staticmethod
    def flagless_stem(model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        return "-".join((model, head, RunNaming.matching_tag(loss), RunNaming.gaussians_tag(n_gaussians), RunNaming.augmentation_tag(augmentation)))

    @classmethod
    def flagless_tag(cls, model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        return f"{cls.flagless_stem(model, head, loss, n_gaussians, augmentation)}-{RunNaming.loss_tag(loss)}"


class RunRenamer:

    PATCH_FILES    = (Path("meta") / "run_summary.json", Path("docs") / "trainer_config.json", Path("docs") / "resolved_entry_config.json")
    RESULT_FILES   = ("training_results.json", "inference_results.json")
    RUNS_SUBDIRS   = ("training", "folds")
    SCHEDULER_DIR  = Path("batch_train_logs")
    SCHEDULER_GLOB = "*_results.json"
    SEED_DIR       = re.compile(r"seed\d+")

    def __init__(self, roots: list[Path], apply: bool) -> None:
        self.roots  = [Path(root) for root in roots]
        self.apply  = apply
        self.logger = Logger(log_dir="logs", name="rename_runs")

    def _process_root(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")

        groups = {}
        self._discover(root, groups)

        if not groups:
            raise FileNotFoundError(f"{root} holds no training runs (*/meta/run_summary.json) and no run collections (*/training, */folds)")

        for parent in sorted(groups):
            self._process_runs(parent, groups[parent])

    def _discover(self, directory: Path, groups: dict[Path, list[Path]]) -> None:
        if (directory / "meta" / "run_summary.json").is_file():
            groups.setdefault(directory.parent, []).append(directory)
            return

        if self._seed_children(directory):
            groups.setdefault(directory.parent, []).append(directory)
            return

        collections = [directory / sub for sub in self.RUNS_SUBDIRS if (directory / sub).is_dir()]
        if collections:
            for collection in collections:
                self._discover(collection, groups)
            return

        for child in sorted(directory.iterdir()):
            if child.is_dir():
                self._discover(child, groups)

    def _process_runs(self, parent: Path, run_dirs: list[Path]) -> None:
        self.logger.section(f"[{self._mode()}] {parent}")

        mapping = self._plan(run_dirs)
        if not mapping:
            self.logger.info("nothing to rename")
            return

        self._check_targets(parent, mapping)

        if self.apply:
            self._patch_scheduler_files(parent, mapping)

            if parent.name in self.RUNS_SUBDIRS:
                for name in self.RESULT_FILES:
                    self._patch_file(parent.parent / "pipeline" / name, mapping)

        self._rename_runs(parent, mapping)

    def _mode(self) -> str:
        return "APPLY" if self.apply else "DRY RUN"

    def _plan(self, run_dirs: list[Path]) -> dict[str, str]:
        mapping = {}
        for run_dir in run_dirs:
            new_name = self._plan_run(run_dir)
            if new_name is not None:
                mapping[run_dir.name] = new_name

        return mapping

    def _plan_run(self, run_dir: Path) -> str | None:
        metadata_dir = self._metadata_dir(run_dir)
        summary      = self._read_json(metadata_dir / "meta" / "run_summary.json")
        if summary is None:
            self.logger.info(f"skip {run_dir.name}: no meta/run_summary.json (not a training run)")
            return None

        model = summary["model_name"]
        if model in DUAL_CONFIG_REGISTRY:
            return self._plan_dual_run(run_dir, metadata_dir, summary)

        if model not in BACKBONE_MODEL_REGISTRY:
            self.logger.info(f"skip {run_dir.name}: model '{model}' is neither a backbone nor a dual model (autoencoder or foreign run)")
            return None

        model_config = self._read_json(metadata_dir / "meta" / "model_config.json")
        trainer      = self._read_json(metadata_dir / "docs" / "trainer_config.json")
        dataset      = self._read_json(metadata_dir / "meta" / "dataset_creation_config.json")
        if model_config is None or trainer is None or dataset is None:
            self.logger.warning(f"skip {run_dir.name}: missing model_config/trainer_config/dataset_creation_config metadata")
            return None

        payload = self._loss_payload(trainer)
        if payload is None:
            self.logger.warning(f"skip {run_dir.name}: trainer config has neither a loss curriculum nor a param_loss")
            return None

        matching = self._matching(payload, summary)
        if matching is None:
            self.logger.warning(f"skip {run_dir.name}: no recognizable matching strategy in trainer config or run summary")
            return None

        head         = model_config["config"].get("head", "conv")
        loss         = self._loss_config(payload, matching)
        augmentation = self._augmentation(dataset["augmentation"])
        n_gaussians  = trainer["gaussian"]["n_default_gaussians"]

        new_stem = RunNaming.stem(model, head, loss, n_gaussians, augmentation)
        new_tag  = RunNaming.tag(model, head, loss, n_gaussians, augmentation)

        name = run_dir.name
        if name == new_tag or name.startswith(f"{new_tag}_") or name.startswith(f"{new_stem}__"):
            self.logger.info(f"skip {name}: already in the new naming")
            return None

        if "__" in name:
            base, component = name.split("__", 1)
            old_stems       = (LegacyNaming.flagless_stem(model, head, loss, n_gaussians, augmentation), LegacyNaming.presence_stem(model, head, loss, n_gaussians, augmentation), LegacyNaming.underscore_stem(model, head, matching), model)
            if base in old_stems:
                return f"{new_stem}__{component}"

            self.logger.warning(f"skip {name}: unit prefix '{base}' matches no known naming generation {old_stems}")
            return None

        old_tags = (LegacyNaming.flagless_tag(model, head, loss, n_gaussians, augmentation), LegacyNaming.presence_tag(model, head, loss, n_gaussians, augmentation), LegacyNaming.underscore_tag(model, head, matching, loss), LegacyNaming.run_prefix(model))
        for old_tag in old_tags:
            if name == old_tag:
                return new_tag

            if name.startswith(f"{old_tag}_"):
                return f"{new_tag}{name[len(old_tag):]}"

        self.logger.warning(f"skip {name}: matches no known naming generation {old_tags}")
        return None

    def _plan_dual_run(self, run_dir: Path, metadata_dir: Path, summary: dict) -> str | None:
        model_config = self._read_json(metadata_dir / "meta" / "dual_model_config.json")
        trainer      = self._read_json(metadata_dir / "docs" / "trainer_config.json")
        dataset      = self._read_json(metadata_dir / "meta" / "dataset_creation_config.json")
        if model_config is None or trainer is None or dataset is None:
            self.logger.warning(f"skip {run_dir.name}: missing dual_model_config/trainer_config/dataset_creation_config metadata")
            return None

        payload = self._loss_payload(trainer)
        if payload is None:
            self.logger.warning(f"skip {run_dir.name}: trainer config has neither a loss curriculum nor a param_loss")
            return None

        matching = self._matching(payload, summary)
        if matching is None:
            self.logger.warning(f"skip {run_dir.name}: no recognizable matching strategy in trainer config or run summary")
            return None

        config  = model_config["config"]
        head    = config["head"]
        unknown = sorted((set(config["params_input"]) | set(config["existence_input"])) - set(TrunkChannelMap.GROUPS))
        if unknown:
            self.logger.warning(f"skip {run_dir.name}: trunk inputs use retired channel groups {unknown}; only {list(TrunkChannelMap.GROUPS)} name a routing tag")
            return None

        trunk_tag    = RunNaming.dual_model_tag(config["params_backbone"], config["existence_backbone"])
        input_tag    = f"{TrunkChannelMap.group_label(tuple(config['params_input']))}.{TrunkChannelMap.group_label(tuple(config['existence_input']))}"
        loss         = self._loss_config(payload, matching)
        augmentation = self._augmentation(dataset["augmentation"])
        n_gaussians  = trainer["gaussian"]["n_default_gaussians"]

        new_tag = RunNaming.tag(trunk_tag, head, loss, n_gaussians, augmentation, extras=(input_tag,))

        name = run_dir.name
        if name == new_tag or name.startswith(f"{new_tag}_"):
            self.logger.info(f"skip {name}: already in the new naming")
            return None

        model    = summary["model_name"]
        old_tags = (RunNaming.tag(model, head, loss, n_gaussians, augmentation, extras=(input_tag,)), RunNaming.tag(model, head, loss, n_gaussians, augmentation))
        for old_tag in old_tags:
            if name == old_tag:
                return new_tag

            if name.startswith(f"{old_tag}_"):
                return f"{new_tag}{name[len(old_tag):]}"

        self.logger.warning(f"skip {name}: matches no known dual naming generation {old_tags}")
        return None

    def _metadata_dir(self, run_dir: Path) -> Path:
        if (run_dir / "meta" / "run_summary.json").is_file():
            return run_dir

        seed_runs = self._seed_children(run_dir)
        return seed_runs[0] if seed_runs else run_dir

    def _read_json(self, path: Path) -> dict | None:
        if not path.is_file():
            return None

        return json.loads(path.read_text())

    def _loss_payload(self, trainer: dict) -> dict | None:
        if "curriculum" in trainer:
            return trainer["curriculum"]["complete"]

        return trainer.get("param_loss")

    def _matching(self, payload: dict, summary: dict) -> str | None:
        raw = payload.get("param_matching") or payload.get("param_match") or summary.get("param_match")
        if raw is None:
            return None

        return LegacyNaming.normalize_matching(raw)

    def _loss_config(self, payload: dict, matching: str) -> LossConfig:
        known  = {spec.name for spec in fields(LossConfig)}
        config = LossConfig(**{key: value for key, value in payload.items() if key in known and key != "param_matching"})

        config.param_matching = ParamMatching(matching)
        return config

    def _augmentation(self, payload: dict) -> AugmentationConfig:
        known = {spec.name for spec in fields(AugmentationConfig)}
        return AugmentationConfig(**{key: value for key, value in payload.items() if key in known})

    def _check_targets(self, parent: Path, mapping: dict[str, str]) -> None:
        sources = {}
        for old, new in mapping.items():
            sources.setdefault(new, []).append(old)

        collisions = {new: olds for new, olds in sources.items() if len(olds) > 1}
        if collisions:
            raise ValueError(f"{parent}: several runs rebuild to the same name: " + "; ".join(f"{new} <- {sorted(olds)}" for new, olds in sorted(collisions.items())))

        taken = [new for new in sorted(sources) if (parent / new).exists()]
        if taken:
            raise FileExistsError(f"{parent}: rename targets already exist; refusing to overwrite {taken}")

    def _rename_runs(self, parent: Path, mapping: dict[str, str]) -> None:
        for old, new in mapping.items():
            source = parent / old

            self.logger.info(f"{old} -> {new}")
            if not self.apply:
                continue

            for holder in (source, *self._seed_children(source)):
                for relative in self.PATCH_FILES:
                    self._patch_file(holder / relative, {old: new})
            self._patch_scheduler_files(source, {old: new})

            source.rename(parent / new)

    def _patch_scheduler_files(self, holder: Path, mapping: dict[str, str]) -> None:
        directory = holder / self.SCHEDULER_DIR
        if not directory.is_dir():
            return

        for path in sorted(directory.glob(self.SCHEDULER_GLOB)):
            self._patch_file(path, mapping)

    def _seed_children(self, directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []

        return sorted(child for child in directory.iterdir() if child.is_dir() and self.SEED_DIR.fullmatch(child.name) and (child / "meta" / "run_summary.json").is_file())

    def _patch_file(self, path: Path, mapping: dict[str, str]) -> None:
        if not path.exists():
            return

        original = path.read_text()
        patched  = self._boundary_replace(original, mapping)
        if patched != original:
            path.write_text(patched)
            self.logger.info(f"patched {path}")

    def _boundary_replace(self, text: str, mapping: dict[str, str]) -> str:
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}", new, text)
        return text

    def run(self) -> None:
        for root in self.roots:
            self._process_root(root)

        if not self.apply:
            self.logger.info("dry run only; re-run with --apply to rename")

        self.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename existing runs to the model-head-matching-K_N-aug-presence-loss_weight naming (dual runs gain the dual_<trunks> tag and the params.existence routing tag), rebuilding each name from the run's persisted metadata (run_summary, model_config or dual_model_config, trainer_config, dataset_creation_config)")
    parser.add_argument("roots", nargs="+", type=Path, help="any mix of run directories and roots holding them at any depth: training/trial roots, benchmark or cross-validation run tags (training/ or folds/), tuning trial trees, or whole runs roots")
    parser.add_argument("--apply", action="store_true", help="perform the renames; without it the script only prints the plan")
    args = parser.parse_args()

    RunRenamer(args.roots, args.apply).run()
