"""Migrates run directories from legacy naming generations to the current run tag scheme.

Each run's name is rebuilt from its own persisted metadata (run_summary, model_config or
dual_model_config, trainer_config, dataset_creation_config, and the profile/image
autoencoder configs for JEPA runs), matched against the legacy naming generations the
directory could have been written under, and renamed. Files that embed the old name
(scheduler result JSONs, trainer/entry configs, pipeline result files) are patched at word
boundaries in the same pass. Renames are only planned by default; ``--apply`` performs them.
"""

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

from configuration.dataset                     import AugmentationConfig
from configuration.training                    import EmbeddingLossConfig, LossConfig, ParamMatching
from configuration.training.jepa               import JepaMaskingConfig
from models                                    import BACKBONE_MODEL_REGISTRY
from models.dual                               import DUAL_CONFIG_REGISTRY
from pipelines.dual.training.pipeline          import TrunkChannelMap
from pipelines.shared.inference.run_classifier import RunArtifacts
from pipelines.shared.training.run_naming      import JepaNamingSpec, RunNaming
from tools.monitoring.logger                   import Logger


class LegacyNaming:
    """Rebuilds the run tags of the retired naming generations from a run's configuration.

    Three generations are covered: the underscore naming (``model_head_matching``), the
    presence-letter naming (``A``/``B``/``F`` flags) and the flagless naming (matching tag
    with no presence letters). A candidate directory name is recognised by matching it
    against every generation's tag.
    """

    MATCHING_ALIASES = {"sort_gt_by_mu": "sorted_gt", "hungarian_active": "hungarian", "hungarian": "hungarian", "sorted_gt": "sorted_gt"}

    @classmethod
    def normalize_matching(cls, raw) -> str | None:
        """Maps a persisted matching value or enum repr onto a canonical matching token.

        Args:
            raw : Matching strategy as persisted, either a bare token or an enum repr such as ``ParamMatching.HUNGARIAN``.

        Returns:
            The canonical token (``hungarian`` or ``sorted_gt``), or None when the value
            names no known strategy.
        """
        token = str(raw).split(".")[-1].lower()
        return cls.MATCHING_ALIASES.get(token)

    @staticmethod
    def run_prefix(model: str) -> str:
        """Returns the oldest generation's run tag, a bare ``run_<model>`` prefix."""
        return f"run_{model}"

    @staticmethod
    def underscore_stem(model: str, head: str, matching: str) -> str:
        """Returns the unit stem of the underscore naming generation."""
        return f"{model}_{head}_{matching}"

    @classmethod
    def underscore_tag(cls, model: str, head: str, matching: str, loss: LossConfig) -> str:
        """Returns the full underscore-generation tag, the stem plus the enabled loss term names."""
        names = [term.name for term in RunNaming.NAMING_ORDER if getattr(loss, term.use_flag)]
        return f"{cls.underscore_stem(model, head, matching)}_{'-'.join(names)}"

    @staticmethod
    def presence_letters(loss: LossConfig) -> str:
        """Returns the A/B/F presence flag letters of the presence naming generation, or "none"."""
        letters = ("A" if loss.use_active_normalization else "") + ("B" if loss.presence_balance else "") + ("F" if loss.amp_focal_gamma > 0.0 else "")
        return letters or "none"

    @classmethod
    def presence_stem(cls, model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        """Returns the unit stem of the presence-letter naming generation."""
        return "-".join((model, head, RunNaming.gaussians_tag(n_gaussians), RunNaming.augmentation_tag(augmentation), cls.presence_letters(loss)))

    @classmethod
    def presence_tag(cls, model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        """Returns the presence-generation tag, the stem plus per-term loss weights."""
        weights = "_".join(f"{term.name}_{getattr(loss, term.weight_key):g}" for term in RunNaming.NAMING_ORDER if getattr(loss, term.use_flag))
        return f"{cls.presence_stem(model, head, loss, n_gaussians, augmentation)}-{weights}"

    @staticmethod
    def flagless_stem(model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        """Returns the unit stem of the flagless naming generation (matching tag, no presence letters)."""
        return "-".join((model, head, RunNaming.matching_tag(loss), RunNaming.gaussians_tag(n_gaussians), RunNaming.augmentation_tag(augmentation)))

    @classmethod
    def flagless_tag(cls, model: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        """Returns the flagless-generation tag, the stem plus the loss tag."""
        return f"{cls.flagless_stem(model, head, loss, n_gaussians, augmentation)}-{RunNaming.loss_tag(loss)}"


class RunRenamer:
    """Plans and applies the migration of run directories to the current naming scheme.

    Roots are walked until training runs (or seed-fanout parents) are found, runs are
    grouped by their parent directory, and each group is renamed as a unit so that name
    references inside sibling bookkeeping files can be patched consistently.

    Attributes:
        roots  : Directories to search for runs; may be run dirs or trees holding them.
        apply  : When False the plan is only logged, when True the renames are performed.
        logger : Logger writing to ``logs/`` under the name ``rename_runs``.
    """

    PATCH_FILES    = (Path("meta") / "run_summary.json", Path("docs") / "trainer_config.json", Path("docs") / "resolved_entry_config.json")
    RESULT_FILES   = ("training_results.json", "inference_results.json")
    RUNS_SUBDIRS   = ("training", "folds")
    SCHEDULER_DIR  = Path("batch_train_logs")
    SCHEDULER_GLOB = "*_results.json"
    SEED_DIR       = re.compile(r"seed\d+")
    NAME_CHAR      = r"[A-Za-z0-9_.\-]"

    def __init__(self, roots: list[Path], apply: bool) -> None:
        """Stores the search roots and the apply switch, and opens the migration logger.

        Args:
            roots : Run directories or trees holding them at any depth.
            apply : Whether to perform the renames instead of only printing the plan.
        """
        self.roots  = [Path(root) for root in roots]
        self.apply  = apply
        self.logger = Logger(log_dir="logs", name="rename_runs")

    def _process_root(self, root: Path) -> None:
        """Discovers the runs under one root and processes them group by group.

        Args:
            root : Directory to search.

        Raises:
            FileNotFoundError : If the path is not a directory, or holds no training run and no run collection.
        """
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")

        groups = {}
        self._discover(root, groups)

        if not groups:
            raise FileNotFoundError(f"{root} holds no training runs (*/meta/run_summary.json) and no run collections (*/training, */folds)")

        for parent in sorted(groups):
            self._process_runs(parent, groups[parent])

    def _discover(self, directory: Path, groups: dict[Path, list[Path]]) -> None:
        """Recursively collects run directories into ``groups``, keyed by their parent.

        A directory is a run when it holds ``meta/run_summary.json`` or when it holds seed
        subdirectories that do; otherwise its ``training``/``folds`` collections, or all of
        its subdirectories, are descended into.

        Args:
            directory : Directory currently being inspected.
            groups    : Accumulator mapping a parent directory to the runs found beneath it.
        """
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
        """Plans, validates and applies the renames for the runs sharing one parent.

        Args:
            parent   : Directory holding the runs.
            run_dirs : Run directories to rename.
        """
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
        """Returns the banner label for the current mode, APPLY or DRY RUN."""
        return "APPLY" if self.apply else "DRY RUN"

    def _plan(self, run_dirs: list[Path]) -> dict[str, str]:
        """Returns the old-name to new-name mapping for the runs that need renaming."""
        mapping = {}
        for run_dir in run_dirs:
            new_name = self._plan_run(run_dir)
            if new_name is not None:
                mapping[run_dir.name] = new_name

        return mapping

    def _plan_run(self, run_dir: Path) -> str | None:
        """Rebuilds the current-scheme name of one backbone run from its persisted metadata.

        Dual and JEPA runs are delegated to their own planners. The rebuilt name is only
        returned when the directory's present name matches one of the known legacy naming
        generations, either as a whole tag or as the ``<stem>__<unit>`` prefix.

        Args:
            run_dir : Run directory to plan.

        Returns:
            The new directory name, or None when the run is already renamed, is not a
            training run, or matches no known naming generation.
        """
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

        if "embedding_loss" in trainer:
            return self._plan_jepa_run(run_dir, metadata_dir, summary, model_config, trainer, dataset)

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
        """Rebuilds the name of a dual-trunk run, adding the trunk and input routing tags.

        Args:
            run_dir      : Run directory to plan.
            metadata_dir : Directory holding the run's ``meta``/``docs`` metadata, which is the first seed subdirectory for a seed fanout.
            summary      : Parsed ``meta/run_summary.json``.

        Returns:
            The new directory name, or None when metadata is missing, the trunk inputs use
            retired channel groups, or the name matches no known dual naming generation.
        """
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

    def _plan_jepa_run(self, run_dir: Path, metadata_dir: Path, summary: dict, model_config: dict, trainer: dict, dataset: dict) -> str | None:
        """Rebuilds the name of a JEPA run, adding the pae_/iae_ autoencoder tags.

        Args:
            run_dir      : Run directory to plan.
            metadata_dir : Directory holding the run's metadata.
            summary      : Parsed ``meta/run_summary.json``.
            model_config : Parsed ``meta/model_config.json``.
            trainer      : Parsed ``docs/trainer_config.json``.
            dataset      : Parsed ``meta/dataset_creation_config.json``.

        Returns:
            The new directory name, or None when the param loss, the matching strategy or a
            coupled autoencoder config cannot be recovered, or the name matches no known
            JEPA naming generation.

        Raises:
            ValueError : If the persisted trainer config predates the input-masking block and therefore cannot produce the current run tag.
        """
        payload = trainer.get("param_loss")
        if payload is None:
            self.logger.warning(f"skip {run_dir.name}: JEPA trainer config has no param_loss")
            return None

        matching = self._matching(payload, summary)
        if matching is None:
            self.logger.warning(f"skip {run_dir.name}: no recognizable matching strategy in trainer config or run summary")
            return None

        profile = self._read_json(metadata_dir / "meta" / RunArtifacts.PROFILE_AE_CONFIG)
        image   = self._read_json(metadata_dir / "meta" / RunArtifacts.IMAGE_AE_CONFIG)

        if trainer.get("autoencoder") is not None and profile is None:
            self.logger.warning(f"skip {run_dir.name}: a profile autoencoder was coupled but meta/profile_autoencoder_config.json is missing; the AE architecture name cannot be recovered")
            return None

        if trainer.get("image_autoencoder") is not None and image is None:
            self.logger.warning(f"skip {run_dir.name}: an image autoencoder was coupled but meta/image_autoencoder_config.json is missing; the AE architecture name cannot be recovered")
            return None

        if profile is None and image is None:
            self.logger.warning(f"skip {run_dir.name}: JEPA run persists neither a profile nor an image autoencoder config")
            return None

        if "masking" not in trainer:
            raise ValueError(f"{run_dir.name}: JEPA trainer config predates the input-masking block, so its run tag cannot be rebuilt. Regenerate the run or drop it before renaming.")

        model        = summary["model_name"]
        head         = model_config["config"].get("head", "conv")
        loss         = self._loss_config(payload, matching)
        augmentation = self._augmentation(dataset["augmentation"])
        n_gaussians  = trainer["gaussian"]["n_default_gaussians"]

        naming = JepaNamingSpec(
            profile_ae      = profile["model_name"] if profile else None,
            profile_mode    = trainer["profile_autoencoder_mode"],
            target_provider = trainer["target_provider"],
            image_ae        = image["model_name"] if image else None,
            image_mode      = trainer["image_autoencoder_mode"],
            embedding_loss  = self._embedding_loss(trainer["embedding_loss"]),
            param_loss      = loss,
            masking         = JepaMaskingConfig(**trainer["masking"]),
        )

        new_tag = RunNaming.jepa_tag(model, head, naming, n_gaussians, augmentation)

        name = run_dir.name
        if name == new_tag or name.startswith(f"{new_tag}_"):
            self.logger.info(f"skip {name}: already in the new naming")
            return None

        old_tags = (RunNaming.tag(model, head, loss, n_gaussians, augmentation), LegacyNaming.flagless_tag(model, head, loss, n_gaussians, augmentation), LegacyNaming.presence_tag(model, head, loss, n_gaussians, augmentation), LegacyNaming.underscore_tag(model, head, matching, loss), LegacyNaming.run_prefix(model))
        for old_tag in old_tags:
            if name == old_tag:
                return new_tag

            if name.startswith(f"{old_tag}_"):
                return f"{new_tag}{name[len(old_tag):]}"

        self.logger.warning(f"skip {name}: matches no known JEPA naming generation {old_tags}")
        return None

    def _metadata_dir(self, run_dir: Path) -> Path:
        """Returns the directory holding the run's metadata, the first seed dir for a fanout."""
        if (run_dir / "meta" / "run_summary.json").is_file():
            return run_dir

        seed_runs = self._seed_children(run_dir)
        return seed_runs[0] if seed_runs else run_dir

    def _read_json(self, path: Path) -> dict | None:
        """Returns the parsed JSON at ``path``, or None when the file does not exist."""
        if not path.is_file():
            return None

        return json.loads(path.read_text())

    def _loss_payload(self, trainer: dict) -> dict | None:
        """Returns the loss section the name is built from: the complete curriculum stage, else param_loss."""
        if "curriculum" in trainer:
            return trainer["curriculum"]["complete"]

        return trainer.get("param_loss")

    def _matching(self, payload: dict, summary: dict) -> str | None:
        """Returns the canonical matching token, read from the loss payload or the run summary."""
        raw = payload.get("param_matching") or payload.get("param_match") or summary.get("param_match")
        if raw is None:
            return None

        return LegacyNaming.normalize_matching(raw)

    def _loss_config(self, payload: dict, matching: str) -> LossConfig:
        """Rebuilds a LossConfig from a persisted loss payload, keeping only fields it still declares.

        Args:
            payload  : Persisted loss section from the trainer config.
            matching : Canonical matching token to install on the rebuilt config.

        Returns:
            The reconstructed loss configuration.
        """
        known  = {spec.name for spec in fields(LossConfig)}
        config = LossConfig(**{key: value for key, value in payload.items() if key in known and key != "param_matching"})

        config.param_matching = ParamMatching(matching)
        return config

    def _augmentation(self, payload: dict) -> AugmentationConfig:
        """Rebuilds an AugmentationConfig from a persisted payload, dropping retired fields."""
        known = {spec.name for spec in fields(AugmentationConfig)}
        return AugmentationConfig(**{key: value for key, value in payload.items() if key in known})

    def _embedding_loss(self, payload: dict) -> EmbeddingLossConfig:
        """Rebuilds an EmbeddingLossConfig from a persisted payload, dropping retired fields."""
        known = {spec.name for spec in fields(EmbeddingLossConfig)}
        return EmbeddingLossConfig(**{key: value for key, value in payload.items() if key in known})

    def _check_targets(self, parent: Path, mapping: dict[str, str]) -> None:
        """Refuses a plan whose targets collide with each other or with existing directories.

        Args:
            parent  : Directory holding the runs.
            mapping : Old-name to new-name plan.

        Raises:
            ValueError      : If two runs rebuild to the same name.
            FileExistsError : If a target name is already taken in ``parent``.
        """
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
        """Patches each run's own metadata files with the new names, then renames the directory.

        Args:
            parent  : Directory holding the runs.
            mapping : Old-name to new-name plan; logged only when ``apply`` is False.
        """
        for old, new in mapping.items():
            source = parent / old

            self.logger.info(f"{old} -> {new}")
            if not self.apply:
                continue

            for holder in (source, *self._seed_children(source)):
                for relative in self.PATCH_FILES:
                    self._patch_file(holder / relative, mapping)
            self._patch_scheduler_files(source, mapping)

            source.rename(parent / new)

    def _patch_scheduler_files(self, holder: Path, mapping: dict[str, str]) -> None:
        """Rewrites the run names inside every scheduler result JSON under ``holder``."""
        directory = holder / self.SCHEDULER_DIR
        if not directory.is_dir():
            return

        for path in sorted(directory.glob(self.SCHEDULER_GLOB)):
            self._patch_file(path, mapping)

    def _seed_children(self, directory: Path) -> list[Path]:
        """Returns the ``seed<N>`` subdirectories of ``directory`` that hold a run summary."""
        if not directory.is_dir():
            return []

        return sorted(child for child in directory.iterdir() if child.is_dir() and self.SEED_DIR.fullmatch(child.name) and (child / "meta" / "run_summary.json").is_file())

    def _patch_file(self, path: Path, mapping: dict[str, str]) -> None:
        """Rewrites the run names inside one file in place, when the file exists and changes."""
        if not path.exists():
            return

        original = path.read_text()
        patched  = self._boundary_replace(original, mapping)
        if patched != original:
            path.write_text(patched)
            self.logger.info(f"patched {path}")

    def _boundary_replace(self, text: str, mapping: dict[str, str]) -> str:
        """Replaces each old name in ``text`` at name boundaries, longest name first.

        Args:
            text    : File content to rewrite.
            mapping : Old-name to new-name plan.

        Returns:
            The rewritten text; occurrences embedded inside a longer run name are left alone.
        """
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            text = re.sub(rf"(?<!{self.NAME_CHAR}){re.escape(old)}(?!{self.NAME_CHAR})", new, text)
        return text

    def run(self) -> None:
        """Processes every root in turn and closes the logger."""
        for root in self.roots:
            self._process_root(root)

        if not self.apply:
            self.logger.info("dry run only; re-run with --apply to rename")

        self.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename existing runs to the model-head-matching-K_N-aug-presence-loss_weight naming (dual runs gain the dual_<trunks> tag and the params.existence routing tag; JEPA runs gain pae_/iae_ autoencoder tags and carry the loss they actually train on), rebuilding each name from the run's persisted metadata (run_summary, model_config or dual_model_config, trainer_config, dataset_creation_config, profile/image autoencoder configs)")
    parser.add_argument("roots", nargs="+", type=Path, help="any mix of run directories and roots holding them at any depth: training/trial roots, benchmark or cross-validation run tags (training/ or folds/), tuning trial trees, or whole runs roots")
    parser.add_argument("--apply", action="store_true", help="perform the renames; without it the script only prints the plan")
    args = parser.parse_args()

    RunRenamer(args.roots, args.apply).run()
