"""Fold planning, naming, and configuration for k-fold cross-validation.

Partitions the dataset row extent into disjoint blocks, assigns one test
and one validation block per fold with a guard margin between neighbours, and
derives the run names and per-fold configs each training family needs. Also
reads a finished run's fold tree back into the same plan, family, model name
and scored splits, for reports rebuilt without the run's configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib     import Path

from configuration.cross_validation              import CrossValidationConfig
from configuration.inference                     import InferenceConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceConfig
from configuration.dataset.gaussian           import GaussianConfig
from pipelines.shared.config.config_factory      import ConfigFactory
from pipelines.shared.inference.metadata         import CompletedInference
from pipelines.shared.inference.run_classifier   import RunClassifier, RunType
from pipelines.shared.orchestration.unit_manifest import UnitManifest
from pipelines.shared.training.seed_sweep        import SeedSet
from tools.data.io            import FileIO
from tools.data.regions       import CropRegion, SplitRegions


class FoldFamily:
    """Training families cross-validation supports and their layout differences.

    Spatial families (backbone, dual, JEPA) write inference directly under
    `inference/`, whereas the autoencoder families nest theirs in a
    family-specific subdirectory.
    """

    SPATIAL      = ("backbone", "dual", "jepa")
    AUTOENCODERS = ("profile_autoencoder", "image_autoencoder")
    TYPES        = SPATIAL + AUTOENCODERS

    INFERENCE_SUBDIR = {
        "profile_autoencoder" : RunType.PROFILE_AE,
        "image_autoencoder"   : RunType.IMAGE_AE,
    }

    @classmethod
    def validate(cls, training_type: str) -> str:
        """Returns the training type unchanged after checking it is supported.

        Args:
            training_type : Candidate training family name.

        Returns:
            The same training type.

        Raises:
            ValueError : If the family is not one cross-validation supports.
        """
        if training_type not in cls.TYPES:
            raise ValueError(f"Unknown training_type '{training_type}', expected one of {sorted(cls.TYPES)}")

        return training_type

    @classmethod
    def inference_subdir(cls, training_type: str) -> str | None:
        """Returns the inference subdirectory of a family, None for spatial families."""
        cls.validate(training_type)

        return None if training_type in cls.SPATIAL else cls.INFERENCE_SUBDIR[training_type]

    @classmethod
    def inference_base(cls, run_directory: Path, training_type: str) -> Path:
        """Returns the directory a family's inference runs are written under."""
        subdir = cls.inference_subdir(training_type)
        base   = Path(run_directory) / "inference"

        return base / subdir if subdir else base

    @classmethod
    def split_inference_dir(cls, run_directory: Path, training_type: str, split: str) -> Path | None:
        """Returns the latest completed inference directory for a split, or None."""
        return CompletedInference.latest(cls.inference_base(run_directory, training_type), split)

    @classmethod
    def model_label(cls, config: CrossValidationConfig) -> str:
        """Returns the model name to report for the configured training family.

        Args:
            config : Cross-validation configuration naming the family and model.

        Returns:
            The backbone name, the autoencoder model name, or the combined dual
            trunk tag, depending on the family.
        """
        from pipelines.shared.training.run_naming import RunNaming

        training_type = cls.validate(config.training_type)

        if training_type == "dual":
            return RunNaming.dual_model_tag(config.dual.params_backbone, config.dual.existence_backbone)

        if training_type == "profile_autoencoder":
            return config.autoencoder.ae_model_name

        if training_type == "image_autoencoder":
            return config.image_autoencoder.ae_model_name

        return config.backbone_name


@dataclass
class FoldPlan:
    """Block assignment and crop regions of a single fold.

    Attributes:
        fold_index    : Zero-based index of the fold.
        test_block    : Index of the row block held out for testing.
        val_block     : Index of the row block used for validation.
        train_blocks  : Indices of the remaining blocks used for training.
        split_regions : Guard-trimmed crop regions for train, val, and test.
    """

    fold_index    : int
    test_block    : int
    val_block     : int
    train_blocks  : list[int]
    split_regions : SplitRegions


class FoldPlanner:
    """Partitions the row extent into fold blocks and derives crop regions.

    Attributes:
        n_folds       : Number of row blocks, and therefore folds.
        guard         : Total guard width in row lines removed between neighbouring blocks; half is trimmed off each interior edge.
        row_start : First row line covered by the folds.
        row_end   : Exclusive last row line covered by the folds.
        col_start   : First col column of every crop region.
        col_end     : Exclusive last col column of every crop region.
        blocks        : (start, end) row bounds of each block before guard trimming.
    """

    INFERENCE_SPLITS = ("test", "val")

    def __init__(self, config: CrossValidationConfig, col_start: int, col_end: int) -> None:
        """Partitions the configured row window into fold blocks.

        Args:
            config      : Cross-validation configuration carrying the folds section.
            col_start : First col column of every crop region.
            col_end   : Exclusive last col column of every crop region.

        Raises:
            ValueError : If fewer than three folds are requested, the guard is negative or odd, the row extent is smaller than the fold count, or the guard consumes the smallest block.
        """
        folds = config.folds

        if folds.n_folds < 3:
            raise ValueError(f"n_folds must be >= 3 so that train, val, and test stay disjoint; got {folds.n_folds}")

        if folds.guard < 0 or folds.guard % 2 != 0:
            raise ValueError(f"folds.guard={folds.guard} must be a non-negative even number of row lines")

        self.n_folds   = folds.n_folds
        self.guard     = folds.guard
        self.row_start = folds.row_start
        self.row_end   = folds.row_end
        self.col_start = col_start
        self.col_end   = col_end
        self.blocks    = self._partition(folds.row_start, folds.row_end, folds.n_folds)

        smallest_block = min(end - start for start, end in self.blocks)
        if smallest_block <= self.guard:
            raise ValueError(f"folds.guard={self.guard} consumes the smallest fold block ({smallest_block} row lines); reduce guard or n_folds")

    def _partition(self, row_start: int, row_end: int, n_folds: int) -> list[tuple[int, int]]:
        """Splits an row window into contiguous blocks of near-equal size.

        Args:
            row_start : First row line.
            row_end   : Exclusive last row line.
            n_folds       : Number of blocks to produce; the last absorbs the remainder.

        Returns:
            (start, end) row bounds of each block.

        Raises:
            ValueError : If the extent holds fewer lines than folds.
        """
        total = row_end - row_start
        if total < n_folds:
            raise ValueError(f"Row extent {total} is smaller than n_folds {n_folds}")

        size   = total // n_folds
        bounds = [row_start + index * size for index in range(n_folds)] + [row_end]

        return [(bounds[index], bounds[index + 1]) for index in range(n_folds)]

    def _merge_adjacent(self, block_indices: list[int]) -> list[tuple[int, int]]:
        """Collapses consecutive block indices into (first, last) runs."""
        runs  = []
        start = block_indices[0]
        prev  = block_indices[0]

        for index in block_indices[1:]:
            if index == prev + 1:
                prev = index
                continue
            runs.append((start, prev))
            start = index
            prev  = index

        runs.append((start, prev))
        return runs

    def _run_region(self, run: tuple[int, int]) -> CropRegion:
        """Returns the guard-trimmed crop region spanning a run of blocks."""
        first_block, last_block = run
        row_start, row_end = self._trim(self.blocks[first_block][0], self.blocks[last_block][1])
        return CropRegion(row_start, row_end, self.col_start, self.col_end)

    def _block_region(self, block_index: int) -> CropRegion:
        """Returns the guard-trimmed crop region of a single block."""
        row_start, row_end = self._trim(*self.blocks[block_index])
        return CropRegion(row_start, row_end, self.col_start, self.col_end)

    def _trim(self, row_start: int, row_end: int) -> tuple[int, int]:
        """Removes half the guard from each interior edge of an row span.

        Args:
            row_start : First row line of the span.
            row_end   : Exclusive last row line of the span.

        Returns:
            The trimmed bounds; edges on the outer boundary of the fold window
            are left untouched.
        """
        margin        = self.guard // 2
        row_start = row_start + margin if row_start > self.row_start else row_start
        row_end   = row_end - margin   if row_end   < self.row_end   else row_end
        return row_start, row_end

    def inference_region(self, fold_index: int, split: str) -> CropRegion:
        """Returns the untrimmed crop region a fold runs inference over.

        Inference deliberately extends across the guard margins so the fold
        mosaics tile without gaps.

        Args:
            fold_index : Zero-based index of the fold.
            split      : Either 'test' or 'val'.

        Returns:
            The full row block of the requested split.

        Raises:
            ValueError : If the split holds no fold-specific region.
        """
        plan   = self.plan(fold_index)
        blocks = {"test": plan.test_block, "val": plan.val_block}

        if split not in self.INFERENCE_SPLITS:
            raise ValueError(f"inference regions exist for splits {sorted(self.INFERENCE_SPLITS)}; got '{split}'")

        row_start, row_end = self.blocks[blocks[split]]
        return CropRegion(row_start, row_end, self.col_start, self.col_end)

    def plan(self, fold_index: int) -> FoldPlan:
        """Builds the block assignment and crop regions of one fold.

        The fold's own block is the test set, the next block cyclically is the
        validation set, and the rest are merged into contiguous training regions.

        Args:
            fold_index : Zero-based index of the fold.

        Returns:
            The fold plan.

        Raises:
            ValueError : If the index is outside the fold range.
        """
        if not 0 <= fold_index < self.n_folds:
            raise ValueError(f"fold_index must be in [0, {self.n_folds}); got {fold_index}")

        test_block   = fold_index
        val_block    = (fold_index + 1) % self.n_folds
        train_blocks = [index for index in range(self.n_folds) if index not in (test_block, val_block)]

        train_regions = [self._run_region(run) for run in self._merge_adjacent(train_blocks)]

        split_regions = SplitRegions(
            train = train_regions if len(train_regions) > 1 else train_regions[0],
            val   = self._block_region(val_block),
            test  = self._block_region(test_block),
        )

        return FoldPlan(
            fold_index    = fold_index,
            test_block    = test_block,
            val_block     = val_block,
            train_blocks  = train_blocks,
            split_regions = split_regions,
        )

    def plans(self) -> list[FoldPlan]:
        """Returns the plan of every fold, in fold order."""
        return [self.plan(fold_index) for fold_index in range(self.n_folds)]


class StoredFoldPlanner:
    """Fold plan read back from the split regions the fold runs stored.

    Stands in for FoldPlanner where only the plan is needed, such as the
    cross-fold report of a run whose configuration is not at hand: the regions
    come from each fold run's dataset creation config rather than from a fold
    window and guard, and the block assignment follows the planner's
    convention, the fold's own block for testing and the next one cyclically
    for validation.

    Attributes:
        n_folds : Number of folds the regions were stored for.
        regions : Split regions of every fold, keyed by fold index.
    """

    def __init__(self, regions: dict[int, SplitRegions]) -> None:
        """Validates that the stored folds are indexed contiguously from zero.

        Args:
            regions : Split regions keyed by fold index.

        Raises:
            ValueError : If no fold was given or the indices do not run from zero without gaps.
        """
        if not regions or sorted(regions) != list(range(len(regions))):
            raise ValueError(f"Stored folds must be indexed contiguously from 0; got {sorted(regions)}")

        self.n_folds = len(regions)
        self.regions = dict(regions)

    def plan(self, fold_index: int) -> FoldPlan:
        """Returns the plan of one fold from its stored regions.

        Args:
            fold_index : Zero-based index of the fold.

        Raises:
            ValueError : If the index is outside the fold range.
        """
        if fold_index not in self.regions:
            raise ValueError(f"fold_index must be in [0, {self.n_folds}); got {fold_index}")

        test_block   = fold_index
        val_block    = (fold_index + 1) % self.n_folds
        train_blocks = [index for index in range(self.n_folds) if index not in (test_block, val_block)]

        return FoldPlan(fold_index=fold_index, test_block=test_block, val_block=val_block, train_blocks=train_blocks, split_regions=self.regions[fold_index])

    def plans(self) -> list[FoldPlan]:
        """Returns the plan of every fold, in fold order."""
        return [self.plan(fold_index) for fold_index in range(self.n_folds)]


class FoldNaming:
    """Encodes and decodes the `fold_N[/seedS]` run-name convention.

    Attributes:
        PATTERN : Pattern a fold directory name ends with, capturing the fold index.
    """

    PATTERN = re.compile(r"(?:^|_)fold_(\d+)$")

    @staticmethod
    def is_fold(name: str) -> bool:
        """Returns whether a directory name is a fold run name, tagged or bare."""
        return FoldNaming.PATTERN.search(name) is not None

    @staticmethod
    def name(index: int) -> str:
        """Returns the seedless run name of a fold."""
        return f"fold_{index}"

    @staticmethod
    def run_name(index: int, seed: int | None) -> str:
        """Returns the fold run name, suffixed with the seed when one is given."""
        base = FoldNaming.name(index)
        return base if seed is None else SeedSet.run_name(base, seed)

    @staticmethod
    def base(name: str) -> str:
        """Returns the run name with any seed suffix stripped."""
        return name.split("/seed")[0]

    @staticmethod
    def index(name: str) -> int:
        """Returns the fold index encoded in a run name."""
        return int(FoldNaming.base(name).split("_")[-1])

    @staticmethod
    def seed(name: str) -> int | None:
        """Returns the seed encoded in a run name, or None when unseeded."""
        parts = name.split("/seed")
        return int(parts[1]) if len(parts) == 2 else None


class FoldUnitManifest:
    """Identity of one fold x seed unit, as the fold worker records it in the unit's run directory.

    The manifest is what a resumed run and the generic inference entries read
    instead of the directory name: the fold index and seed identify the unit,
    and the inference regions carry the untrimmed row block each scored
    split covers.

    Attributes:
        KIND : Manifest kind every cross-validation unit is written with.
    """

    KIND = "cross_validation"

    @staticmethod
    def identity(manifest: dict) -> tuple:
        """Returns the (fold index, seed) pair a manifest identifies, the seed None for an unseeded unit."""
        return (int(manifest["fold_index"]), manifest["seed"])

    @classmethod
    def write(cls, run_directory: Path, run_tag: str, unit: str, fold_index: int, seed: int | None, training_type: str, inference_regions: dict) -> Path:
        """Writes the manifest of one unit before its training starts.

        Args:
            run_directory     : Run directory of the unit.
            run_tag           : Tag of the cross-validation run the unit belongs to.
            unit              : Run name of the unit, relative to the folds directory.
            fold_index        : Zero-based index of the fold.
            seed              : Seed of the unit, or None when the sweep has no seed dimension.
            training_type     : Training family of the run.
            inference_regions : Untrimmed crop region per scored split, each as (row_start, row_end, col_start, col_end).

        Returns:
            Path of the written manifest.
        """
        payload = {
            "run_tag"           : run_tag,
            "unit"              : unit,
            "fold_index"        : int(fold_index),
            "seed"              : seed,
            "training_type"     : training_type,
            "inference_regions" : {split: list(region) for split, region in inference_regions.items()},
        }

        return UnitManifest.write(run_directory, cls.KIND, payload)

    @classmethod
    def inference_region(cls, run_directory: Path, split: str) -> CropRegion | None:
        """Returns the crop region a fold unit scores one split over, or None for a run that is no fold unit.

        Args:
            run_directory : Run directory to inspect.
            split         : Dataset split being scored.

        Raises:
            ValueError : If the run is a fold unit but recorded no region for the split, which means the split is not one the fold scores.
        """
        if not UnitManifest.exists(run_directory):
            return None

        manifest = UnitManifest.read(run_directory)
        if manifest["kind"] != cls.KIND:
            return None

        regions = manifest["inference_regions"]
        if split not in regions:
            raise ValueError(f"Fold unit {run_directory} scores the splits {sorted(regions)} over its own row block, not '{split}'; pick one of those splits")

        return CropRegion(*regions[split])


class FoldRunNaming:
    """Composes fold run names with the family-specific training tag.

    Attributes:
        config       : Cross-validation configuration.
        training_tag : Descriptive tag prefixed to every fold run name, or None for the autoencoder families that carry no such tag.
    """

    def __init__(self, config: CrossValidationConfig) -> None:
        """Resolves the training tag for the configured family.

        Args:
            config : Cross-validation configuration.
        """
        self.config       = config
        self.training_tag = self._training_tag()

    def _n_gaussians(self) -> int:
        """Returns the default Gaussian count the dataset was built with."""
        return GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path).n_default_gaussians

    def _training_tag(self) -> str | None:
        """Returns the run-name tag describing the model and training recipe.

        Returns:
            The backbone, dual, or JEPA tag, or None for autoencoder families.
        """
        from pipelines.shared.training.run_naming import JepaNamingSpec, RunNaming

        config = self.config
        FoldFamily.validate(config.training_type)

        if config.training_type == "backbone":
            return RunNaming.training_tag(config.backbone_name, config.backbone_head, config.curriculum, self._n_gaussians(), config.augmentation)

        if config.training_type == "dual":
            return RunNaming.training_tag(FoldFamily.model_label(config), "set_pred", config.curriculum, self._n_gaussians(), config.augmentation, extras=(self._dual_input_tag(),))

        if config.training_type == "jepa":
            return RunNaming.jepa_tag(config.backbone_name, config.backbone_head, JepaNamingSpec.resolve(config.jepa), self._n_gaussians(), config.augmentation)

        return None

    def _dual_input_tag(self) -> str:
        """Returns the run-name fragment naming both dual trunks' input channel groups."""
        from pipelines.dual.training.pipeline import TrunkChannelMap

        dual = self.config.dual

        return f"{TrunkChannelMap.group_label(tuple(dual.params_input))}.{TrunkChannelMap.group_label(tuple(dual.existence_input))}"

    def run_name(self, fold_index: int, seed: int | None) -> str:
        """Returns the full run name of one fold and seed.

        Args:
            fold_index : Zero-based index of the fold.
            seed       : Seed of the unit, or None for an unseeded run.

        Returns:
            The fold name, prefixed with the training tag when the family has one.
        """
        from pipelines.shared.training.run_naming import RunNaming

        fold = FoldNaming.run_name(fold_index, seed)

        return fold if self.training_tag is None else RunNaming.compose(self.training_tag, fold)

    def units(self) -> list[tuple[int, int | None, str]]:
        """Returns every (fold index, seed, run name) unit the sweep will launch."""
        indices = {self.run_name(index, None): index for index in range(self.config.folds.n_folds)}
        units   = SeedSet.units(list(indices), self.config.seeds)

        return [(indices[base], seed, run_name) for base, seed, run_name in units]


class FoldConfigFactory(ConfigFactory):
    """Config factory that adds the fold planner and per-fold inference configs."""

    def __init__(self, config: CrossValidationConfig) -> None:
        """Binds the factory to a cross-validation configuration."""
        super().__init__(config)
        self._planner: FoldPlanner | None = None

    def planner(self) -> FoldPlanner:
        """Returns the fold planner, building it against the dataset crop on first use.

        Raises:
            ValueError : If the configured fold row window is not contained in the dataset global crop.
        """
        if self._planner is None:
            crop  = self.global_crop()
            folds = self.config.folds

            if folds.row_start < crop.row_start or folds.row_end > crop.row_end:
                raise ValueError(f"Fold row window [{folds.row_start}, {folds.row_end}) must lie within the dataset global crop row extent [{crop.row_start}, {crop.row_end})")

            self._planner = FoldPlanner(self.config, col_start=crop.col_start, col_end=crop.col_end)
        return self._planner

    def fold_inference_config(self, run_directory: Path, split: str, fold_index: int) -> InferenceConfig:
        """Builds the spatial-family inference config restricted to a fold's region.

        Args:
            run_directory : Directory of the trained fold run.
            split         : Split to run inference on, 'test' or 'val'.
            fold_index    : Zero-based index of the fold.

        Returns:
            An inference config whose region override is the fold's untrimmed
            row block.
        """
        inference_config                 = self.inference_config(run_directory)
        inference_config.split           = split
        inference_config.region_override = list(self.planner().inference_region(fold_index, split).as_tuple())

        return inference_config

    def fold_profile_ae_inference_config(self, run_directory: Path, split: str) -> ProfileAeInferenceConfig:
        """Builds the profile-autoencoder inference config for one fold and split."""
        inference = self.config.inference

        return ProfileAeInferenceConfig(
            run_directory   = run_directory,
            output_subdir   = None,
            split           = split,
            seed            = self.config.seed,
            checkpoint_name = inference.checkpoint_name,
            batch_size      = inference.batch_size,
            num_workers     = inference.num_workers,
            pixel_subsample = self.config.autoencoder.pixel_subsample,
            keep_empty_frac = self.config.autoencoder.keep_empty_frac,
            save_plots      = inference.save_plots,
        )

    def fold_image_ae_inference_config(self, run_directory: Path, split: str) -> ImageAeInferenceConfig:
        """Builds the image-autoencoder inference config for one fold and split."""
        inference = self.config.inference

        return ImageAeInferenceConfig(
            run_directory   = run_directory,
            output_subdir   = None,
            split           = split,
            checkpoint_name = inference.checkpoint_name,
            batch_size      = inference.batch_size,
            num_workers     = inference.num_workers,
            save_plots      = inference.save_plots,
        )


class FoldTreeSurvey:
    """Reads what a run's fold tree records about the sweep that produced it.

    Everything the cross-fold report needs is stored by the fold runs
    themselves: the model family in the architecture config under meta, the
    model name in the run summary, the fold's split regions in the dataset
    creation config, and the scored splits in the inference completion
    markers. A run whose pipeline directory never made it to disk is therefore
    still reportable.

    Attributes:
        run_dir          : Cross-validation run directory holding the folds directory.
        folds_dir        : Directory holding one fold run, or one seed sweep, per fold.
        FOLDS_DIRNAME    : Name of the folds directory inside a run.
        META_DIRNAME     : Directory marking a training run inside a fold.
        SUMMARY_FILENAME : Run summary naming the trained model, under the meta directory.
        DATASET_FILENAME : Dataset creation config carrying the split regions, under the meta directory.
        FAMILIES         : Cross-validation training family per run type the classifier reports.
    """

    FOLDS_DIRNAME    = "folds"
    META_DIRNAME     = "meta"
    SUMMARY_FILENAME = "run_summary.json"
    DATASET_FILENAME = "dataset_creation_config.json"

    FAMILIES = {
        RunType.BACKBONE   : "backbone",
        RunType.DUAL       : "dual",
        RunType.PROFILE_AE : "profile_autoencoder",
        RunType.IMAGE_AE   : "image_autoencoder",
    }

    def __init__(self, run_dir: Path) -> None:
        """Binds the survey to a run directory."""
        self.run_dir   = Path(run_dir)
        self.folds_dir = self.run_dir / self.FOLDS_DIRNAME

    def fold_runs(self) -> dict[int, Path]:
        """Returns one representative training run per fold, keyed by fold index.

        A fold trained under several seeds holds one run per seed; the first
        seed represents the fold, since every seed stored the same regions and
        model.

        Raises:
            FileNotFoundError : If the folds directory is missing, holds no fold, or a fold holds no training run.
        """
        if not self.folds_dir.is_dir():
            raise FileNotFoundError(f"No '{self.FOLDS_DIRNAME}' directory under {self.run_dir}")

        fold_dirs = sorted(child for child in self.folds_dir.iterdir() if child.is_dir() and FoldNaming.is_fold(child.name))
        if not fold_dirs:
            raise FileNotFoundError(f"No fold run directory under {self.folds_dir}")

        return {FoldNaming.index(fold_dir.name): self._representative(fold_dir) for fold_dir in fold_dirs}

    def _representative(self, fold_dir: Path) -> Path:
        """Returns the fold directory itself when it is a training run, else its first seed run.

        Raises:
            FileNotFoundError : If neither the fold directory nor any seed directory inside it holds a meta directory.
        """
        if (fold_dir / self.META_DIRNAME).is_dir():
            return fold_dir

        seeds = sorted(child for child in fold_dir.iterdir() if child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) and (child / self.META_DIRNAME).is_dir())
        if not seeds:
            raise FileNotFoundError(f"Fold '{fold_dir.name}' holds no training run: neither a '{self.META_DIRNAME}' directory nor a seed run with one under {fold_dir}")

        return seeds[0]

    def training_type(self, run_dir: Path) -> str:
        """Returns the cross-validation family of a fold run, read off its architecture config.

        Args:
            run_dir : Representative training run of a fold.

        Raises:
            ValueError : If the run carries no architecture config, or one of a family cross-validation does not train.
        """
        run_type = RunClassifier.classify(run_dir)

        if run_type not in self.FAMILIES:
            raise ValueError(f"Fold run {run_dir} is a '{run_type}' run, which cross-validation does not train; expected one of {sorted(self.FAMILIES)}")

        return self.FAMILIES[run_type]

    def model_name(self, run_dir: Path) -> str:
        """Returns the model name the fold run recorded in its run summary."""
        return str(FileIO.load_json(run_dir / self.META_DIRNAME / self.SUMMARY_FILENAME)["model_name"])

    def split_regions(self, run_dir: Path) -> SplitRegions:
        """Returns the split regions the fold run trained and scored on."""
        return SplitRegions.from_payload(FileIO.load_json(run_dir / self.META_DIRNAME / self.DATASET_FILENAME)["split_regions"])

    def splits(self, fold_runs: dict[int, Path], training_type: str) -> list[str]:
        """Returns the fold-specific splits at least one fold holds a completed inference for, in planner order.

        Args:
            fold_runs     : Representative training run per fold.
            training_type : Family deciding where the fold inference lives.
        """
        return [split for split in FoldPlanner.INFERENCE_SPLITS if any(CompletedInference.stamps(FoldFamily.inference_base(run_dir, training_type), split) for run_dir in fold_runs.values())]

    def planner(self, fold_runs: dict[int, Path]) -> StoredFoldPlanner:
        """Returns the fold plan read back from every fold's stored regions."""
        return StoredFoldPlanner({index: self.split_regions(run_dir) for index, run_dir in fold_runs.items()})
