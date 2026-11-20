
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib     import Path

from configuration.cross_validation               import CrossValidationConfig
from configuration.inference                      import InferenceConfig
from pipelines.shared.config.config_factory       import ConfigFactory
from pipelines.shared.dataset.dataset_files       import DatasetFiles
from pipelines.shared.inference.metadata          import CompletedInference
from pipelines.shared.inference.run_classifier    import InferenceLayout
from pipelines.shared.orchestration.unit_manifest import UnitManifest
from pipelines.shared.training.seed_sweep         import SeedSet
from tools.data.io                                import FileIO
from tools.data.regions                           import CropRegion, SplitRegions


@dataclass
class FoldPlan:

    fold_index    : int
    test_block    : int
    val_block     : int
    train_blocks  : list[int]
    split_regions : SplitRegions


class FoldPlanner:

    INFERENCE_SPLITS = ("test", "val")

    def __init__(self, config: CrossValidationConfig, col_start: int, col_end: int) -> None:
        folds = config.folds

        self.n_folds   = folds.n_folds
        self.guard     = folds.guard
        self.row_start = folds.row_start
        self.row_end   = folds.row_end
        self.col_start = col_start
        self.col_end   = col_end
        self.blocks    = self._partition(folds.row_start, folds.row_end, folds.n_folds)

    def _partition(self, row_start: int, row_end: int, n_folds: int) -> list[tuple[int, int]]:
        total = row_end - row_start

        size   = total // n_folds
        bounds = [row_start + index * size for index in range(n_folds)] + [row_end]

        return [(bounds[index], bounds[index + 1]) for index in range(n_folds)]

    def _merge_adjacent(self, block_indices: list[int]) -> list[tuple[int, int]]:
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
        first_block, last_block = run
        row_start, row_end = self._trim(self.blocks[first_block][0], self.blocks[last_block][1])
        return CropRegion(row_start, row_end, self.col_start, self.col_end)

    def _block_region(self, block_index: int) -> CropRegion:
        row_start, row_end = self._trim(*self.blocks[block_index])
        return CropRegion(row_start, row_end, self.col_start, self.col_end)

    def _trim(self, row_start: int, row_end: int) -> tuple[int, int]:
        margin        = self.guard // 2
        row_start = row_start + margin if row_start > self.row_start else row_start
        row_end   = row_end - margin   if row_end   < self.row_end   else row_end
        return row_start, row_end

    def inference_region(self, fold_index: int, split: str) -> CropRegion:
        plan   = self.plan(fold_index)
        blocks = {"test": plan.test_block, "val": plan.val_block}

        row_start, row_end = self.blocks[blocks[split]]
        return CropRegion(row_start, row_end, self.col_start, self.col_end)

    def plan(self, fold_index: int) -> FoldPlan:
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
        return [self.plan(fold_index) for fold_index in range(self.n_folds)]


class StoredFoldPlanner:

    def __init__(self, regions: dict[int, SplitRegions]) -> None:
        self.n_folds = len(regions)
        self.regions = dict(regions)

    def plan(self, fold_index: int) -> FoldPlan:
        test_block   = fold_index
        val_block    = (fold_index + 1) % self.n_folds
        train_blocks = [index for index in range(self.n_folds) if index not in (test_block, val_block)]

        return FoldPlan(fold_index=fold_index, test_block=test_block, val_block=val_block, train_blocks=train_blocks, split_regions=self.regions[fold_index])

    def plans(self) -> list[FoldPlan]:
        return [self.plan(fold_index) for fold_index in range(self.n_folds)]


class FoldNaming:

    PATTERN = re.compile(r"(?:^|_)fold_(\d+)$")

    @staticmethod
    def is_fold(name: str) -> bool:
        return FoldNaming.PATTERN.search(name) is not None

    @staticmethod
    def name(index: int) -> str:
        return f"fold_{index}"

    @staticmethod
    def run_name(index: int, seed: int | None) -> str:
        base = FoldNaming.name(index)
        return base if seed is None else SeedSet.run_name(base, seed)

    @staticmethod
    def base(name: str) -> str:
        return name.split("/seed")[0]

    @staticmethod
    def index(name: str) -> int:
        return int(FoldNaming.base(name).split("_")[-1])

    @staticmethod
    def seed(name: str) -> int | None:
        parts = name.split("/seed")
        return int(parts[1]) if len(parts) == 2 else None


class FoldUnitManifest:

    KIND = "cross_validation"

    @staticmethod
    def identity(manifest: dict) -> tuple:
        return (int(manifest["fold_index"]), manifest["seed"])

    @classmethod
    def write(cls, run_directory: Path, run_tag: str, unit: str, fold_index: int, seed: int | None, inference_regions: dict) -> Path:
        payload = {
            "run_tag"           : run_tag,
            "unit"              : unit,
            "fold_index"        : int(fold_index),
            "seed"              : seed,
            "inference_regions" : {split: list(region) for split, region in inference_regions.items()},
        }

        return UnitManifest.write(run_directory, cls.KIND, payload)

    @classmethod
    def inference_region(cls, run_directory: Path, split: str) -> CropRegion | None:
        if not UnitManifest.exists(run_directory):
            return None

        manifest = UnitManifest.read(run_directory)
        if manifest["kind"] != cls.KIND:
            return None

        regions = manifest["inference_regions"]

        return CropRegion(*regions[split])


class FoldRunNaming:

    def __init__(self, config: CrossValidationConfig) -> None:
        self.config       = config
        self.training_tag = self._training_tag()

    def _training_tag(self) -> str:
        from pipelines.shared.training.run_naming import RunNaming

        config = self.config

        return RunNaming.tag(config.backbone_name, config.loss, config.augmentation)

    def run_name(self, fold_index: int, seed: int | None) -> str:
        from pipelines.shared.training.run_naming import RunNaming

        return RunNaming.compose(self.training_tag, FoldNaming.run_name(fold_index, seed))

    def units(self) -> list[tuple[int, int | None, str]]:
        indices = {self.run_name(index, None): index for index in range(self.config.folds.n_folds)}
        units   = SeedSet.units(list(indices), self.config.seeds)

        return [(indices[base], seed, run_name) for base, seed, run_name in units]


class FoldConfigFactory(ConfigFactory):

    def __init__(self, config: CrossValidationConfig) -> None:
        super().__init__(config)
        self._planner: FoldPlanner | None = None

    def planner(self) -> FoldPlanner:
        if self._planner is None:
            files = DatasetFiles.read(self.config.paths.dataset_path)

            self._planner = FoldPlanner(self.config, col_start=0, col_end=files.cols)
        return self._planner

    def fold_inference_config(self, run_directory: Path, split: str, fold_index: int) -> InferenceConfig:
        inference_config                 = self.inference_config(run_directory)
        inference_config.split           = split
        inference_config.region_override = list(self.planner().inference_region(fold_index, split).as_tuple())

        return inference_config


class FoldTreeSurvey:

    FOLDS_DIRNAME    = "folds"
    META_DIRNAME     = "meta"
    SUMMARY_FILENAME = "run_summary.json"
    DATASET_FILENAME = "dataset_creation_config.json"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir   = Path(run_dir)
        self.folds_dir = self.run_dir / self.FOLDS_DIRNAME

    def fold_runs(self) -> dict[int, Path]:
        fold_dirs = sorted(child for child in self.folds_dir.iterdir() if child.is_dir() and FoldNaming.is_fold(child.name))

        return {FoldNaming.index(fold_dir.name): self._representative(fold_dir) for fold_dir in fold_dirs}

    def _representative(self, fold_dir: Path) -> Path:
        if (fold_dir / self.META_DIRNAME).is_dir():
            return fold_dir

        seeds = sorted(child for child in fold_dir.iterdir() if child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) and (child / self.META_DIRNAME).is_dir())

        return seeds[0]

    def model_name(self, run_dir: Path) -> str:
        return str(FileIO.load_json(run_dir / self.META_DIRNAME / self.SUMMARY_FILENAME)["model_name"])

    def split_regions(self, run_dir: Path) -> SplitRegions:
        return SplitRegions.from_bounds(FileIO.load_json(run_dir / self.META_DIRNAME / self.DATASET_FILENAME)["split_regions"])

    def splits(self, fold_runs: dict[int, Path]) -> list[str]:
        return [split for split in FoldPlanner.INFERENCE_SPLITS if any(CompletedInference.stamps(InferenceLayout.base(run_dir), split) for run_dir in fold_runs.values())]

    def planner(self, fold_runs: dict[int, Path]) -> StoredFoldPlanner:
        return StoredFoldPlanner({index: self.split_regions(run_dir) for index, run_dir in fold_runs.items()})
