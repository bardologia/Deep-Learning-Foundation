from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.cross_validation    import CrossValidationConfig
from configuration.inference           import InferenceConfig
from configuration.dataset.gaussian import GaussianConfig
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.training.seed_sweep   import SeedSet
from tools.data.regions import CropRegion, SplitRegions


@dataclass
class FoldPlan:
    fold_index    : int
    test_block    : int
    val_block     : int
    train_blocks  : list[int]
    split_regions : SplitRegions


class FoldPlanner:
    def __init__(self, config: CrossValidationConfig, col_start: int, col_end: int) -> None:
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
        total = row_end - row_start
        if total < n_folds:
            raise ValueError(f"Row extent {total} is smaller than n_folds {n_folds}")

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
        margin    = self.guard // 2
        row_start = row_start + margin if row_start > self.row_start else row_start
        row_end   = row_end - margin   if row_end   < self.row_end   else row_end
        return row_start, row_end

    def plan(self, fold_index: int) -> FoldPlan:
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
        return [self.plan(fold_index) for fold_index in range(self.n_folds)]


class FoldNaming:
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


class FoldRunNaming:
    def __init__(self, config: CrossValidationConfig) -> None:
        self.config       = config
        self.training_tag = self._training_tag()

    def _n_gaussians(self) -> int:
        return GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path).n_default_gaussians

    def _training_tag(self) -> str | None:
        from pipelines.shared.training.run_naming import RunNaming

        config = self.config

        if config.training_type == "backbone":
            return RunNaming.training_tag(config.backbone_name, config.backbone_head, config.curriculum, self._n_gaussians(), config.augmentation)

        if config.training_type == "jepa":
            return RunNaming.tag(config.backbone_name, config.backbone_head, config.jepa.param_loss, self._n_gaussians(), config.augmentation)

        return None

    def run_name(self, fold_index: int, seed: int | None) -> str:
        from pipelines.shared.training.run_naming import RunNaming

        fold = FoldNaming.run_name(fold_index, seed)

        return fold if self.training_tag is None else RunNaming.compose(self.training_tag, fold)

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
            crop  = self.global_crop()
            folds = self.config.folds

            if folds.row_start < crop.row_start or folds.row_end > crop.row_end:
                raise ValueError(f"Fold row window [{folds.row_start}, {folds.row_end}) must lie within the dataset global crop row extent [{crop.row_start}, {crop.row_end})")

            self._planner = FoldPlanner(self.config, col_start=crop.col_start, col_end=crop.col_end)
        return self._planner

    def fold_inference_config(self, run_directory: Path, split: str) -> InferenceConfig:
        inference_config               = self.inference_config(run_directory)
        inference_config.split         = split
        inference_config.output_subdir = split

        return inference_config
