from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.patch_sweep               import PatchSweepConfig
from pipelines.shared.dataset.dataset_queue  import DatasetQueueResolver
from pipelines.shared.model.model_builder    import ModelBuilder


@dataclass(frozen=True)
class SweepUnit:
    dataset_path            : Path
    parameters_path         : Path
    patch_size              : tuple[int, int]
    patch_stride            : tuple[int, int]
    batch_size              : int
    lr_reference_batch_size : int

    @property
    def dataset(self) -> str:
        return self.dataset_path.name

    @property
    def name(self) -> str:
        return f"{self.dataset}-p{self.patch_size[0]:03d}x{self.patch_size[1]:03d}"


class ArchitecturePatchStep:

    PYRAMID_BACKBONES = frozenset({"unet", "unet_skip", "resunet", "attention_unet"})

    def __init__(self, backbone_name: str, backbone_head: str, model_overrides: dict) -> None:
        self.backbone_name   = backbone_name
        self.backbone_head   = backbone_head
        self.model_overrides = model_overrides

    def resolve(self) -> int:
        if self.backbone_name not in self.PYRAMID_BACKBONES:
            raise ValueError(f"Backbone '{self.backbone_name}' is not a verified one-halving-per-stage U-Net ({sorted(self.PYRAMID_BACKBONES)}); the admissible patch step cannot be derived from the architecture, set patch.step explicitly")

        model_config = ModelBuilder.config_from_registry(self.backbone_name, self.model_overrides, head=self.backbone_head)
        features     = model_config.features

        if not features:
            raise ValueError(f"Backbone '{self.backbone_name}' has an empty encoder feature pyramid; set patch.step explicitly")

        return 2 ** len(features)


class PatchSweepPlanner:
    def __init__(self, config: PatchSweepConfig) -> None:
        self.config   = config
        self.datasets = DatasetQueueResolver(config.dataset_base_path, config.dataset_filter).resolve()

        self._validate()

    def _validate(self) -> None:
        if not self.datasets:
            raise ValueError(f"No datasets to sweep under {self.config.dataset_base_path}; select at least one in dataset_filter or add dataset directories to the base")

        names      = [dataset.name for dataset in self.datasets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"dataset_filter must select unique directory names (they key the sweep units), duplicated: {duplicates}")

        if not 0 < self.config.patch.stride_ratio <= 1:
            raise ValueError(f"patch.stride_ratio={self.config.patch.stride_ratio} must be in (0, 1]")

        self.parameters_template()

    def parameters_template(self) -> Path:
        paths = self.config.paths

        try:
            return Path(paths.parameters_path).relative_to(paths.dataset_path)
        except ValueError:
            raise ValueError(f"paths.parameters_path={paths.parameters_path} must live inside paths.dataset_path={paths.dataset_path}; the sweep re-roots this relative layout onto every swept dataset")

    def patch_step(self) -> int:
        explicit = self.config.patch.step
        if explicit > 0:
            return explicit

        return ArchitecturePatchStep(self.config.backbone_name, self.config.backbone_head, self.config.model_overrides).resolve()

    def _axis_sizes(self, axis: int, label: str) -> list[int]:
        grid  = self.config.patch
        step  = self.patch_step()
        start = grid.minimum[axis] if grid.minimum[axis] > 0 else step

        if start % step != 0:
            raise ValueError(f"patch.minimum {label}={start} is not a multiple of the admissible step {step}")
        if grid.maximum[axis] < start:
            raise ValueError(f"patch.maximum {label}={grid.maximum[axis]} is below the first admissible size {start}")

        return list(range(start, grid.maximum[axis] + 1, step))

    def patch_sizes(self) -> tuple[list[int], list[int]]:
        azimuth_sizes = self._axis_sizes(0, "azimuth")
        range_sizes   = self._axis_sizes(1, "range")

        if len(azimuth_sizes) * len(range_sizes) < 2:
            raise ValueError(f"The grid azimuth {azimuth_sizes} x range {range_sizes} yields {len(azimuth_sizes) * len(range_sizes)} patch size; a sweep needs at least 2")

        return azimuth_sizes, range_sizes

    def _pixel_rescaled(self, base: int, patch_size: tuple[int, int]) -> int:
        if not self.config.patch.constant_pixel_budget:
            return base

        reference    = self.config.training.patch_size
        pixel_budget = base * reference[0] * reference[1]

        return max(1, pixel_budget // (patch_size[0] * patch_size[1]))

    def unit(self, name: str) -> SweepUnit:
        by_name = {unit.name: unit for unit in self.units()}
        if name not in by_name:
            raise KeyError(f"Unknown sweep unit '{name}', expected one of {sorted(by_name)}")

        return by_name[name]

    def summary(self) -> dict:
        azimuth_sizes, range_sizes = self.patch_sizes()

        return {
            "Datasets"      : [dataset.name for dataset in self.datasets],
            "Patch step"    : self.patch_step(),
            "Azimuth sizes" : azimuth_sizes,
            "Range sizes"   : range_sizes,
            "Units"         : len(self.datasets) * len(azimuth_sizes) * len(range_sizes),
            "Seeds"         : list(self.config.seeds) or [self.config.seed],
        }

    def units(self) -> list[SweepUnit]:
        stride_ratio               = self.config.patch.stride_ratio
        azimuth_sizes, range_sizes = self.patch_sizes()
        template                   = self.parameters_template()

        plans = []
        for dataset in self.datasets:
            for azimuth in azimuth_sizes:
                for range_size in range_sizes:
                    size = (azimuth, range_size)
                    plans.append(SweepUnit(
                        dataset_path            = dataset,
                        parameters_path         = dataset / template,
                        patch_size              = size,
                        patch_stride            = tuple(max(1, int(round(edge * stride_ratio))) for edge in size),
                        batch_size              = self._pixel_rescaled(self.config.training.batch_size, size),
                        lr_reference_batch_size = self._pixel_rescaled(self.config.training.lr_reference_batch_size, size),
                    ))

        return plans
