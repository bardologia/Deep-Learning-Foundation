
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib     import Path

from configuration.dataset                    import DatasetConfig, SplitRegions
from configuration.dataset.gaussian           import DatasetParameterPairing
from configuration.inference.general          import InferenceConfig
from pipelines.shared.dataset.dataset_spatial import Layout
from tools.data.io                            import FileIO
from tools.data.regions                       import CropRegion
from tools.runtime.completion                 import CompletionMarker
from tools.runtime.environment_capture        import EnvironmentCapture
from tools.runtime.run_tag                    import RunTag


@dataclass(frozen=True)
class RegionPass:

    index : int
    count : int
    group : str

    KEY = "region_pass"

    def payload(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_payload(payload: dict) -> "RegionPass | None":
        if RegionPass.KEY not in payload:
            return None

        entry = payload[RegionPass.KEY]

        return RegionPass(index=int(entry["index"]), count=int(entry["count"]), group=str(entry["group"]))


class TargetDataset:

    KEY   = "target_dataset"
    SPLIT = "test"

    def __init__(self, dataset_path: Path, parameters_path: Path, secondary_labels: tuple | None, region: CropRegion) -> None:
        self.dataset_path     = Path(dataset_path)
        self.parameters_path  = Path(parameters_path)
        self.secondary_labels = secondary_labels
        self.region           = region

    @property
    def name(self) -> str:
        return self.dataset_path.name

    @staticmethod
    def _check_block(config: InferenceConfig) -> None:
        if config.split != TargetDataset.SPLIT:
            raise ValueError(f"inference.target is enabled with split='{config.split}', but a target crop is scored as the held-out {TargetDataset.SPLIT} split of the target scene; leave inference.split at '{TargetDataset.SPLIT}'.")

        if config.region_override:
            raise ValueError(f"inference.target is enabled together with region_override={config.region_override}; the scored crop of a target dataset is inference.target.row_start/row_end/col_start/col_end, so leave region_override unset.")

    @staticmethod
    def _check_paths(dataset_path: Path, parameters_path: Path) -> CropRegion:
        layout_path = dataset_path / Layout.LAYOUT_RELPATH
        if not layout_path.is_file():
            raise FileNotFoundError(f"inference.target.dataset_path={dataset_path} holds no {Layout.LAYOUT_RELPATH}; point it at a dataset directory written by generate_dataset.")

        DatasetParameterPairing.relative_template(dataset_path, parameters_path)

        if not parameters_path.is_file():
            raise FileNotFoundError(f"inference.target.parameters_path={parameters_path} does not exist; the target scene needs a parameter run to score the predictions against.")

        return CropRegion(*FileIO.load_json(layout_path)["global_crop"])

    @staticmethod
    def _region(config: InferenceConfig, global_crop: CropRegion) -> CropRegion:
        target      = config.target
        row_bounds  = (global_crop.row_start, global_crop.row_end) if target.row_start == target.row_end == 0 else (target.row_start, target.row_end)
        col_bounds  = (global_crop.col_start, global_crop.col_end) if target.col_start == target.col_end == 0 else (target.col_start, target.col_end)
        bounds      = (*row_bounds, *col_bounds)

        if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
            raise ValueError(f"inference.target crop {bounds} is empty or inverted; set start < end on each axis inside the target's global crop {global_crop.as_tuple()}, or leave both bounds of an axis at 0 to span the whole crop along it.")

        region = CropRegion(*bounds)
        if not global_crop.contains(region):
            raise ValueError(f"inference.target crop {region.as_tuple()} is not contained in the global crop {global_crop.as_tuple()} of {target.dataset_path}.")

        return region

    @staticmethod
    def of(config: InferenceConfig) -> "TargetDataset | None":
        target = config.target
        if not target.enabled:
            return None

        TargetDataset._check_block(config)

        dataset_path    = Path(target.dataset_path)
        parameters_path = Path(target.parameters_path)
        global_crop     = TargetDataset._check_paths(dataset_path, parameters_path)
        labels          = tuple(target.secondary_labels) if target.secondary_labels else None

        return TargetDataset(dataset_path, parameters_path, labels, TargetDataset._region(config, global_crop))

    def dataset_config(self, trained: DatasetConfig) -> DatasetConfig:
        return replace(
            trained,
            preprocessing_run_directory = self.dataset_path,
            parameters_path             = self.parameters_path,
            secondary_labels            = self.secondary_labels,
            split_regions               = SplitRegions(train=[], val=[], test=self.region),
        )

    def payload(self) -> dict:
        return {
            "dataset_path"     : str(self.dataset_path),
            "parameters_path"  : str(self.parameters_path),
            "secondary_labels" : list(self.secondary_labels) if self.secondary_labels is not None else None,
            "region"           : list(self.region.as_tuple()),
        }

    def describe(self) -> str:
        channels = ", ".join(self.secondary_labels) if self.secondary_labels else "every secondary channel"

        return f"{self.name}, labels {self.parameters_path.parent.name}, crop {self.region.as_tuple()}, secondaries {channels}"


class CompletedInference:

    CHECKPOINT_KEY = "checkpoint"

    @staticmethod
    def _marked(base: Path, split: str) -> list[tuple[Path, dict]]:
        base = Path(base)
        if not base.is_dir():
            return []

        marked = []
        for candidate in sorted(base.iterdir()):
            if not candidate.is_dir() or not CompletionMarker.is_complete(candidate):
                continue

            payload = CompletionMarker.payload(candidate)

            if "split" not in payload:
                raise KeyError(f"inference completion marker at {candidate} carries no 'split' key; re-run scripts/backfill_completion_markers.py to rewrite it from the recorded metrics")
            if payload["split"] == split:
                marked.append((candidate, payload))

        return marked

    @staticmethod
    def _groups(marked: list[tuple[Path, dict]]) -> dict[str, tuple[list[int], int]]:
        groups: dict[str, tuple[list[int], int]] = {}

        for _stamp, payload in marked:
            region = RegionPass.from_payload(payload)
            if region is None:
                continue

            present, _count = groups.setdefault(region.group, ([], region.count))
            present.append(region.index)

        return {group: (sorted(present), count) for group, (present, count) in groups.items()}

    @staticmethod
    def incomplete_groups(base: Path, split: str) -> dict[str, tuple[list[int], int]]:
        groups = CompletedInference._groups(CompletedInference._marked(base, split))

        return {group: (present, count) for group, (present, count) in groups.items() if len(present) < count}

    @staticmethod
    def stamps(base: Path, split: str) -> list[Path]:
        marked     = CompletedInference._marked(base, split)
        incomplete = {group for group, (present, count) in CompletedInference._groups(marked).items() if len(present) < count}

        selected = []
        for stamp, payload in marked:
            region = RegionPass.from_payload(payload)
            if region is not None and region.group in incomplete:
                continue
            selected.append(stamp)

        return selected

    @staticmethod
    def checkpoint_name(stamp: Path) -> str | None:
        payload = CompletionMarker.payload(stamp)

        if CompletedInference.CHECKPOINT_KEY not in payload:
            return None

        return str(payload[CompletedInference.CHECKPOINT_KEY]["name"])

    @staticmethod
    def target(stamp: Path) -> dict:
        payload = CompletionMarker.payload(stamp)

        if TargetDataset.KEY not in payload:
            raise KeyError(f"inference completion marker at {stamp} records no '{TargetDataset.KEY}'; every stamp under a cross-dataset tree is written with one, so this stamp was not written by the target-dataset inference.")

        return payload[TargetDataset.KEY]

    @staticmethod
    def latest(base: Path, split: str) -> Path | None:
        stamps = CompletedInference.stamps(base, split)

        return stamps[-1] if stamps else None


class InferenceMetadata:

    SUBDIR: str

    def __init__(self, config) -> None:
        paths = config.paths

        base = config.run_directory / "inference" / self.SUBDIR

        self.output_dir   = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir  = self.output_dir / paths.figures_subdir
        self.logs_dir     = self.output_dir / paths.logs_subdir
        self.metrics_path = self.output_dir / paths.metrics_filename
        self.report_path  = self.output_dir / paths.report_filename

    def create_dirs(self) -> None:
        FileIO.ensure_dirs(self.output_dir, self.figures_dir, self.logs_dir)
        FileIO.save_json(EnvironmentCapture.snapshot(), self.output_dir / "environment.json")
