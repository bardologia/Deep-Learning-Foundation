
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib     import Path

from configuration.dataset                  import DatasetConfig
from configuration.inference.general        import InferenceConfig
from pipelines.shared.dataset.dataset_files import DatasetFiles
from tools.runtime.completion               import CompletionMarker


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

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = Path(dataset_path)

    @property
    def name(self) -> str:
        return self.dataset_path.name

    @staticmethod
    def of(config: InferenceConfig) -> "TargetDataset | None":
        return TargetDataset(config.target.dataset_path) if config.target.enabled else None

    def dataset_config(self, trained: DatasetConfig) -> DatasetConfig:
        return replace(trained, dataset_directory=self.dataset_path, split_regions=DatasetFiles.read(self.dataset_path).splits)

    def payload(self) -> dict:
        return {"dataset_path": str(self.dataset_path)}

    def describe(self) -> str:
        return f"{self.name}, scored on its own {self.SPLIT} split"


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
    def checkpoint_name(stamp: Path) -> str:
        return str(CompletionMarker.payload(stamp)[CompletedInference.CHECKPOINT_KEY]["name"])

    @staticmethod
    def target(stamp: Path) -> dict:
        payload = CompletionMarker.payload(stamp)

        return payload[TargetDataset.KEY]

    @staticmethod
    def latest(base: Path, split: str) -> Path | None:
        stamps = CompletedInference.stamps(base, split)

        return stamps[-1] if stamps else None
