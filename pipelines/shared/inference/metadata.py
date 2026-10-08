"""Discovery and layout of inference output stamps under a training run.

A split stored as several disjoint regions is scored once per region, each
pass stamping its own output directory. The region pass recorded in every
such marker lets the discovery treat the group as one inference that is
complete only when every region of it carries a marker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib     import Path

from tools.data.io                     import FileIO
from tools.runtime.completion          import CompletionMarker
from tools.runtime.environment_capture import EnvironmentCapture
from tools.runtime.run_tag             import RunTag


@dataclass(frozen=True)
class RegionPass:
    """Position of one scoring pass inside a split stored as several disjoint regions.

    Attributes:
        index : Zero-based index of the region this pass scores.
        count : Number of regions the split is stored as, all of which must be stamped for the split to count as scored.
        group : Stamp shared by every pass of the same launch, naming the group the pass belongs to.
        KEY   : Completion-marker entry the pass is recorded under.
    """

    index : int
    count : int
    group : str

    KEY = "region_pass"

    def payload(self) -> dict:
        """Returns the marker entry recording this pass."""
        return asdict(self)

    @staticmethod
    def from_payload(payload: dict) -> "RegionPass | None":
        """Returns the region pass a completion marker records, or None for a single-pass stamp."""
        if RegionPass.KEY not in payload:
            return None

        entry = payload[RegionPass.KEY]

        return RegionPass(index=int(entry["index"]), count=int(entry["count"]), group=str(entry["group"]))


class CompletedInference:
    """Finds the completed inference stamps of a run for a given data split.

    Stamps of a region group count only when every region of the group is
    stamped, so a launch that scored some regions and then failed never
    passes for a completed inference of the split.

    Attributes:
        CHECKPOINT_KEY : Completion-marker entry recording the checkpoint a stamp was scored with.
    """

    CHECKPOINT_KEY = "checkpoint"

    @staticmethod
    def _marked(base: Path, split: str) -> list[tuple[Path, dict]]:
        """Returns every stamp under the base carrying a completion marker for the split, with its payload, oldest first.

        Args:
            base  : Inference directory of a run (or one of its sub-families).
            split : Data split the marker must record ("train", "val", "test").

        Raises:
            KeyError : A completion marker predates split recording and must be backfilled.
        """
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
        """Returns, per region group among the marked stamps, the stamped region indices and the regions expected.

        Args:
            marked : Stamps with their marker payloads, as returned by _marked.

        Returns:
            Mapping from group stamp to (sorted indices of the stamped regions, number of regions the split is stored as); single-pass stamps form no group.
        """
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
        """Returns the region groups stamped on the split whose regions are not all stamped.

        Args:
            base  : Inference directory of a run (or one of its sub-families).
            split : Data split the markers must record.

        Returns:
            Mapping from group stamp to (sorted indices of the stamped regions, number of regions expected), for the groups still missing regions.
        """
        groups = CompletedInference._groups(CompletedInference._marked(base, split))

        return {group: (present, count) for group, (present, count) in groups.items() if len(present) < count}

    @staticmethod
    def stamps(base: Path, split: str) -> list[Path]:
        """Returns the completed inference stamps scored on one split, oldest first.

        Args:
            base  : Inference directory of a run (or one of its sub-families).
            split : Data split the stamp must record ("train", "val", "test").

        Returns:
            Timestamped stamp directories carrying a completion marker for that
            split, leaving out every stamp of a region group that is missing
            regions; empty when the base directory does not exist.

        Raises:
            KeyError : A completion marker predates split recording and must be backfilled.
        """
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
        """Returns the checkpoint file a completed stamp was scored with, or None when its marker records none.

        Backbone and dual stamps record their checkpoint; the autoencoder
        families, and backbone stamps written before the checkpoint was
        recorded, do not.

        Args:
            stamp : Completed inference stamp directory.
        """
        payload = CompletionMarker.payload(stamp)

        if CompletedInference.CHECKPOINT_KEY not in payload:
            return None

        return str(payload[CompletedInference.CHECKPOINT_KEY]["name"])

    @staticmethod
    def latest(base: Path, split: str) -> Path | None:
        """Returns the newest completed stamp for a split, or None when there is none."""
        stamps = CompletedInference.stamps(base, split)

        return stamps[-1] if stamps else None


class InferenceMetadata:
    """Resolves the output paths of one inference run.

    Attributes:
        SUBDIR       : Family sub-directory under the run's inference/ tree.
        output_dir   : Stamp directory holding this inference run's artifacts.
        figures_dir  : Directory for the generated figures.
        logs_dir     : Directory for the run's text logs.
        metrics_path : Path of the metrics JSON file.
        report_path  : Path of the markdown report.
    """

    SUBDIR: str

    def __init__(self, config) -> None:
        """Derives the stamp directory and artifact paths from the inference config.

        Args:
            config : Inference configuration; an explicit output_subdir names the stamp, otherwise the current timestamp is used.
        """
        paths = config.paths

        base = config.run_directory / "inference" / self.SUBDIR

        self.output_dir   = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir  = self.output_dir / paths.figures_subdir
        self.logs_dir     = self.output_dir / paths.logs_subdir
        self.metrics_path = self.output_dir / paths.metrics_filename
        self.report_path  = self.output_dir / paths.report_filename

    def create_dirs(self) -> None:
        """Creates the stamp, figures and logs directories and records the environment snapshot."""
        FileIO.ensure_dirs(self.output_dir, self.figures_dir, self.logs_dir)
        FileIO.save_json(EnvironmentCapture.snapshot(), self.output_dir / "environment.json")
