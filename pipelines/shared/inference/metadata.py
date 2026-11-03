"""Discovery and layout of inference output stamps under a training run.

A split stored as several disjoint regions is scored once per region, each
pass stamping its own output directory. The region pass recorded in every
such marker lets the discovery treat the group as one inference that is
complete only when every region of it carries a marker. A run scored on a
foreign dataset records that target in its marker and stamps under its own
tree, apart from the run's own inference.
"""

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


class TargetDataset:
    """Foreign dataset one inference pass scores a trained run on, in place of the run's own split.

    The target scene has no train, validation or test partition: none of it was
    seen in training, so its scored crop is the test split of a dataset
    configuration that keeps the run's patch geometry, input channel layout and
    batch settings and swaps the scene, the labels and the secondary channels.

    Attributes:
        KEY              : Completion-marker and metrics entry the target is recorded under.
        SPLIT            : Split name the target crop is scored as.
        dataset_path     : Dataset directory of the target scene.
        parameters_path  : Gaussian label cube of the target parameter run.
        secondary_labels : Secondary channel labels fed to the model, or None for every secondary channel of the target stack.
        region           : Scored crop of the target scene.
    """

    KEY   = "target_dataset"
    SPLIT = "test"

    def __init__(self, dataset_path: Path, parameters_path: Path, secondary_labels: tuple | None, region: CropRegion) -> None:
        """Stores the target scene, its labels, the secondary channels fed to the model and the scored crop."""
        self.dataset_path     = Path(dataset_path)
        self.parameters_path  = Path(parameters_path)
        self.secondary_labels = secondary_labels
        self.region           = region

    @property
    def name(self) -> str:
        """Directory name of the target dataset, naming its stamp tree inside a run."""
        return self.dataset_path.name

    @staticmethod
    def _check_block(config: InferenceConfig) -> None:
        """Refuses the inference settings that only make sense on the run's own split.

        Raises:
            ValueError : If the split is not the test split or a region override is set.
        """
        if config.split != TargetDataset.SPLIT:
            raise ValueError(f"inference.target is enabled with split='{config.split}', but a target crop is scored as the held-out {TargetDataset.SPLIT} split of the target scene; leave inference.split at '{TargetDataset.SPLIT}'.")

        if config.region_override:
            raise ValueError(f"inference.target is enabled together with region_override={config.region_override}; the scored crop of a target dataset is inference.target.row_start/row_end/col_start/col_end, so leave region_override unset.")

    @staticmethod
    def _check_paths(dataset_path: Path, parameters_path: Path) -> CropRegion:
        """Checks that the target dataset and its label cube exist and belong together.

        Args:
            dataset_path    : Dataset directory of the target scene.
            parameters_path : Gaussian label cube inside that dataset.

        Returns:
            The global crop the target dataset covers.

        Raises:
            FileNotFoundError : If the dataset layout manifest or the label cube is missing.
            ValueError        : If the label cube does not live inside the target dataset.
        """
        layout_path = dataset_path / Layout.LAYOUT_RELPATH
        if not layout_path.is_file():
            raise FileNotFoundError(f"inference.target.dataset_path={dataset_path} holds no {Layout.LAYOUT_RELPATH}; point it at a dataset directory written by generate_dataset.")

        DatasetParameterPairing.relative_template(dataset_path, parameters_path)

        if not parameters_path.is_file():
            raise FileNotFoundError(f"inference.target.parameters_path={parameters_path} does not exist; the target scene needs a parameter run to score the predictions against.")

        return CropRegion(*FileIO.load_json(layout_path)["global_crop"])

    @staticmethod
    def _region(config: InferenceConfig, global_crop: CropRegion) -> CropRegion:
        """Returns the scored crop of the target scene, checked against its global crop.

        An axis whose start and end are both zero spans the whole global crop along
        that axis, so leaving all four bounds at zero scores the full target scene.

        Args:
            config      : Inference configuration carrying the target crop bounds.
            global_crop : Global crop of the target dataset.

        Raises:
            ValueError : If the crop is empty or inverted, or does not lie inside the global crop.
        """
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
        """Returns the checked target of an inference configuration, or None when the run is scored on its own split.

        Args:
            config : Inference configuration whose target block is read.

        Raises:
            ValueError        : If the split or region override conflicts with a target, the crop is invalid, or the label cube lives outside the target dataset.
            FileNotFoundError : If the target dataset or its label cube is missing.
        """
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
        """Returns the run's dataset configuration moved onto the target scene.

        Args:
            trained : Dataset configuration the run was trained with.

        Returns:
            The same patch geometry, channel layout and batch settings over the target scene, its labels and secondary channels, with the scored crop as the only (test) region.
        """
        return replace(
            trained,
            preprocessing_run_directory = self.dataset_path,
            parameters_path             = self.parameters_path,
            secondary_labels            = self.secondary_labels,
            split_regions               = SplitRegions(train=[], val=[], test=self.region),
        )

    def payload(self) -> dict:
        """Returns the marker and metrics entry naming the target scene, its labels, secondary channels and crop."""
        return {
            "dataset_path"     : str(self.dataset_path),
            "parameters_path"  : str(self.parameters_path),
            "secondary_labels" : list(self.secondary_labels) if self.secondary_labels is not None else None,
            "region"           : list(self.region.as_tuple()),
        }

    def describe(self) -> str:
        """Returns a one-line description of the target for logs and reports."""
        channels = ", ".join(self.secondary_labels) if self.secondary_labels else "every secondary channel"

        return f"{self.name}, labels {self.parameters_path.parent.name}, crop {self.region.as_tuple()}, secondaries {channels}"


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
    def target(stamp: Path) -> dict:
        """Returns the target dataset a completed stamp under a cross-dataset tree was scored on, as its marker records it.

        Args:
            stamp : Completed inference stamp directory under a run's cross-dataset tree.

        Raises:
            KeyError : If the marker records no target dataset, which no stamp of that tree is written without.
        """
        payload = CompletionMarker.payload(stamp)

        if TargetDataset.KEY not in payload:
            raise KeyError(f"inference completion marker at {stamp} records no '{TargetDataset.KEY}'; every stamp under a cross-dataset tree is written with one, so this stamp was not written by the target-dataset inference.")

        return payload[TargetDataset.KEY]

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
