"""Per-family adapters that run inference on one trained run directory.

Each runner takes the entry-point configuration, repoints its inference block at
a given run directory, and drives the matching pipeline. A spatial run that is a
cross-validation fold unit is scored over the row block its manifest records,
so a fold can be re-scored by the generic entry, and a spatial run whose split
is stored as several disjoint regions is scored once per region. A backbone or
dual run scored on a foreign target dataset skips both, since the fold block and
the stored regions describe the scene it was trained on, and is scored in one
pass over the target crop. Pipeline imports are deferred so that selecting a
family does not pull in the others.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

from pipelines.shared.inference.metadata import RegionPass
from tools.data.io                       import FileIO
from tools.data.regions                  import CropRegion, SplitRegions
from tools.runtime.run_tag               import RunTag


class SpatialRunConfig:
    """Derives the per-run spatial inference config, honouring a fold unit's recorded block."""

    @staticmethod
    def build(inference_config, run_directory: Path):
        """Returns the inference config pointed at one run.

        An explicit region override in the entry configuration is kept; otherwise
        a fold unit's manifest supplies the untrimmed row block of the split.

        Args:
            inference_config : Inference block of the entry configuration.
            run_directory    : Training run to score.

        Raises:
            ValueError : If the run is a fold unit that does not score the configured split.
        """
        from pipelines.cross_validation.folds import FoldUnitManifest

        config = replace(inference_config, run_directory=Path(run_directory), output_subdir=None)

        if config.region_override:
            return config

        region = FoldUnitManifest.inference_region(Path(run_directory), config.split)
        if region is not None:
            config.region_override = list(region.as_tuple())

        return config


class SplitRegionLoop:
    """Scores a spatial run once per disjoint region when its split is stored as several regions.

    Stitching is defined over one rectangular crop, so a split persisted as a list
    of disjoint regions is scored region by region: each region becomes the region
    override, its output stamp carries the region index, and its pipeline records
    the region pass (index, region count and launch group) in the completion marker,
    so the group counts as scored only once every region is stamped. A configured
    region override, or a split stored as one region, is scored in a single pass.

    Attributes:
        config       : Spatial inference configuration pointed at one run.
        pipeline_cls : Spatial inference pipeline class constructed for every pass.
        components   : Inference component set handed to every pass.
    """

    def __init__(self, config, pipeline_cls, components) -> None:
        """Stores the run's inference configuration, the pipeline class and its components."""
        self.config       = config
        self.pipeline_cls = pipeline_cls
        self.components   = components

    def _stored_regions(self) -> list[CropRegion]:
        """Returns the regions the run's dataset creation config stores for the scored split.

        Raises:
            FileNotFoundError : If the run has no meta/dataset_creation_config.json.
            KeyError          : If the config stores no split regions, or they do not name the scored split.
        """
        payload = FileIO.load_json(self.config.run_directory / "meta" / "dataset_creation_config.json")

        return SplitRegions.from_payload(payload["split_regions"]).regions(self.config.split)

    def _passes(self) -> list[tuple[object, RegionPass | None]]:
        """Returns the inference configuration and region pass of every scoring pass.

        Returns:
            The configuration itself with no region pass when it carries a region override or the split is one region, otherwise one copy per stored region with that region as override and a region-indexed output stamp, paired with the region pass naming its index, the region count and the shared launch stamp.
        """
        if self.config.region_override:
            return [(self.config, None)]

        regions = self._stored_regions()
        if len(regions) == 1:
            return [(self.config, None)]

        stamp = RunTag.now()

        return [(replace(self.config, region_override=list(region.as_tuple()), output_subdir=f"{stamp}_region{index}"), RegionPass(index=index, count=len(regions), group=stamp)) for index, region in enumerate(regions)]

    def run(self) -> None:
        """Runs the pipeline once per scoring pass, handing each its region pass."""
        for config, region_pass in self._passes():
            self.pipeline_cls(config, components=self.components, region_pass=region_pass).run()


class SpatialScoring:
    """Scores one backbone or dual run either on its own split or on the target dataset its configuration enables."""

    @staticmethod
    def run(inference_config, run_directory: Path, pipeline_cls, components) -> None:
        """Runs the spatial pipeline over one run.

        Args:
            inference_config : Inference block of the entry configuration.
            run_directory    : Training run to score.
            pipeline_cls     : Spatial inference pipeline class.
            components       : Inference component set handed to the pipeline.
        """
        if inference_config.target.enabled:
            config = replace(inference_config, run_directory=Path(run_directory), output_subdir=None)
            pipeline_cls(config, components=components).run()
            return

        config = SpatialRunConfig.build(inference_config, run_directory)

        SplitRegionLoop(config, pipeline_cls, components).run()


class BackboneInferenceRunner:
    """Runs spatial backbone inference, resolving the component set per run.

    Attributes:
        entry : Entry-point configuration holding the inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one backbone, dual or JEPA run with its resolved components, one pass per stored region of the split or one pass over an enabled target dataset.

        Args:
            run_directory : Training run to score; the output stamp is created under its inference/ tree, or its cross-dataset tree when a target dataset is enabled.
        """
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        components = InferenceComponentsResolver.for_run(Path(run_directory))

        SpatialScoring.run(self.entry.inference, run_directory, InferencePipeline, components)


class ProfileAeInferenceRunner:
    """Runs inference for a standalone 1-D profile autoencoder run.

    Attributes:
        entry : Entry-point configuration holding the profile inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one profile-autoencoder run.

        Args:
            run_directory : Training run to score.
        """
        from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferencePipeline

        config = replace(self.entry.profile_inference, run_directory=Path(run_directory), output_subdir=None)

        ProfileAeInferencePipeline(config).run()


class ImageAeInferenceRunner:
    """Runs inference for a standalone 2-D image autoencoder run.

    Attributes:
        entry : Entry-point configuration holding the image inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one image-autoencoder run.

        Args:
            run_directory : Training run to score.
        """
        from pipelines.image_autoencoder.inference.pipeline import ImageAeInferencePipeline

        config = replace(self.entry.image_inference, run_directory=Path(run_directory), output_subdir=None)

        ImageAeInferencePipeline(config).run()


class DualInferenceRunner:
    """Runs spatial inference for a dual-trunk run with the dual component set.

    Attributes:
        entry : Entry-point configuration holding the inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one dual-trunk run, one pass per stored region of the split or one pass over an enabled target dataset.

        Args:
            run_directory : Training run to score.
        """
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.dual.inference.pipeline     import DUAL_INFERENCE_COMPONENTS

        SpatialScoring.run(self.entry.inference, run_directory, InferencePipeline, DUAL_INFERENCE_COMPONENTS)
