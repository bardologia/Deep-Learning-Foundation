
from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

from pipelines.shared.inference.metadata import RegionPass
from tools.data.io                       import FileIO
from tools.data.regions                  import CropRegion, SplitRegions
from tools.runtime.run_tag               import RunTag


class SpatialRunConfig:

    @staticmethod
    def build(inference_config, run_directory: Path):
        from pipelines.cross_validation.folds import FoldUnitManifest

        config = replace(inference_config, run_directory=Path(run_directory), output_subdir=None)

        if config.region_override:
            return config

        region = FoldUnitManifest.inference_region(Path(run_directory), config.split)
        if region is not None:
            config.region_override = list(region.as_tuple())

        return config


class SplitRegionLoop:

    def __init__(self, config, pipeline_cls) -> None:
        self.config       = config
        self.pipeline_cls = pipeline_cls

    def _stored_regions(self) -> list[CropRegion]:
        payload = FileIO.load_json(self.config.run_directory / "meta" / "dataset_creation_config.json")

        return SplitRegions.from_payload(payload["split_regions"]).regions(self.config.split)

    def _passes(self) -> list[tuple[object, RegionPass | None]]:
        if self.config.region_override:
            return [(self.config, None)]

        regions = self._stored_regions()
        if len(regions) == 1:
            return [(self.config, None)]

        stamp = RunTag.now()

        return [(replace(self.config, region_override=list(region.as_tuple()), output_subdir=f"{stamp}_region{index}"), RegionPass(index=index, count=len(regions), group=stamp)) for index, region in enumerate(regions)]

    def run(self) -> None:
        for config, region_pass in self._passes():
            self.pipeline_cls(config, region_pass=region_pass).run()


class BackboneInferenceRunner:

    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline

        inference_config = self.entry.inference

        if inference_config.target.enabled:
            InferencePipeline(replace(inference_config, run_directory=Path(run_directory), output_subdir=None)).run()
            return

        SplitRegionLoop(SpatialRunConfig.build(inference_config, run_directory), InferencePipeline).run()
