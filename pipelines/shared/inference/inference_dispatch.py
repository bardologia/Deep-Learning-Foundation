from __future__ import annotations

from dataclasses import replace
from pathlib     import Path


class BackboneInferenceRunner:
    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        config     = replace(self.entry.inference, run_directory=Path(run_directory), output_subdir=None)
        components = InferenceComponentsResolver.for_run(Path(run_directory))

        InferencePipeline(config, components=components).run()


class ProfileAeInferenceRunner:
    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferencePipeline

        config = replace(self.entry.profile_inference, run_directory=Path(run_directory), output_subdir=None)

        ProfileAeInferencePipeline(config).run()


class ImageAeInferenceRunner:
    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        from pipelines.image_autoencoder.inference.pipeline import ImageAeInferencePipeline

        config = replace(self.entry.image_inference, run_directory=Path(run_directory), output_subdir=None)

        ImageAeInferencePipeline(config).run()


class UnrolledInferenceRunner:
    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        from pipelines.unrolled.inference.pipeline import UnrolledInferencePipeline

        config = replace(self.entry.unrolled_inference, run_directory=Path(run_directory), output_subdir=None)

        UnrolledInferencePipeline(config).run()


class DualInferenceRunner:
    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.dual.inference.pipeline     import DUAL_INFERENCE_COMPONENTS

        config = replace(self.entry.inference, run_directory=Path(run_directory), output_subdir=None)

        InferencePipeline(config, components=DUAL_INFERENCE_COMPONENTS).run()
