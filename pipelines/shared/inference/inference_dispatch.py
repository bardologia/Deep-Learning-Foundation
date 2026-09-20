"""Per-family adapters that run inference on one trained run directory.

Each runner takes the entry-point configuration, repoints its inference block at
a given run directory, and drives the matching pipeline. A spatial run that is a
cross-validation fold unit is scored over the row block its manifest records,
so a fold can be re-scored by the generic entry. Pipeline imports are deferred
so that selecting a family does not pull in the others.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib     import Path


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


class BackboneInferenceRunner:
    """Runs spatial backbone inference, resolving the component set per run.

    Attributes:
        entry : Entry-point configuration holding the inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one backbone, dual or JEPA run with its resolved components.

        Args:
            run_directory : Training run to score; the output stamp is created under its inference/ tree.
        """
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        config     = SpatialRunConfig.build(self.entry.inference, run_directory)
        components = InferenceComponentsResolver.for_run(Path(run_directory))

        InferencePipeline(config, components=components).run()


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


class UnrolledInferenceRunner:
    """Runs inference for an unrolled optimisation run.

    Attributes:
        entry : Entry-point configuration holding the unrolled inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one unrolled run.

        Args:
            run_directory : Training run to score.
        """
        from pipelines.unrolled.inference.pipeline import UnrolledInferencePipeline

        config = replace(self.entry.unrolled_inference, run_directory=Path(run_directory), output_subdir=None)

        UnrolledInferencePipeline(config).run()


class DualInferenceRunner:
    """Runs spatial inference for a dual-trunk run with the dual component set.

    Attributes:
        entry : Entry-point configuration holding the inference block.
    """

    def __init__(self, entry_config) -> None:
        """Stores the entry-point configuration to derive per-run configs from."""
        self.entry = entry_config

    def run(self, run_directory: Path) -> None:
        """Scores one dual-trunk run.

        Args:
            run_directory : Training run to score.
        """
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.dual.inference.pipeline     import DUAL_INFERENCE_COMPONENTS

        config = SpatialRunConfig.build(self.entry.inference, run_directory)

        InferencePipeline(config, components=DUAL_INFERENCE_COMPONENTS).run()
