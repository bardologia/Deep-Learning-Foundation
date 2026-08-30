"""Per-family adapters that run inference on one trained run directory.

Each runner takes the entry-point configuration, repoints its inference block at
a given run directory, and drives the matching pipeline. Pipeline imports are
deferred so that selecting a family does not pull in the others.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib     import Path


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

        config     = replace(self.entry.inference, run_directory=Path(run_directory), output_subdir=None)
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

        config = replace(self.entry.inference, run_directory=Path(run_directory), output_subdir=None)

        InferencePipeline(config, components=DUAL_INFERENCE_COMPONENTS).run()
