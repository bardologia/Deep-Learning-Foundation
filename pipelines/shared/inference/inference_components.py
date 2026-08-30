"""Selection of the inference component set implied by a run's persisted configs."""

from __future__ import annotations

from pathlib import Path

from pipelines.backbone.inference.pipeline      import InferenceComponents
from pipelines.dual.inference.pipeline          import DUAL_INFERENCE_COMPONENTS
from pipelines.jepa.inference.pipeline          import JEPA_INFERENCE_COMPONENTS, JEPA_PARAM_INFERENCE_COMPONENTS
from pipelines.shared.config.config_persistence import BackboneModelConfigIO, DualModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO


class InferenceComponentsResolver:
    """Picks the spatial inference components matching a trained run's family."""

    @staticmethod
    def for_run(run_directory: Path) -> InferenceComponents:
        """Returns the inference components a spatial run should be scored with.

        A dual config selects the dual components; a profile- or image-autoencoder
        config alongside a backbone config marks a JEPA run and selects its curve
        or parameter components; anything else falls back to the plain backbone
        components.

        Args:
            run_directory : Training run whose meta/ directory is inspected.

        Raises:
            ValueError : The run is a standalone autoencoder run, which this resolver does not cover.
        """
        meta         = Path(run_directory) / "meta"
        has_backbone = BackboneModelConfigIO.exists(meta)

        if DualModelConfigIO.exists(meta):
            return DUAL_INFERENCE_COMPONENTS

        if ProfileAutoencoderConfigIO.exists(meta):
            if not has_backbone:
                raise ValueError(f"Run '{run_directory}' is a standalone profile-autoencoder run, not a JEPA run; this resolver only selects backbone/JEPA spatial components. Score standalone profile-autoencoder runs with main/inference/infer_profile_autoencoder.py.")
            return JEPA_INFERENCE_COMPONENTS

        if ImageAutoencoderConfigIO.exists(meta):
            if not has_backbone:
                raise ValueError(f"Run '{run_directory}' is a standalone image-autoencoder run, not a JEPA run; this resolver only selects backbone/JEPA spatial components. Score standalone image-autoencoder runs with main/inference/infer_image_autoencoder.py.")
            return JEPA_PARAM_INFERENCE_COMPONENTS

        return InferenceComponents()
