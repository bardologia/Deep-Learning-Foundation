"""Inference configuration dataclasses for every model family.

Re-exports the shared backbone inference config together with the per-family
entry configs consumed by the `main/inference` entry points.
"""

from configuration.inference.general import (
    InferencePaths,
    InferenceConfig,
)
from configuration.inference.backbone            import BackboneInferenceEntryConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceEntryConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceEntryConfig
from configuration.inference.dual                import DualInferenceEntryConfig
