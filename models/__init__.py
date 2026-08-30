"""Public surface of the model zoo: backbones, autoencoders, unrolled and dual nets.

Re-exports every model class, its matching configuration dataclass and the
per-family registries, so that callers can build any architecture through a
single import and resolve the right configuration registry from a training-type
string.
"""

from .backbone import *  # noqa: F401,F403
from .backbone import BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS, BACKBONE_IMAGE_SIZE_MODELS, BACKBONE_MODEL_REGISTRY, get_backbone
from .blocks   import DropPath, LayerFactory
from .profile_autoencoder import (
    PROFILE_AE_CONFIG_REGISTRY,
    PROFILE_AE_MODEL_REGISTRY,
    ProfileAutoencoderBase,
    Conv1dAutoencoder,
    MlpAutoencoder,
    Transformer1dAutoencoder,
    ResMlpAutoencoder,
    TcnAutoencoder,
    GruAutoencoder,
    CnnAttnAutoencoder,
    get_profile_autoencoder,
)
from .image_autoencoder import (
    IMAGE_AE_CONFIG_REGISTRY,
    IMAGE_AE_MODEL_REGISTRY,
    Conv2dImageAutoencoder,
    ResNet2dImageAutoencoder,
    ConvNeXt2dImageAutoencoder,
    DilatedConv2dImageAutoencoder,
    ViTImageAutoencoder,
    ImageAutoencoderBase,
    get_image_autoencoder,
)
from .dual import (
    DUAL_CONFIG_REGISTRY,
    DUAL_MODEL_REGISTRY,
    DualResUNet,
    get_dual,
)

def config_registry(training_type: str) -> dict:
    """Returns the architecture-configuration registry for a training type.

    Args:
        training_type : Training family name, one of ``profile_autoencoder``, ``image_autoencoder``, ``unrolled`` or ``dual``; any other value falls back to the backbone registry.

    Returns:
        Mapping from architecture name to its configuration dataclass.
    """
    if training_type == "profile_autoencoder":
        return PROFILE_AE_CONFIG_REGISTRY
    if training_type == "image_autoencoder":
        return IMAGE_AE_CONFIG_REGISTRY
    if training_type == "dual":
        return DUAL_CONFIG_REGISTRY
    return BACKBONE_CONFIG_REGISTRY


__all__ = [
    "BACKBONE_CONFIG_REGISTRY",
    "config_registry",
    "BACKBONE_MODEL_REGISTRY",
    "BACKBONE_HEADS",
    "BACKBONE_IMAGE_SIZE_MODELS",
    "get_backbone",
    "DropPath",
    "LayerFactory",
    "PROFILE_AE_CONFIG_REGISTRY",
    "PROFILE_AE_MODEL_REGISTRY",
    "ProfileAutoencoderBase",
    "MlpAutoencoder",
    "Conv1dAutoencoder",
    "Transformer1dAutoencoder",
    "ResMlpAutoencoder",
    "TcnAutoencoder",
    "GruAutoencoder",
    "CnnAttnAutoencoder",
    "get_profile_autoencoder",
    "IMAGE_AE_CONFIG_REGISTRY",
    "IMAGE_AE_MODEL_REGISTRY",
    "ImageAutoencoderBase",
    "Conv2dImageAutoencoder",
    "ResNet2dImageAutoencoder",
    "ConvNeXt2dImageAutoencoder",
    "DilatedConv2dImageAutoencoder",
    "ViTImageAutoencoder",
    "get_image_autoencoder",
    "DUAL_CONFIG_REGISTRY",
    "DUAL_MODEL_REGISTRY",
    "DualResUNet",
    "get_dual",
]
