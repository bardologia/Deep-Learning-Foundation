from .backbone import *  # noqa: F401,F403
from .backbone import BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS, BACKBONE_IMAGE_SIZE_MODELS, BACKBONE_MODEL_REGISTRY, get_backbone
from .blocks   import DropPath, build_activation, build_norm2d, build_upsample, initialize_weights
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
from .unrolled import (
    UNROLLED_CONFIG_REGISTRY,
    UNROLLED_MODEL_REGISTRY,
    GammaNet,
    TomoOperator,
    get_unrolled,
)
from .dual import (
    DUAL_CONFIG_REGISTRY,
    DUAL_MODEL_REGISTRY,
    DualResUNet,
    get_dual,
)

def config_registry(training_type: str) -> dict:
    if training_type == "profile_autoencoder":
        return PROFILE_AE_CONFIG_REGISTRY
    if training_type == "image_autoencoder":
        return IMAGE_AE_CONFIG_REGISTRY
    if training_type == "unrolled":
        return UNROLLED_CONFIG_REGISTRY
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
    "build_activation",
    "build_norm2d",
    "build_upsample",
    "initialize_weights",
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
    "UNROLLED_CONFIG_REGISTRY",
    "UNROLLED_MODEL_REGISTRY",
    "GammaNet",
    "TomoOperator",
    "get_unrolled",
    "DUAL_CONFIG_REGISTRY",
    "DUAL_MODEL_REGISTRY",
    "DualResUNet",
    "get_dual",
]
