from configuration.architectures import Conv2dImageAutoencoderConfig, ImageAutoencoderBaseConfig, ResNet2dImageAutoencoderConfig, ConvNeXt2dImageAutoencoderConfig, DilatedConv2dImageAutoencoderConfig, ViTImageAutoencoderConfig
from ..registry                  import RegistryFactory
from .base                       import ImageAutoencoderBase
from .conv2d                     import Conv2dImageAutoencoder
from .resnet2d                   import ResNet2dImageAutoencoder
from .convnext2d                 import ConvNeXt2dImageAutoencoder
from .dilated2d                  import DilatedConv2dImageAutoencoder
from .vit                        import ViTImageAutoencoder


IMAGE_AE_MODEL_REGISTRY: dict[str, type] = {
    "conv2d_ae"     : Conv2dImageAutoencoder,
    "resnet2d_ae"   : ResNet2dImageAutoencoder,
    "convnext2d_ae" : ConvNeXt2dImageAutoencoder,
    "dilated2d_ae"  : DilatedConv2dImageAutoencoder,
    "vit_ae"        : ViTImageAutoencoder,
}

IMAGE_AE_CONFIG_REGISTRY: dict[str, type] = {
    "conv2d_ae"     : Conv2dImageAutoencoderConfig,
    "resnet2d_ae"   : ResNet2dImageAutoencoderConfig,
    "convnext2d_ae" : ConvNeXt2dImageAutoencoderConfig,
    "dilated2d_ae"  : DilatedConv2dImageAutoencoderConfig,
    "vit_ae"        : ViTImageAutoencoderConfig,
}


get_image_autoencoder = RegistryFactory(IMAGE_AE_MODEL_REGISTRY, IMAGE_AE_CONFIG_REGISTRY, "image autoencoder").build


__all__ = [
    "ImageAutoencoderBase",
    "Conv2dImageAutoencoder",
    "ResNet2dImageAutoencoder",
    "ConvNeXt2dImageAutoencoder",
    "DilatedConv2dImageAutoencoder",
    "ViTImageAutoencoder",
    "ImageAutoencoderBaseConfig",
    "Conv2dImageAutoencoderConfig",
    "ResNet2dImageAutoencoderConfig",
    "ConvNeXt2dImageAutoencoderConfig",
    "DilatedConv2dImageAutoencoderConfig",
    "ViTImageAutoencoderConfig",
    "IMAGE_AE_MODEL_REGISTRY",
    "IMAGE_AE_CONFIG_REGISTRY",
    "get_image_autoencoder",
]
