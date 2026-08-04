from __future__ import annotations

from configuration.architectures import image_autoencoder

from model_library_base import ModelDefaultsLibrary


class ImageAutoencoderModelLibrary(ModelDefaultsLibrary):

    CONFIG_MODULE = image_autoencoder

    CONFIG_CLASSES = {
        "conv2d_ae"     : "Conv2dImageAutoencoderConfig",
        "resnet2d_ae"   : "ResNet2dImageAutoencoderConfig",
        "convnext2d_ae" : "ConvNeXt2dImageAutoencoderConfig",
        "dilated2d_ae"  : "DilatedConv2dImageAutoencoderConfig",
        "vit_ae"        : "ViTImageAutoencoderConfig",
    }

    NOTE_FILES = {
        "conv2d_ae"     : "Conv2D Image Autoencoder.md",
        "resnet2d_ae"   : "ResNet2D Image Autoencoder.md",
        "convnext2d_ae" : "ConvNeXt2D Image Autoencoder.md",
        "dilated2d_ae"  : "Dilated2D Image Autoencoder.md",
        "vit_ae"        : "ViT Image Autoencoder.md",
    }

    FALLBACK_ACTIVATION    = "gelu"
    FALLBACK_NORMALIZATION = "batch"

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "Image autoencoder",
                "blurb"  : "Compress the SAR image stack into a 2D spatial embedding and reconstruct it. The encoder later serves as the JEPA image front-end.",
                "models" : [
                    {
                        "key": "conv2d_ae", "name": "Conv2D Image Autoencoder", "skip": "2D conv encoder/decoder",
                        "head": "Conv to 2D embedding", "params": "~48.4K", "recommended": True,
                        "when": "The default image autoencoder. A small 2D convolutional encoder downsamples the SAR image stack to a spatial embedding and a mirrored decoder reconstructs it; the encoder is reused as the pretrained JEPA image front-end.",
                    },
                    {
                        "key": "resnet2d_ae", "name": "ResNet2D Image Autoencoder", "skip": "Residual 2D conv",
                        "head": "Residual conv to 2D embedding", "params": "~208.2K", "recommended": False,
                        "when": "A deeper, residual alternative to the default. Pre-activation residual blocks with strided downsampling give a more expressive encoder that trains stably at greater depth, at a higher parameter cost.",
                    },
                    {
                        "key": "convnext2d_ae", "name": "ConvNeXt2D Image Autoencoder", "skip": "ConvNeXt blocks",
                        "head": "ConvNeXt to 2D embedding", "params": "~143.3K", "recommended": False,
                        "when": "A modern convolutional design. Depthwise 7x7 convolutions, layer normalisation and inverted bottlenecks capture wider spatial context than plain 3x3 stacks while staying parameter-efficient.",
                    },
                    {
                        "key": "dilated2d_ae", "name": "Dilated2D Image Autoencoder", "skip": "Atrous residual conv",
                        "head": "Dilated conv to 2D embedding", "params": "~113.3K", "recommended": False,
                        "when": "Preserves full spatial resolution. Stacked dilated residual convolutions grow the receptive field without downsampling, so the embedding keeps the input grid size; best when fine spatial detail must survive the bottleneck.",
                    },
                    {
                        "key": "vit_ae", "name": "ViT Image Autoencoder", "skip": "Patch attention",
                        "head": "Transformer to 2D embedding", "params": "~3.6M", "recommended": False,
                        "when": "The most expressive image model. Patches are tokenised and related by a transformer encoder and decoder with convolutional position encoding, modelling global structure across the stack at the highest parameter cost.",
                    },
                ],
            },
        ]
