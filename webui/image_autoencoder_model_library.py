"""Web UI catalogue of the image autoencoder architectures and their defaults."""

from __future__ import annotations

from configuration.architectures import image_autoencoder

from model_library_base import ModelDefaultsLibrary


class ImageAutoencoderModelLibrary(ModelDefaultsLibrary):
    """Serves the image autoencoder family, its config defaults and field hints.

    The class attributes bind each model key to its configuration dataclass and
    vault note, list the architecture fields hidden from the launch panel, and
    supply the one-line hint shown next to every exposed field.
    """

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

    ARCH_EXCLUDED = ("in_channels",)

    ARCH_EXCLUDED_BY_MODEL = {
        "dilated2d_ae" : ("downsample_factor", "depth"),
        "vit_ae"       : ("downsample_factor", "base_channels"),
    }

    ARCH_HINTS = {
        "embedding_dim"         : "channels of the spatial embedding; a coupled JEPA backbone consumes this stack as its input",
        "embedding_norm"        : "normalization applied to the embedding channels",
        "downsample_factor"     : "spatial reduction of the embedding grid relative to the input",
        "base_channels"         : "channels of the first encoder stage",
        "depth"                 : "number of stacked blocks or stages",
        "activation"            : "nonlinearity used throughout the encoder and decoder",
        "normalization"         : "feature normalization inside the convolutional blocks",
        "dropout"               : "dropout probability inside the blocks",
        "init_mode"             : "weight initialization scheme; default keeps the PyTorch layer defaults",
        "encoder_lr"            : "learning rate of the encoder parameter group",
        "decoder_lr"            : "learning rate of the decoder parameter group",
        "encoder_wd"            : "weight decay of the encoder parameter group",
        "decoder_wd"            : "weight decay of the decoder parameter group",
        "upsample_mode"         : "decoder upsampling operator",
        "stochastic_depth_rate" : "probability of dropping a residual branch during training",
        "dilation_depth"        : "number of dilated stages; each doubles the dilation",
        "patch_size"            : "pixels per token side before the attention stack",
        "hidden_dim"            : "width of the transformer token embedding",
        "num_heads"             : "attention heads per transformer block",
        "mlp_ratio"             : "feed-forward width as a multiple of the token embedding",
    }

    def _families(self) -> list[dict]:
        """Returns the single image autoencoder family with one entry per model.

        Returns:
            A one-element list holding the family name, its blurb and the model
            records (key, display name, skip and head labels, parameter count,
            recommendation flag and the guidance on when to pick it).
        """
        return [
            {
                "family" : "Image autoencoder",
                "blurb"  : "Compress the input channel stack into a 2D spatial embedding and reconstruct it. The encoder later serves as the JEPA image front-end.",
                "models" : [
                    {
                        "key": "conv2d_ae", "name": "Conv2D Image Autoencoder", "skip": "2D conv encoder/decoder",
                        "head": "Conv to 2D embedding", "params": "~48.4K", "recommended": True,
                        "when": "The default image autoencoder. A small 2D convolutional encoder downsamples the input channel stack to a spatial embedding and a mirrored decoder reconstructs it; the encoder is reused as the pretrained JEPA image front-end.",
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
