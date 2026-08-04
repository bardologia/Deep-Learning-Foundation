from __future__ import annotations

from configuration.architectures import backbone

from model_library_base import ModelDefaultsLibrary


class BackboneModelLibrary(ModelDefaultsLibrary):

    CONFIG_MODULE = backbone

    CONFIG_CLASSES = {
        "unet"           : "UNetConfig",
        "unet_skip"      : "UNetSkipConfig",
        "resunet"        : "ResUNetConfig",
        "attention_unet" : "AttentionUNetConfig",
        "unetplusplus"   : "UNetPlusPlusConfig",
        "linknet"        : "LinkNetConfig",
        "swin_unet"      : "SwinUNetConfig",
        "transunet"      : "TransUNetConfig",
        "unetr"          : "UNETRConfig",
        "deeplabv3plus"  : "DeepLabV3PlusConfig",
        "segformer"      : "SegFormerLiteConfig",
        "convnext_unet"  : "ConvNeXtUNetConfig",
        "dense_unet"     : "DenseUNetConfig",
        "hrnet"          : "HRNetLiteConfig",
        "multires_unet"  : "MultiResUNetConfig",
        "fpn"            : "FPNNetConfig",
        "u2net"          : "U2NetLiteConfig",
        "pixel_mlp"      : "PixelMLPNetConfig",
        "local_cnn"      : "LocalCNNConfig",
        "nafnet"         : "NAFNetConfig",
    }

    NOTE_FILES = {
        "unet"           : "UNet.md",
        "unet_skip"      : "UNet Skip.md",
        "resunet"        : "ResUNet.md",
        "attention_unet" : "Attention UNet.md",
        "unetplusplus"   : "UNet++.md",
        "linknet"        : "LinkNet.md",
        "swin_unet"      : "SwinUNet.md",
        "transunet"      : "TransUNet.md",
        "unetr"          : "UNETR.md",
        "deeplabv3plus"  : "DeepLabV3+.md",
        "segformer"      : "SegFormer.md",
        "convnext_unet"  : "ConvNeXt UNet.md",
        "dense_unet"     : "DenseUNet.md",
        "hrnet"          : "HRNet.md",
        "multires_unet"  : "MultiResUNet.md",
        "fpn"            : "FPN.md",
        "u2net"          : "U2-Net.md",
        "pixel_mlp"      : "PixelMLP.md",
        "local_cnn"      : "Local CNN.md",
        "nafnet"         : "NAFNet.md",
    }

    HEAD_NOTE_FILES = {
        "conv"         : "Head Conv.md",
        "multihead"    : "Head Multihead.md",
        "per_gaussian" : "Head Per-Gaussian.md",
        "set_pred"     : "Head Set-Prediction.md",
    }

    FALLBACK_ACTIVATION    = "gelu"
    FALLBACK_NORMALIZATION = "layer"

    DISPLAY_DEFAULTS = {
        "nafnet" : ("simplegate", "layer"),
    }

    def _note_files(self) -> dict[str, str]:
        return {**self.NOTE_FILES, **self.HEAD_NOTE_FILES}

    def heads(self) -> list[dict]:
        return [
            {
                "key": "conv", "name": "Conv", "structure": "Single conv projection",
                "default": True,
                "when": "The default head. One convolution maps the backbone embedding straight to the packed parameter channels; every parameter shares the same projection.",
            },
            {
                "key": "multihead", "name": "Multihead", "structure": "3 PixelMLP heads",
                "default": False,
                "when": "Systematic bias across parameter types. Separate PixelMLP heads regress amplitude, mean, and spread from the shared embedding.",
            },
            {
                "key": "per_gaussian", "name": "Per-Gaussian", "structure": "K PixelMLP heads",
                "default": False,
                "when": "Systematic bias across Gaussian slots. One PixelMLP head per slot imposes slot independence.",
            },
            {
                "key": "set_pred", "name": "Set-Prediction", "structure": "K PixelMLP heads + existence gate",
                "default": False,
                "when": "Attacking Gaussian slot collapse. Per-slot heads plus an existence-logit gate decouple the slot on/off decision from amplitude regression; pair with hungarian param matching.",
            },
        ]

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "CNN encoder-decoder",
                "blurb"  : "Convolutional U-shaped backbones. Strong baselines, modest parameter budgets.",
                "models" : [
                    {
                        "key": "unet", "name": "UNet", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~31.0M", "recommended": False,
                        "when": "The default starting point. Small datasets or limited GPU memory; minimal complexity, strong baseline.",
                    },
                    {
                        "key": "resunet", "name": "ResUNet", "skip": "Skip + residual",
                        "head": "Single 1x1 conv", "params": "~32.4M", "recommended": False,
                        "when": "Unstable or very deep training. Residual encoder blocks prevent gradient vanishing; stride-2 downsampling per Zhang et al. 2018.",
                    },
                    {
                        "key": "unet_skip", "name": "UNet Skip", "skip": "Skip + residual",
                        "head": "Single 1x1 conv", "params": "~32.4M", "recommended": True,
                        "when": "Continuity with results trained before 2026-06-04. Pre-correction ResUNet (residual blocks, MaxPool downsampling); the archived checkpoint and benchmark-winning results correspond to this architecture.",
                    },
                    {
                        "key": "attention_unet", "name": "Attention UNet", "skip": "Attention-gated concat",
                        "head": "Single 1x1 conv", "params": "~32.4M", "recommended": False,
                        "when": "Spatially heterogeneous scenes (mixed urban and vegetated). Gates suppress irrelevant skip features per region.",
                    },
                    {
                        "key": "unetplusplus", "name": "UNet++", "skip": "Nested dense skips",
                        "head": "Single 1x1 conv", "params": "~31.1M", "recommended": False,
                        "when": "Skip-connection quality is the bottleneck. Graduated fusion reduces the encoder-decoder semantic gap.",
                    },
                    {
                        "key": "linknet", "name": "LinkNet", "skip": "Additive skip",
                        "head": "Single 1x1 conv", "params": "~31.0M", "recommended": False,
                        "when": "Efficiency is the priority. Additive skips shrink decoder channels and the parameter budget.",
                    },
                    {
                        "key": "convnext_unet", "name": "ConvNeXt UNet", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~19.1M", "recommended": False,
                        "when": "Modern convolution design under the unchanged U topology. Large depthwise kernels, inverted bottlenecks, LayerNorm and GELU.",
                    },
                    {
                        "key": "dense_unet", "name": "DenseUNet", "skip": "Dense concat",
                        "head": "Single 1x1 conv", "params": "~1.0M", "recommended": False,
                        "when": "Parameter efficiency through feature reuse. At matched capacity the deepest, thinnest model in the zoo.",
                    },
                    {
                        "key": "multires_unet", "name": "MultiResUNet", "skip": "ResPath concat",
                        "head": "Single 1x1 conv", "params": "~14.4M", "recommended": False,
                        "when": "Structures span several spatial scales. Every block sees 3/5/7-pixel receptive fields in parallel, and ResPaths close the semantic gap.",
                    },
                    {
                        "key": "u2net", "name": "U2-Net", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~8.3M", "recommended": False,
                        "when": "Intra-stage multi-scale mixing is desired. Each stage is itself a small U-Net, so even full-resolution layers integrate wide context.",
                    },
                ],
            },
            {
                "family" : "Transformer",
                "blurb"  : "Global attention for long-range spatial dependencies, at a higher parameter cost.",
                "models" : [
                    {
                        "key": "swin_unet", "name": "Swin-UNet", "skip": "Hierarchical Swin",
                        "head": "1x1 conv", "params": "~28.8M", "recommended": False,
                        "when": "Large homogeneous structures spanning many pixels. Windowed attention scales better than full ViT.",
                    },
                    {
                        "key": "transunet", "name": "TransUNet", "skip": "Transformer patch tokens",
                        "head": "CNN decoder", "params": "~30.6M", "recommended": False,
                        "when": "CNN locality fused with global ViT context. Hybrid encoder; use when both fine detail and long-range structure matter.",
                    },
                    {
                        "key": "unetr", "name": "UNETR", "skip": "Transformer skip outputs",
                        "head": "CNN decoder", "params": "~34.4M", "recommended": False,
                        "when": "Pure ViT encoder with a CNN decoder. Global receptive field from the first layer.",
                    },
                    {
                        "key": "segformer", "name": "SegFormer", "skip": "Pyramid to MLP decoder",
                        "head": "1x1 conv", "params": "~5.2M", "recommended": False,
                        "when": "A strong hierarchical attention encoder with a near-trivial MLP decoder. The complementary hypothesis to the heavy-decoder transformers.",
                    },
                ],
            },
            {
                "family" : "Context and resolution",
                "blurb"  : "Dense-prediction designs that manage context and resolution differently from the symmetric U.",
                "models" : [
                    {
                        "key": "deeplabv3plus", "name": "DeepLabV3+", "skip": "Low-level fusion",
                        "head": "ASPP + decoder", "params": "~10.3M", "recommended": False,
                        "when": "Multi-scale context at fixed dilation rates matters. ASPP aggregates several receptive fields in parallel without extra downsampling.",
                    },
                    {
                        "key": "hrnet", "name": "HRNet", "skip": "Branch fusion",
                        "head": "Concat + 1x1 conv", "params": "~3.1M", "recommended": False,
                        "when": "Per-pixel position accuracy is the priority. The full-resolution stream is never downsampled, avoiding the encode-decode round trip.",
                    },
                    {
                        "key": "fpn", "name": "FPN", "skip": "Lateral additions",
                        "head": "Pyramid sum + 1x1 conv", "params": "~7.8M", "recommended": False,
                        "when": "Probing how much decoder capacity the task needs. The minimal decoder concentrates parameters in the encoder.",
                    },
                ],
            },
            {
                "family" : "Restoration",
                "blurb"  : "Image-restoration designs built for high per-pixel fidelity rather than semantic abstraction.",
                "models" : [
                    {
                        "key": "nafnet", "name": "NAFNet", "skip": "Additive skip",
                        "head": "Single 3x3 conv", "params": "~29.2M", "recommended": False,
                        "when": "Testing the restoration hypothesis: dense continuous regression is closer to restoration than segmentation. Gated blocks and channel attention with no conventional activations.",
                    },
                ],
            },
            {
                "family" : "Controls",
                "blurb"  : "Scientific control baselines that bound how much spatial context contributes at all.",
                "models" : [
                    {
                        "key": "pixel_mlp", "name": "PixelMLP", "skip": "None (single stream)",
                        "head": "Single 1x1 conv", "params": "~30.8M", "recommended": False,
                        "when": "The no-spatial-context control. A per-pixel MLP of 1x1 convolutions; the margin any spatial backbone holds over it is the measured value of spatial context.",
                    },
                    {
                        "key": "local_cnn", "name": "Local CNN", "skip": "None (single stream)",
                        "head": "Single 1x1 conv", "params": "~31.2M", "recommended": False,
                        "when": "The local-context-only control. Full-resolution 3x3 ConvBlocks with a fixed small receptive field, between PixelMLP and the encode-decode backbones.",
                    },
                ],
            },
        ]
