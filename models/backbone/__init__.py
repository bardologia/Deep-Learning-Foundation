from configuration.architectures import AttentionUNetConfig, ConvNeXtUNetConfig, DeepLabV3PlusConfig, DenseUNetConfig, FPNNetConfig, HRNetLiteConfig, LinkNetConfig, LocalCNNConfig, MultiResUNetConfig, NAFNetConfig, PixelMLPNetConfig, ResUNetConfig, SegFormerLiteConfig, SwinUNetConfig, TransUNetConfig, U2NetLiteConfig, UNETRConfig, UNetConfig, UNetPlusPlusConfig, UNetSkipConfig
from ..blocks         import DropPath, build_activation, build_norm2d, build_upsample, initialize_weights
from ..registry       import RegistryFactory
from .unet            import UNet
from .resunet         import ResUNet, UNetSkip
from .attention_unet  import AttentionUNet
from .unet_plus_plus  import UNetPlusPlus
from .link_net        import LinkNet
from .swin_unet       import SwinUNet
from .trans_unet      import TransUNet
from .unetr           import UNETR
from .deeplab_v3_plus import DeepLabV3Plus
from .segformer_lite  import SegFormerLite
from .convnext_unet   import ConvNeXtUNet
from .dense_unet      import DenseUNet
from .hrnet_lite      import HRNetLite
from .multires_unet   import MultiResUNet
from .fpn_net         import FPNNet
from .u2net_lite      import U2NetLite
from .pixel_baselines import LocalCNN, PixelMLPNet
from .nafnet          import NAFNet

BACKBONE_MODEL_REGISTRY: dict[str, type] = {
    "unet"           : UNet,
    "unet_skip"      : UNetSkip,
    "resunet"        : ResUNet,
    "attention_unet" : AttentionUNet,
    "unetplusplus"   : UNetPlusPlus,
    "linknet"        : LinkNet,
    "swin_unet"      : SwinUNet,
    "transunet"      : TransUNet,
    "unetr"          : UNETR,
    "deeplabv3plus"  : DeepLabV3Plus,
    "segformer"      : SegFormerLite,
    "convnext_unet"  : ConvNeXtUNet,
    "dense_unet"     : DenseUNet,
    "hrnet"          : HRNetLite,
    "multires_unet"  : MultiResUNet,
    "fpn"            : FPNNet,
    "u2net"          : U2NetLite,
    "pixel_mlp"      : PixelMLPNet,
    "local_cnn"      : LocalCNN,
    "nafnet"         : NAFNet,
}

BACKBONE_CONFIG_REGISTRY: dict[str, type] = {
    "unet"           : UNetConfig,
    "unet_skip"      : UNetSkipConfig,
    "resunet"        : ResUNetConfig,
    "attention_unet" : AttentionUNetConfig,
    "unetplusplus"   : UNetPlusPlusConfig,
    "linknet"        : LinkNetConfig,
    "swin_unet"      : SwinUNetConfig,
    "transunet"      : TransUNetConfig,
    "unetr"          : UNETRConfig,
    "deeplabv3plus"  : DeepLabV3PlusConfig,
    "segformer"      : SegFormerLiteConfig,
    "convnext_unet"  : ConvNeXtUNetConfig,
    "dense_unet"     : DenseUNetConfig,
    "hrnet"          : HRNetLiteConfig,
    "multires_unet"  : MultiResUNetConfig,
    "fpn"            : FPNNetConfig,
    "u2net"          : U2NetLiteConfig,
    "pixel_mlp"      : PixelMLPNetConfig,
    "local_cnn"      : LocalCNNConfig,
    "nafnet"         : NAFNetConfig,
}


BACKBONE_HEADS: tuple[str, ...] = ("conv", "multihead", "per_gaussian", "set_pred")

BACKBONE_IMAGE_SIZE_MODELS: frozenset[str] = frozenset({"swin_unet", "transunet", "unetr"})


get_backbone = RegistryFactory(BACKBONE_MODEL_REGISTRY, BACKBONE_CONFIG_REGISTRY, "backbone").build


__all__ = [
    "UNet",
    "UNetSkip",
    "ResUNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "LinkNet",
    "SwinUNet",
    "TransUNet",
    "UNETR",
    "DeepLabV3Plus",
    "SegFormerLite",
    "ConvNeXtUNet",
    "DenseUNet",
    "HRNetLite",
    "MultiResUNet",
    "FPNNet",
    "U2NetLite",
    "PixelMLPNet",
    "LocalCNN",
    "NAFNet",
    "UNetConfig",
    "UNetSkipConfig",
    "ResUNetConfig",
    "AttentionUNetConfig",
    "UNetPlusPlusConfig",
    "LinkNetConfig",
    "SwinUNetConfig",
    "TransUNetConfig",
    "UNETRConfig",
    "DeepLabV3PlusConfig",
    "SegFormerLiteConfig",
    "ConvNeXtUNetConfig",
    "DenseUNetConfig",
    "HRNetLiteConfig",
    "MultiResUNetConfig",
    "FPNNetConfig",
    "U2NetLiteConfig",
    "PixelMLPNetConfig",
    "LocalCNNConfig",
    "NAFNetConfig",
    "get_backbone",
    "BACKBONE_MODEL_REGISTRY",
    "BACKBONE_CONFIG_REGISTRY",
    "BACKBONE_HEADS",
    "BACKBONE_IMAGE_SIZE_MODELS",
    "build_activation",
    "build_norm2d",
    "build_upsample",
    "initialize_weights",
    "DropPath",
]
