from configuration.architectures.backbone import (
    UNetConfig,
    ResUNetConfig,
    UNetSkipConfig,
    AttentionUNetConfig,
    UNetPlusPlusConfig,
    LinkNetConfig,
    SwinUNetConfig,
    TransUNetConfig,
    UNETRConfig,
    DeepLabV3PlusConfig,
    SegFormerLiteConfig,
    ConvNeXtUNetConfig,
    DenseUNetConfig,
    HRNetLiteConfig,
    MultiResUNetConfig,
    FPNNetConfig,
    U2NetLiteConfig,
    PixelMLPNetConfig,
    LocalCNNConfig,
    NAFNetConfig,
)
from configuration.architectures.unrolled import (
    GammaNetConfig,
)
from configuration.architectures.dual import (
    DualResUNetConfig,
)
from configuration.architectures.image_autoencoder import (
    ImageAutoencoderBaseConfig,
    Conv2dImageAutoencoderConfig,
    ResNet2dImageAutoencoderConfig,
    ConvNeXt2dImageAutoencoderConfig,
    DilatedConv2dImageAutoencoderConfig,
    ViTImageAutoencoderConfig,
)
from configuration.architectures.profile_autoencoder import (
    ProfileAutoencoderBaseConfig,
    MlpAutoencoderConfig,
    Conv1dAutoencoderConfig,
    Transformer1dAutoencoderConfig,
    ResMlpAutoencoderConfig,
    TcnAutoencoderConfig,
    GruAutoencoderConfig,
    CnnAttnAutoencoderConfig,
)
