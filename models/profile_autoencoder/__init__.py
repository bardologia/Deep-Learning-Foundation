from configuration.architectures import ProfileAutoencoderBaseConfig, Conv1dAutoencoderConfig, MlpAutoencoderConfig, Transformer1dAutoencoderConfig, ResMlpAutoencoderConfig, TcnAutoencoderConfig, GruAutoencoderConfig, CnnAttnAutoencoderConfig
from ..registry                  import RegistryFactory
from .base                       import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from .mlp                        import MlpAutoencoder
from .conv1d                     import Conv1dAutoencoder
from .transformer1d              import Transformer1dAutoencoder
from .resmlp                     import ResMlpAutoencoder
from .tcn                        import TcnAutoencoder
from .gru                        import GruAutoencoder
from .cnn_attn                   import CnnAttnAutoencoder


PROFILE_AE_MODEL_REGISTRY: dict[str, type] = {
    "mlp_ae"           : MlpAutoencoder,
    "conv1d_ae"        : Conv1dAutoencoder,
    "transformer1d_ae" : Transformer1dAutoencoder,
    "resmlp_ae"        : ResMlpAutoencoder,
    "tcn_ae"           : TcnAutoencoder,
    "gru_ae"           : GruAutoencoder,
    "cnn_attn_ae"      : CnnAttnAutoencoder,
}

PROFILE_AE_CONFIG_REGISTRY: dict[str, type] = {
    "mlp_ae"           : MlpAutoencoderConfig,
    "conv1d_ae"        : Conv1dAutoencoderConfig,
    "transformer1d_ae" : Transformer1dAutoencoderConfig,
    "resmlp_ae"        : ResMlpAutoencoderConfig,
    "tcn_ae"           : TcnAutoencoderConfig,
    "gru_ae"           : GruAutoencoderConfig,
    "cnn_attn_ae"      : CnnAttnAutoencoderConfig,
}


get_profile_autoencoder = RegistryFactory(PROFILE_AE_MODEL_REGISTRY, PROFILE_AE_CONFIG_REGISTRY, "autoencoder").build


__all__ = [
    "ProfileAutoencoderBase",
    "ProfileAutoencoderBlocks",
    "MlpAutoencoder",
    "Conv1dAutoencoder",
    "Transformer1dAutoencoder",
    "ResMlpAutoencoder",
    "TcnAutoencoder",
    "GruAutoencoder",
    "CnnAttnAutoencoder",
    "ProfileAutoencoderBaseConfig",
    "MlpAutoencoderConfig",
    "Conv1dAutoencoderConfig",
    "Transformer1dAutoencoderConfig",
    "ResMlpAutoencoderConfig",
    "TcnAutoencoderConfig",
    "GruAutoencoderConfig",
    "CnnAttnAutoencoderConfig",
    "PROFILE_AE_MODEL_REGISTRY",
    "PROFILE_AE_CONFIG_REGISTRY",
    "get_profile_autoencoder",
]
