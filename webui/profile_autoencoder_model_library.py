from __future__ import annotations

from configuration.architectures import profile_autoencoder

from model_library_base import ModelDefaultsLibrary


class ProfileAutoencoderModelLibrary(ModelDefaultsLibrary):

    CONFIG_MODULE      = profile_autoencoder
    NORMALIZATION_ATTR = "embedding_norm"

    CONFIG_CLASSES = {
        "mlp_ae"           : "MlpAutoencoderConfig",
        "conv1d_ae"        : "Conv1dAutoencoderConfig",
        "transformer1d_ae" : "Transformer1dAutoencoderConfig",
        "resmlp_ae"        : "ResMlpAutoencoderConfig",
        "tcn_ae"           : "TcnAutoencoderConfig",
        "gru_ae"           : "GruAutoencoderConfig",
        "cnn_attn_ae"      : "CnnAttnAutoencoderConfig",
    }

    NOTE_FILES = {
        "mlp_ae"           : "MLP Autoencoder.md",
        "conv1d_ae"        : "Conv1D Autoencoder.md",
        "transformer1d_ae" : "Transformer1D Autoencoder.md",
        "resmlp_ae"        : "ResMLP Autoencoder.md",
        "tcn_ae"           : "TCN Autoencoder.md",
        "gru_ae"           : "GRU Autoencoder.md",
        "cnn_attn_ae"      : "Conv-Attention Autoencoder.md",
    }

    FALLBACK_ACTIVATION    = "gelu"
    FALLBACK_NORMALIZATION = "none"

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "Profile autoencoder",
                "blurb"  : "Compress the fitted, normalized elevation profile into a latent embedding and reconstruct it. The encoder defines the output latent space later predicted by JEPA.",
                "models" : [
                    {
                        "key": "mlp_ae", "name": "MLP Autoencoder", "skip": "Symmetric MLP",
                        "head": "Dense to embedding", "params": "~1.86M", "recommended": True,
                        "when": "The default starting point. Treats the profile as a flat vector; a symmetric dense encoder and decoder compress it to the embedding and reconstruct it. Cheapest and strongest baseline.",
                    },
                    {
                        "key": "conv1d_ae", "name": "Conv1D Autoencoder", "skip": "1D convolutions",
                        "head": "Conv stack to embedding", "params": "~1.60M", "recommended": False,
                        "when": "Exploits the local smoothness of the elevation profile. Stacked 1D convolutions over the range axis capture neighbouring-bin correlations before pooling to the embedding.",
                    },
                    {
                        "key": "transformer1d_ae", "name": "Transformer1D Autoencoder", "skip": "Self-attention",
                        "head": "Transformer to embedding", "params": "~1.89M", "recommended": False,
                        "when": "Long-range dependencies along the profile. A self-attention encoder and decoder model interactions between distant elevation bins, at a higher parameter cost.",
                    },
                    {
                        "key": "resmlp_ae", "name": "ResMLP Autoencoder", "skip": "Residual MLP",
                        "head": "Dense to embedding", "params": "~1.99M", "recommended": False,
                        "when": "A deeper dense alternative to the MLP baseline. Pre-norm residual blocks let the encoder and decoder go deeper without optimisation trouble, trading parameters for capacity while keeping the flat-vector treatment of the profile.",
                    },
                    {
                        "key": "tcn_ae", "name": "TCN Autoencoder", "skip": "Dilated 1D conv",
                        "head": "Dilated conv to embedding", "params": "~1.61M", "recommended": False,
                        "when": "Multi-scale local structure. Stacked dilated residual convolutions grow the receptive field exponentially over the range axis, capturing both narrow and broad elevation features without the cost of attention.",
                    },
                    {
                        "key": "gru_ae", "name": "GRU Autoencoder", "skip": "Recurrent",
                        "head": "BiGRU to embedding", "params": "~1.83M", "recommended": False,
                        "when": "A compact recurrent option. A bidirectional GRU sweeps the profile sequentially and a GRU decoder unrolls the embedding back into the curve; the cheapest model that still models ordering explicitly.",
                    },
                    {
                        "key": "cnn_attn_ae", "name": "Conv-Attention Autoencoder", "skip": "Conv tokens + attention",
                        "head": "Tokenized transformer to embedding", "params": "~1.92M", "recommended": False,
                        "when": "The most expressive profile model. A convolutional tokenizer splits the profile into patches that a real multi-token self-attention stack relates to one another, combining local convolution with global attention at the highest parameter cost.",
                    },
                ],
            },
        ]
