"""Model-zoo entry for the profile autoencoder family shown in the web UI.

Maps each profile autoencoder key to its architecture config class, its vault
note, and the curated blurbs and per-field hints the launch panels render.
"""

from __future__ import annotations

from configuration.architectures import profile_autoencoder

from model_library_base import ModelDefaultsLibrary


class ProfileAutoencoderModelLibrary(ModelDefaultsLibrary):
    """Curated catalog of the seven profile autoencoder architectures."""

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

    ARCH_EXCLUDED = ("profile_length", "encoder_lr", "decoder_lr", "encoder_wd", "decoder_wd")

    ARCH_EXCLUDED_BY_MODEL = {
        "transformer1d_ae" : ("num_heads",),
    }

    ARCH_HINTS = {
        "embedding_dim"   : "size of the latent embedding; a coupled JEPA backbone later predicts this vector per pixel",
        "embedding_norm"  : "normalization applied to the embedding before the decoder and before JEPA matching",
        "activation"      : "nonlinearity used throughout the encoder and decoder",
        "init_mode"       : "weight initialization scheme; default keeps the PyTorch layer defaults",
        "hidden_dim"      : "width of the hidden layers",
        "depth"           : "number of stacked blocks or layers",
        "dropout"         : "dropout probability inside the blocks",
        "seq_channels"    : "channels of the 1D convolutional stack",
        "seq_kernel_size" : "kernel size of the 1D convolutions along the profile",
        "num_heads"         : "attention heads per transformer block",
        "patch_size"        : "profile bins per token before the attention stack",
        "bidirectional"     : "sweep the profile in both directions",
        "normalization"     : "normalization inside the encoder and decoder bodies; each family honors its supported modes and rejects the rest",
        "pooling"           : "reduction of the sequence or tokens to the latent vector: avg, max, or learned attention",
        "hidden_dims"       : "explicit per-layer hidden widths; leave empty to use depth copies of hidden_dim",
        "seq_channels_list" : "explicit per-layer channel widths; leave empty to use depth copies of seq_channels",
        "ffn_ratio"         : "feed-forward width as a multiple of the token width",
        "pos_embedding"     : "positional embedding of the tokens: learned, sinusoidal, or none",
    }

    def _families(self) -> list[dict]:
        """Returns the single profile-autoencoder family with its member model cards."""
        return [
            {
                "family" : "Profile autoencoder",
                "blurb"  : "Compress the fitted, normalized z profile into a latent embedding and reconstruct it. The encoder defines the output latent space later predicted by JEPA.",
                "models" : [
                    {
                        "key": "mlp_ae", "name": "MLP Autoencoder", "skip": "Symmetric MLP",
                        "head": "Dense to embedding", "params": "~1.86M", "recommended": True,
                        "when": "The default starting point. Treats the profile as a flat vector; a symmetric dense encoder and decoder compress it to the embedding and reconstruct it. Cheapest and strongest baseline.",
                    },
                    {
                        "key": "conv1d_ae", "name": "Conv1D Autoencoder", "skip": "1D convolutions",
                        "head": "Conv stack to embedding", "params": "~1.60M", "recommended": False,
                        "when": "Exploits the local smoothness of the z profile. Stacked 1D convolutions over the range axis capture neighbouring-bin correlations before pooling to the embedding.",
                    },
                    {
                        "key": "transformer1d_ae", "name": "Transformer1D Autoencoder", "skip": "Single-token transformer",
                        "head": "Transformer to embedding", "params": "~1.89M", "recommended": False,
                        "when": "A transformer stack over one token: the whole profile is embedded into a single vector, so the attention layers reduce to normalized residual feed-forward blocks and no bin-to-bin mixing happens. Pick cnn_attn_ae for real multi-token attention along the profile.",
                    },
                    {
                        "key": "resmlp_ae", "name": "ResMLP Autoencoder", "skip": "Residual MLP",
                        "head": "Dense to embedding", "params": "~1.99M", "recommended": False,
                        "when": "A deeper dense alternative to the MLP baseline. Pre-norm residual blocks let the encoder and decoder go deeper without optimisation trouble, trading parameters for capacity while keeping the flat-vector treatment of the profile.",
                    },
                    {
                        "key": "tcn_ae", "name": "TCN Autoencoder", "skip": "Dilated 1D conv",
                        "head": "Dilated conv to embedding", "params": "~1.61M", "recommended": False,
                        "when": "Multi-scale local structure. Stacked dilated residual convolutions grow the receptive field exponentially over the range axis, capturing both narrow and broad z features without the cost of attention.",
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
