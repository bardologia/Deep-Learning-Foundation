"""Shared fixtures and tiny-model builders for the JEPA training and inference tests.

Provides a small MLP profile autoencoder, an identity profile normalizer and normalisation stats,
an z axis, a minimal embedding-loss config and a one-by-one convolutional stand-in
backbone, all sized by the module-level constants so the tests stay fast."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configuration.architectures                         import MlpAutoencoderConfig
from configuration.training.jepa                         import EmbeddingLossConfig
from models.profile_autoencoder                          import get_profile_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.conftest import SilentLogger


PROFILE_LENGTH = 8
EMBEDDING_DIM  = 4
HIDDEN_DIM     = 8
DEPTH          = 1
SPATIAL        = 3
N_GAUSSIANS    = 2


class IdentityNormStats:
    """Normalisation stats stub that returns predicted parameters unchanged."""
    def denormalize_output(self, params):
        """Returns the parameter tensor unchanged."""
        return params


def make_autoencoder(embedding_norm: str = "none"):
    """Builds a tiny MLP profile autoencoder with the requested embedding normalisation.

    Args:
        embedding_norm : Embedding normalisation mode passed to MlpAutoencoderConfig, one of "none", "l2" or "layernorm".

    Returns:
        The instantiated profile autoencoder module.
    """
    config = MlpAutoencoderConfig(
        profile_length = PROFILE_LENGTH,
        embedding_dim  = EMBEDDING_DIM,
        hidden_dim     = HIDDEN_DIM,
        depth          = DEPTH,
        embedding_norm = embedding_norm,
    )
    autoencoder, _ = get_profile_autoencoder("mlp_ae", config)
    return autoencoder


@pytest.fixture
def autoencoder():
    """Returns a tiny MLP profile autoencoder without embedding normalisation."""
    return make_autoencoder("none")


@pytest.fixture
def profile_normalizer():
    """Returns a profile normalizer that leaves curves untouched."""
    return ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))


@pytest.fixture
def norm_stats():
    """Returns normalisation stats that leave parameters untouched."""
    return IdentityNormStats()


@pytest.fixture
def z_axis():
    """Returns the z axis of shape (PROFILE_LENGTH,) spanning -4 to 4 metres."""
    return torch.linspace(-4.0, 4.0, PROFILE_LENGTH)


@pytest.fixture
def embedding_loss_cfg():
    """Returns an embedding-loss config with only the embedding MSE term active at weight one."""
    return EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=1.0, use_curve_recon=False)


@pytest.fixture
def fake_logger():
    """Returns a logger that swallows every message."""
    return SilentLogger()


@pytest.fixture
def tiny_backbone():
    """Returns a one-by-one convolution mapping two input channels to the embedding width."""
    return nn.Conv2d(2, EMBEDDING_DIM, kernel_size=1)
