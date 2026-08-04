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
    def denormalize_output(self, params):
        return params


def make_autoencoder(embedding_norm: str = "none"):
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
    return make_autoencoder("none")


@pytest.fixture
def profile_normalizer():
    return ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))


@pytest.fixture
def norm_stats():
    return IdentityNormStats()


@pytest.fixture
def x_axis():
    return torch.linspace(-4.0, 4.0, PROFILE_LENGTH)


@pytest.fixture
def embedding_loss_cfg():
    return EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=1.0, use_curve_recon=False)


@pytest.fixture
def fake_logger():
    return SilentLogger()


@pytest.fixture
def tiny_backbone():
    return nn.Conv2d(2, EMBEDDING_DIM, kernel_size=1)
