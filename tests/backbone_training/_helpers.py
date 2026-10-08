"""Shared fixtures for the backbone training tests.

Provides the z axis, normalizers with known statistics, geometry and Gaussian
configs, loss and model builders, parameter tensors and a scalar-recording tracker stub.
"""

from __future__ import annotations

import numpy as np
import torch

from configuration.normalization           import ChannelStats, ChannelStrategy, NormMethod, OutputClampConfig
from configuration.dataset.gaussian        import GaussianConfig
from configuration.training.backbone       import BackboneTrainerConfig
from configuration.training.general.loss   import LossConfig
from pipelines.backbone.dataset.normalizer import Normalizer
from pipelines.backbone.dataset.stats      import Stats
from pipelines.backbone.training.loss      import Loss
from models                                import get_backbone
from tools.monitoring.logger               import NullLogger
from tools.monitoring.tracker              import NullTracker


X_MIN = -20.0
X_MAX = 80.0
X_LEN = 32


def x_axis_tensor(length: int = X_LEN) -> torch.Tensor:
    """Returns the z axis of shape (length,) spanning X_MIN to X_MAX, in metres."""
    return torch.linspace(X_MIN, X_MAX, length, dtype=torch.float32)


def x_axis_numpy(length: int = X_LEN) -> np.ndarray:
    """Returns the z axis of shape (length,) spanning X_MIN to X_MAX, in metres."""
    return np.linspace(X_MIN, X_MAX, length, dtype=np.float32)


def identity_normalizer(n_channels: int) -> Normalizer:
    """Returns a z-score output normalizer with zero location and unit scale on every channel."""
    strategy = ChannelStrategy(NormMethod.ZSCORE, apply_log1p=False)
    stats    = ChannelStats(
        loc        = [0.0] * n_channels,
        scale      = [1.0] * n_channels,
        names      = [f"c{i}" for i in range(n_channels)],
        strategies = [strategy] * n_channels,
        clampable  = [False] * n_channels,
    )
    return Normalizer(Stats(input_stats=None, output_stats=stats))


def log1p_normalizer(n_gaussians: int) -> Normalizer:
    """Returns an output normalizer applying log1p robust-IQR to amplitude and width and z-score to the centre."""
    log1p  = ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=True)
    zscore = ChannelStrategy(NormMethod.ZSCORE,     apply_log1p=False)

    stats = ChannelStats(
        loc        = [2.0, 10.0, 1.0] * n_gaussians,
        scale      = [1.5,  8.0, 0.7] * n_gaussians,
        names      = [f"G{g + 1}_{role}" for g in range(n_gaussians) for role in ("amp", "mu", "sigma")],
        strategies = [log1p, zscore, log1p] * n_gaussians,
        clampable  = [True, False, True] * n_gaussians,
    )
    return Normalizer(Stats(input_stats=None, output_stats=stats))


def zscore_normalizer(n_gaussians: int) -> Normalizer:
    """Returns a z-score output normalizer with the same statistics as log1p_normalizer but no log1p."""
    zscore = ChannelStrategy(NormMethod.ZSCORE, apply_log1p=False)

    stats = ChannelStats(
        loc        = [2.0, 10.0, 1.0] * n_gaussians,
        scale      = [1.5,  8.0, 0.7] * n_gaussians,
        names      = [f"G{g + 1}_{role}" for g in range(n_gaussians) for role in ("amp", "mu", "sigma")],
        strategies = [zscore] * (3 * n_gaussians),
        clampable  = [True, False, True] * n_gaussians,
    )
    return Normalizer(Stats(input_stats=None, output_stats=stats))


def spock_normalizer() -> Normalizer:
    """Returns a two-Gaussian fixed-bounds output normalizer with output clamping disabled."""
    fixed = ChannelStrategy(NormMethod.FIXED_BOUNDS)
    mins  = [1e-05, -15.0, 0.01, 1e-05, 1.0, 1.0]
    maxs  = [10.0, 5.0, 5.0, 10.0, 40.0, 20.0]

    stats = ChannelStats(
        loc        = mins,
        scale      = [hi - lo for lo, hi in zip(mins, maxs)],
        names      = [f"G{g + 1}_{role}" for g in range(2) for role in ("amp", "mu", "sigma")],
        strategies = [fixed] * 6,
        clampable  = [True, False, True] * 2,
    )

    normalizer = Normalizer(Stats(input_stats=None, output_stats=stats))
    normalizer.stats.clamp = OutputClampConfig(enabled=False)

    return normalizer


def gaussian_config(n_gaussians: int) -> GaussianConfig:
    """Returns a Gaussian config with n_gaussians slots over the X_MIN to X_MAX z range."""
    return GaussianConfig(n_default_gaussians=n_gaussians, x_min=X_MIN, x_max=X_MAX, activity_threshold=1e-4)


def build_loss(n_gaussians: int = 2, loss_cfg: LossConfig | None = None, log_all_losses: bool = False, length: int = X_LEN, sampler=None, norm_stats: Normalizer | None = None, logger=None) -> Loss:
    """Builds a training loss over the shared axis and configs.

    Args:
        n_gaussians    : Number of Gaussian slots, setting the 3 * n_gaussians channels.
        loss_cfg       : Loss configuration, defaulting to parameter L1 with unit weight.
        log_all_losses : Whether every loss term is logged individually.
        length         : Number of z bins on the axis.
        sampler        : Optional parameter sampler fed with denormalized predictions.
        norm_stats     : Output normalizer, defaulting to the identity normalizer.
        logger         : Logger receiving the configuration tables and warnings, a null logger when omitted.

    Returns:
        Loss bound to a null tracker.
    """
    n_channels = n_gaussians * 3
    loss_cfg   = loss_cfg if loss_cfg is not None else LossConfig(use_param_l1=True, weight_param_l1=1.0)

    return Loss(
        z_axis         = x_axis_tensor(length),
        logger         = logger if logger is not None else NullLogger(),
        tracker        = NullTracker(),
        gaussian_cfg   = gaussian_config(n_gaussians),
        loss_cfg       = loss_cfg,
        norm_stats     = norm_stats if norm_stats is not None else identity_normalizer(n_channels),
        log_all_losses = log_all_losses,
        sampler        = sampler,
    )


def param_tensor(batch: int, n_gaussians: int, height: int, width: int, seed: int) -> torch.Tensor:
    """Returns a standard normal parameter tensor of shape (batch, 3 * n_gaussians, height, width)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_gaussians * 3, height, width, generator=gen, dtype=torch.float32)


def valid_param_tensor(batch: int, n_gaussians: int, height: int, width: int, seed: int) -> torch.Tensor:
    """Returns a physically valid parameter tensor.

    Args:
        batch       : Number of samples.
        n_gaussians : Number of Gaussian slots.
        height      : Row extent, in pixels.
        width       : Range extent, in pixels.
        seed        : Seed of the generator drawing the parameters.

    Returns:
        Tensor of shape (batch, 3 * n_gaussians, height, width) with positive
        amplitudes, centres inside the z range in metres and widths of at
        least 4 metres.
    """
    gen   = torch.Generator().manual_seed(seed)
    parts = []

    for _ in range(n_gaussians):
        amp = 1.0 + torch.rand(batch, 1, height, width, generator=gen) * 5.0
        mu  = X_MIN + 10.0 + torch.rand(batch, 1, height, width, generator=gen) * (X_MAX - X_MIN - 20.0)
        sig = 4.0 + torch.rand(batch, 1, height, width, generator=gen) * 6.0
        parts.extend([amp, mu, sig])

    return torch.cat(parts, dim=1).float()


def tiny_model(in_channels: int = 2, n_gaussians: int = 2):
    """Returns a two-level ResUNet and its model config for the given channel and slot counts."""
    out_channels = n_gaussians * 3
    return get_backbone(
        "resunet",
        in_channels       = in_channels,
        out_channels      = out_channels,
        features          = [8, 16],
        bottleneck_factor = 1,
        dropout           = 0.0,
        normalization     = "instance",
    )


def tiny_trainer_config(n_gaussians: int = 2, epochs: int = 1) -> BackboneTrainerConfig:
    """Returns a backbone trainer config with resource monitoring off and parameter L1 as the complete-stage loss."""
    cfg = BackboneTrainerConfig(gaussian=gaussian_config(n_gaussians))

    cfg.training.epochs               = epochs
    cfg.training.validation_frequency = 1
    cfg.resources.enabled             = False

    cfg.curriculum.complete.use_param_l1    = True
    cfg.curriculum.complete.weight_param_l1 = 1.0

    return cfg


class TrackerStub:
    """Tracker recording every logged scalar instead of writing it out.

    Attributes:
        debug   : Whether debug-level logging is requested.
        scalars : Recorded (tag, value, step) triples in logging order.
    """
    def __init__(self) -> None:
        """Initializes an empty scalar record with debug logging off."""
        self.debug   = False
        self.scalars = []

    def log_scalar(self, tag, value, step=None) -> None:
        """Records a single scalar under its tag and step."""
        self.scalars.append((tag, float(value), step))

    def log_metrics(self, prefix, values, step=None) -> None:
        """Records every value of the mapping under the prefixed tag and step."""
        for key, value in values.items():
            self.scalars.append((f"{prefix}/{key}", float(value), step))
