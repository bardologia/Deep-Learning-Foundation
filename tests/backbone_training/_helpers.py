from __future__ import annotations

import numpy as np
import torch

from configuration.normalization           import ChannelStats, ChannelStrategy, NormMethod, OutputClampConfig
from configuration.sar.gaussian_config     import GaussianConfig
from configuration.sar.geometry_config     import GeometryConfig
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
    return torch.linspace(X_MIN, X_MAX, length, dtype=torch.float32)


def x_axis_numpy(length: int = X_LEN) -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, length, dtype=np.float32)


def identity_normalizer(n_channels: int) -> Normalizer:
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


def geometry_config(n_tracks: int = 3) -> GeometryConfig:
    baselines = tuple(float(10 * i) for i in range(n_tracks))
    return GeometryConfig(wavelength=0.23, slant_range=5000.0, baselines=baselines)


def gaussian_config(n_gaussians: int) -> GaussianConfig:
    return GaussianConfig(n_default_gaussians=n_gaussians, x_min=X_MIN, x_max=X_MAX)


def build_loss(n_gaussians: int = 2, loss_cfg: LossConfig | None = None, log_all_losses: bool = False, length: int = X_LEN, sampler=None, norm_stats: Normalizer | None = None) -> Loss:
    n_channels = n_gaussians * 3
    loss_cfg   = loss_cfg if loss_cfg is not None else LossConfig(use_param_l1=True, weight_param_l1=1.0)

    return Loss(
        x_axis         = x_axis_tensor(length),
        logger         = NullLogger(),
        tracker        = NullTracker(),
        gaussian_cfg   = gaussian_config(n_gaussians),
        loss_cfg       = loss_cfg,
        norm_stats     = norm_stats if norm_stats is not None else identity_normalizer(n_channels),
        geometry_cfg   = geometry_config(),
        log_all_losses = log_all_losses,
        sampler        = sampler,
    )


def param_tensor(batch: int, n_gaussians: int, height: int, width: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_gaussians * 3, height, width, generator=gen, dtype=torch.float32)


def valid_param_tensor(batch: int, n_gaussians: int, height: int, width: int, seed: int) -> torch.Tensor:
    gen   = torch.Generator().manual_seed(seed)
    parts = []

    for _ in range(n_gaussians):
        amp = 1.0 + torch.rand(batch, 1, height, width, generator=gen) * 5.0
        mu  = X_MIN + 10.0 + torch.rand(batch, 1, height, width, generator=gen) * (X_MAX - X_MIN - 20.0)
        sig = 4.0 + torch.rand(batch, 1, height, width, generator=gen) * 6.0
        parts.extend([amp, mu, sig])

    return torch.cat(parts, dim=1).float()


def tiny_model(in_channels: int = 2, n_gaussians: int = 2):
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
    cfg = BackboneTrainerConfig(gaussian=gaussian_config(n_gaussians))

    cfg.io.writer                     = None
    cfg.training.epochs               = epochs
    cfg.training.validation_frequency = 1
    cfg.resources.enabled             = False
    cfg.geometry                      = geometry_config()

    cfg.curriculum.complete.use_param_l1    = True
    cfg.curriculum.complete.weight_param_l1 = 1.0

    return cfg


class TrackerStub:
    def __init__(self) -> None:
        self.debug   = False
        self.scalars = []

    def log_scalar(self, tag, value, step=None) -> None:
        self.scalars.append((tag, float(value), step))

    def log_metrics(self, prefix, values, step=None) -> None:
        for key, value in values.items():
            self.scalars.append((f"{prefix}/{key}", float(value), step))
