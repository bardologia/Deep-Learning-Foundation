"""Normalization strategy configuration for input and output channels.

Re-exports the normalization method enum, the per-channel strategies and their
presets, the fitted channel statistics container and the output clamp config.
"""

from configuration.normalization.general import (
    NormMethod,
    ChannelStrategy,
    Presets,
    ChannelStats,
    OutputClampConfig,
    NormalizationConfig,
)
