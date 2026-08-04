from __future__ import annotations

import numpy as np

from configuration.synthesis                    import SynthesisConfig
from pipelines.processing.synthesis.fields      import SmoothFieldSampler


class ParameterFieldBuilder:
    STREAMS_PER_SLOT = 4

    def __init__(self, config: SynthesisConfig) -> None:
        self.config  = config
        self.sampler = SmoothFieldSampler(
            shape            = (config.row_size, config.col_size),
            smoothing_pixels = config.field_smoothing_pixels,
            seed             = config.seed,
        )

    def _slot_band(self, slot: int) -> tuple:
        low, high = self.config.z_range
        width     = (high - low) / self.config.k_max
        margin    = self.config.slot_band_margin * width

        return (low + slot * width + margin, low + (slot + 1) * width - margin)

    def build(self) -> tuple:
        config = self.config
        amps   = np.zeros((config.k_max, config.row_size, config.col_size), dtype=np.float32)
        mus    = np.zeros_like(amps)
        sigmas = np.zeros_like(amps)

        empty = self.sampler.below_fraction(0, config.empty_pixel_fraction)

        for slot in range(config.k_max):
            base      = 1 + slot * self.STREAMS_PER_SLOT
            band      = self._slot_band(slot)
            amplitude = self.sampler.bounded(base + 0, *config.amplitude_range)
            mean      = self.sampler.bounded(base + 1, *band)
            sigma     = self.sampler.bounded(base + 2, *config.sigma_range)

            dropped = self.sampler.below_fraction(base + 3, config.dropped_slot_fraction) if slot > 0 else np.zeros_like(empty)
            silent  = empty | dropped

            amps  [slot] = np.where(silent, 0.0, amplitude)
            mus   [slot] = np.where(silent, 0.0, mean)
            sigmas[slot] = np.where(silent, 0.0, sigma)

        return amps, mus, sigmas
