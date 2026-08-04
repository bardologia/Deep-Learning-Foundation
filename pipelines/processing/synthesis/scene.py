from __future__ import annotations

import numpy as np

from configuration.synthesis               import SynthesisConfig
from pipelines.processing.synthesis.fields import SmoothFieldSampler, WhiteNoiseSampler


class ZAxis:
    @staticmethod
    def build(config: SynthesisConfig) -> np.ndarray:
        return np.linspace(config.z_range[0], config.z_range[1], config.profile_length, dtype=np.float32)


class ProfileBuilder:
    def __init__(self, z_axis: np.ndarray) -> None:
        self.z_axis = z_axis

    def evaluate(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        profiles = np.zeros((self.z_axis.shape[0],) + amps.shape[1:], dtype=np.float32)
        axis     = self.z_axis[:, None, None]

        for slot in range(amps.shape[0]):
            self.accumulate(profiles, amps[slot], mus[slot], sigmas[slot], axis)

        return profiles

    @staticmethod
    def accumulate(target: np.ndarray, amp: np.ndarray, mu: np.ndarray, sigma: np.ndarray, axis: np.ndarray) -> None:
        safe     = np.maximum(sigma, 1e-6)[None, :, :]
        exponent = -((axis - mu[None, :, :]) ** 2) / (2.0 * safe ** 2)

        target += amp[None, :, :] * np.exp(np.clip(exponent, -100.0, 0.0), dtype=np.float32)


class CubeBuilder:
    def __init__(self, config: SynthesisConfig, noise: WhiteNoiseSampler) -> None:
        self.config = config
        self.noise  = noise

    def build(self, profiles: np.ndarray) -> np.ndarray:
        peak  = profiles.max(axis=0, keepdims=True)
        phase = self.noise.uniform_phase(100, profiles.shape)

        cube  = (profiles * np.exp(1j * phase)).astype(np.complex64)
        cube += self.noise.complex(101, profiles.shape, self.config.cube_noise_fraction * peak)

        return cube.astype(np.complex64)


class ChannelBuilder:
    def __init__(self, config: SynthesisConfig, z_axis: np.ndarray, noise: WhiteNoiseSampler) -> None:
        self.config  = config
        self.z_axis  = z_axis
        self.noise   = noise
        self.sampler = SmoothFieldSampler(
            shape            = (config.row_size, config.col_size),
            smoothing_pixels = config.phase_smoothing_pixels,
            seed             = config.seed + 977,
        )

    def sensitivity_window(self, index: int) -> np.ndarray:
        low, high = self.config.z_range
        span      = high - low
        centre    = low + (index + 0.5) * span / self.config.n_secondaries
        width     = self.config.sensitivity_width_fraction * span

        return np.exp(-((self.z_axis - centre) ** 2) / (2.0 * width ** 2)).astype(np.float32)

    def _weighted_mean(self, profiles: np.ndarray, window: np.ndarray) -> np.ndarray:
        return (window[:, None, None] * profiles).sum(axis=0) / float(window.sum())

    def _coherence(self, profiles: np.ndarray, phase_rate: float) -> np.ndarray:
        total  = profiles.sum(axis=0)
        kernel = np.exp(1j * phase_rate * self.z_axis).astype(np.complex64)
        summed = (kernel[:, None, None] * profiles).sum(axis=0)

        return (summed / np.maximum(total, 1e-8)).astype(np.complex64)

    def build(self, profiles: np.ndarray) -> tuple:
        config    = self.config
        flat_mean = profiles.mean(axis=0)
        reference = self.sampler.angle(0)

        primary  = (flat_mean * np.exp(1j * reference)).astype(np.complex64)
        primary += self.noise.complex(200, primary.shape, config.channel_noise_fraction * flat_mean)

        secondaries = np.empty((config.n_secondaries,) + primary.shape, dtype=np.complex64)

        for index in range(config.n_secondaries):
            response  = self._weighted_mean(profiles, self.sensitivity_window(index))
            magnitude = np.minimum(response, config.max_amplitude_clip * flat_mean)
            coherence = self._coherence(profiles, float(config.channel_phase_rates[index]))

            channel_field  = (magnitude * np.exp(1j * (reference - np.angle(coherence)))).astype(np.complex64)
            channel_field += self.noise.complex(201 + index, primary.shape, config.channel_noise_fraction * flat_mean)

            secondaries[index] = channel_field

        derived = (primary[None, :, :] * np.conj(secondaries)).astype(np.complex64)

        return primary, secondaries, derived


class AuxFieldBuilder:
    def __init__(self, config: SynthesisConfig) -> None:
        self.config  = config
        self.sampler = SmoothFieldSampler(
            shape            = (config.row_size, config.col_size),
            smoothing_pixels = config.field_smoothing_pixels * 2.0,
            seed             = config.seed + 4231,
        )

    def build(self) -> np.ndarray:
        return self.sampler.bounded(0, *self.config.aux_range)
