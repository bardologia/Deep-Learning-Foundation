from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from configuration.param_extraction import ChannelOrder
from tools.data.regions             import CropRegion
from tools.monitoring.logger        import Logger


class ChannelMapper:

    def __init__(self, source_order: str, n_source_gaussians: int, k_slots: int) -> None:
        self.source_order = source_order
        self.positions    = ChannelOrder.positions(source_order)
        self.n_source     = int(n_source_gaussians)
        self.n_slots      = int(k_slots) if k_slots else int(n_source_gaussians)

        if self.n_slots < self.n_source:
            raise ValueError(f"k_slots={k_slots} is smaller than the {self.n_source} Gaussians the external sources carry; padding can add empty slots but never drops fitted ones")

    @property
    def n_target_channels(self) -> int:
        return 3 * self.n_slots

    @property
    def n_padded_slots(self) -> int:
        return self.n_slots - self.n_source

    def channel_pairs(self) -> list[Tuple[int, int]]:
        pairs = []

        for gaussian in range(self.n_source):
            for component, position in enumerate(self.positions):
                pairs.append((3 * gaussian + position, 3 * gaussian + component))

        return pairs


class TargetAssembler:

    def __init__(self, mapper: ChannelMapper, target_crop: CropRegion, logger: Logger) -> None:
        self.mapper      = mapper
        self.target_crop = target_crop
        self.logger      = logger

    def _intersection(self, window: CropRegion) -> Optional[CropRegion]:
        azimuth_start = max(window.azimuth_start, self.target_crop.azimuth_start)
        azimuth_end   = min(window.azimuth_end,   self.target_crop.azimuth_end)
        range_start   = max(window.range_start,   self.target_crop.range_start)
        range_end     = min(window.range_end,     self.target_crop.range_end)

        if azimuth_start >= azimuth_end or range_start >= range_end:
            return None

        return CropRegion(azimuth_start, azimuth_end, range_start, range_end)

    def _paint(self, source, overlap: CropRegion, parameters: np.ndarray, covered: np.ndarray) -> None:
        source_azimuth, source_range = overlap.local_slices(source.window)
        target_azimuth, target_range = overlap.local_slices(self.target_crop)

        already = covered[target_azimuth, target_range]
        if already.any():
            raise ValueError(f"External source {source.path.name} overlaps a region of {overlap.as_labeled_string()} already filled by another source on {int(already.sum())} pixels; two independent parameter files must not claim the same pixels")

        for source_channel, target_channel in self.mapper.channel_pairs():
            parameters[target_channel, target_azimuth, target_range] = source.cube.channel_window(source_channel, source_azimuth, source_range)

        covered[target_azimuth, target_range] = True

        self.logger.subsection(f"{source.path.name}: filled {overlap.as_labeled_string()} ({overlap.azimuth_size} x {overlap.range_size} pixels)")

    def _validate_coverage(self, covered: np.ndarray) -> None:
        if covered.all():
            return

        missing        = ~covered
        azimuth_gaps   = np.where(missing.any(axis=1))[0]
        range_gaps     = np.where(missing.any(axis=0))[0]
        azimuth_extent = f"{self.target_crop.azimuth_start + int(azimuth_gaps[0])}-{self.target_crop.azimuth_start + int(azimuth_gaps[-1]) + 1}"
        range_extent   = f"{self.target_crop.range_start + int(range_gaps[0])}-{self.target_crop.range_start + int(range_gaps[-1]) + 1}"

        raise ValueError(f"External sources leave {int(missing.sum())} of {missing.size} pixels of the dataset crop {self.target_crop.as_labeled_string()} unfilled, spanning azimuth {azimuth_extent} and range {range_extent}; every pixel of the crop needs external parameters or training would read zeros as ground truth")

    def run(self, sources: list) -> np.ndarray:
        self.logger.section(f"[Assembling] Target crop {self.target_crop.as_labeled_string()} into {self.mapper.n_target_channels} channels")

        parameters = np.zeros((self.mapper.n_target_channels, self.target_crop.azimuth_size, self.target_crop.range_size), dtype=np.float32)
        covered    = np.zeros((self.target_crop.azimuth_size, self.target_crop.range_size), dtype=bool)

        for source in sources:
            overlap = self._intersection(source.window)

            if overlap is None:
                self.logger.subsection(f"{source.path.name}: window {source.window.as_labeled_string()} lies outside the dataset crop, skipped")
                continue

            self._paint(source, overlap, parameters, covered)

        self._validate_coverage(covered)

        return parameters


class ParameterValidator:

    def __init__(self, n_slots: int, activity_threshold: float, height_range: Tuple[float, float], logger: Logger) -> None:
        self.n_slots            = int(n_slots)
        self.activity_threshold = float(activity_threshold)
        self.height_range       = (float(height_range[0]), float(height_range[1]))
        self.logger             = logger

    def _components(self, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        reshaped = parameters.reshape(self.n_slots, 3, parameters.shape[1], parameters.shape[2])

        return reshaped[:, 0], reshaped[:, 1], reshaped[:, 2]

    def _validate_finite(self, parameters: np.ndarray) -> None:
        non_finite = ~np.isfinite(parameters)

        if non_finite.any():
            raise ValueError(f"External parameters hold {int(non_finite.sum())} non-finite values; the fit that produced them left gaps that cannot be used as training targets")

    def _validate_ranges(self, amps: np.ndarray, sigmas: np.ndarray) -> None:
        if amps.min() < 0.0:
            raise ValueError(f"External parameters hold negative amplitudes (minimum {float(amps.min()):.6g}); amplitudes are non-negative in the Gaussian mixture the models predict")

        if sigmas.min() < 0.0:
            raise ValueError(f"External parameters hold negative widths (minimum {float(sigmas.min()):.6g}); sigmas are non-negative in the Gaussian mixture the models predict")

    def _validate_heights(self, amps: np.ndarray, mus: np.ndarray) -> None:
        active = amps > self.activity_threshold

        if not active.any():
            raise ValueError(f"No external Gaussian has an amplitude above the activity threshold {self.activity_threshold:g}; every pixel would train against an empty profile")

        active_mus = mus[active]
        low, high  = self.height_range
        outside    = (active_mus < low) | (active_mus > high)

        if outside.any():
            raise ValueError(f"{int(outside.sum())} active Gaussian means fall outside the dataset height axis [{low:g}, {high:g}] m, spanning [{float(active_mus.min()):.4g}, {float(active_mus.max()):.4g}] m; the external heights do not follow the same convention as this dataset's tomogram")

    def _report(self, amps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> None:
        rows = []

        for slot in range(self.n_slots):
            active = amps[slot] > self.activity_threshold
            rows.append({
                "Slot"      : slot,
                "Active"    : f"{100.0 * float(active.mean()):.2f} %",
                "Amplitude" : f"[{float(amps[slot].min()):.4g}, {float(amps[slot].max()):.4g}]",
                "Mean (m)"  : f"[{float(mus[slot].min()):.4g}, {float(mus[slot].max()):.4g}]",
                "Sigma (m)" : f"[{float(sigmas[slot].min()):.4g}, {float(sigmas[slot].max()):.4g}]",
            })

        self.logger.section("[External Parameter Statistics]")
        self.logger.metrics_table(rows, ["Slot", "Active", "Amplitude", "Mean (m)", "Sigma (m)"])

    def run(self, parameters: np.ndarray) -> None:
        amps, mus, sigmas = self._components(parameters)

        self._validate_finite(parameters)
        self._validate_ranges(amps, sigmas)
        self._validate_heights(amps, mus)
        self._report(amps, mus, sigmas)
