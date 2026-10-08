"""Gaussian-mixture z profiles: evaluation, reconstruction, slot activity and ordering.

The volumetric label of a pixel is a mixture of Gaussians over the z
axis, each component described by an amplitude, a mean height in metres and a
width in metres. This module evaluates those mixtures in NumPy, holds the one
definition of an active slot and orders the slots; the torch-side reconstruction
and clamp live in `tools.data.gaussians_torch`.
"""
from __future__ import annotations

from typing import List

import numpy as np


class GaussianAxis:
    """Builds the z axis a Gaussian mixture is sampled on."""
    @staticmethod
    def build(x_min: float, x_max: float, length: int) -> np.ndarray:
        """Returns an evenly spaced z axis.

        Args:
            x_min  : First z in metres.
            x_max  : Last z in metres.
            length : Number of z bins.

        Returns:
            Z axis in metres of shape (length,).
        """
        return np.linspace(x_min, x_max, length, dtype=np.float32)


class GaussianMixture:
    """Evaluation of Gaussian mixtures over an z axis, in NumPy.

    Attributes:
        SIGMA_FLOOR : Smallest width in metres used in the exponent denominator.
        EXPON_FLOOR : Lower clamp of the exponent, guarding against underflow.
        EXPON_CEIL  : Upper clamp of the exponent.
    """
    SIGMA_FLOOR = 1e-6
    EXPON_FLOOR = -100.0
    EXPON_CEIL  = 0.0

    @classmethod
    def safe_sigma_sq(cls, sigmas: np.ndarray) -> np.ndarray:
        """Returns 2*sigma^2 in square metres, with sigma floored away from zero."""
        clamped = np.maximum(sigmas, cls.SIGMA_FLOOR)
        return 2.0 * clamped ** 2

    @classmethod
    def kernel(cls, amp: np.ndarray, mu: np.ndarray, sig: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluates one Gaussian component on the given zs.

        Args:
            amp : Component amplitude, broadcastable against x.
            mu  : Component mean height in metres, broadcastable against x.
            sig : Component width in metres, broadcastable against x.
            x   : Z positions in metres to evaluate on.

        Returns:
            Component values at x, with the broadcast shape of the inputs.
        """
        expon = np.clip(-((x - mu) ** 2) / cls.safe_sigma_sq(sig), cls.EXPON_FLOOR, cls.EXPON_CEIL)
        return amp * np.exp(expon)

    @classmethod
    def evaluate_batch(cls, z_axis: np.ndarray, amps: np.ndarray, mus: np.ndarray, sigs: np.ndarray) -> np.ndarray:
        """Evaluates a batch of mixtures on a shared z axis.

        Args:
            z_axis : Z positions in metres of shape (n_bins,).
            amps        : Amplitudes of shape (batch, n_gaussians).
            mus         : Mean z_values in metres of shape (batch, n_gaussians).
            sigs        : Widths in metres of shape (batch, n_gaussians).

        Returns:
            Mixture profiles of shape (batch, n_bins).
        """
        pred = np.zeros((amps.shape[0], z_axis.shape[0]), dtype=np.float32)
        h    = z_axis[None, :]

        for g in range(amps.shape[1]):
            pred += cls.kernel(amps[:, g:g + 1], mus[:, g:g + 1], sigs[:, g:g + 1], h)

        return pred

    @classmethod
    def evaluate_slice(cls, parameters_array: np.ndarray, h_val: float, n_gaussians: int) -> np.ndarray:
        """Evaluates a parameter map at a single z.

        Args:
            parameters_array : Parameter cube of shape (3*n_gaussians, rows, cols), interleaved as amplitude, mean in metres, width in metres per slot.
            h_val            : Z position in metres to evaluate at.
            n_gaussians      : Number of mixture components to sum.

        Returns:
            Mixture value per pixel of shape (rows, cols).
        """
        reconstructed = np.zeros(parameters_array.shape[1:], dtype=np.float32)

        for k in range(n_gaussians):
            reconstructed += cls.kernel(parameters_array[3 * k], parameters_array[3 * k + 1], parameters_array[3 * k + 2], h_val)

        return reconstructed

    @classmethod
    def evaluate_pixel(cls, params: np.ndarray, z_axis: np.ndarray, n_gaussians: int) -> tuple:
        """Evaluates the mixture of one pixel and keeps its components.

        Args:
            params      : Flat parameter vector of length 3*n_gaussians, interleaved as amplitude, mean in metres, width in metres per slot.
            z_axis : Z positions in metres of shape (n_bins,).
            n_gaussians : Number of mixture components to sum.

        Returns:
            Tuple of the summed profile of shape (n_bins,) and the list of the
            individual component profiles, each of shape (n_bins,).
        """
        components = []
        total      = np.zeros_like(z_axis, dtype=np.float64)

        for k in range(n_gaussians):
            comp = cls.kernel(float(params[3 * k]), float(params[3 * k + 1]), float(params[3 * k + 2]), z_axis)
            components.append(comp)
            total += comp

        return total, components


class GaussianReconstructor:
    """Reconstructs profiles from slot-major Gaussian parameter arrays."""
    @staticmethod
    def reconstruct_batch(gauss: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Reconstructs one profile per batch element from slot-major parameters.

        Args:
            gauss : Parameters of shape (batch, n_gaussians, 3), the last axis holding amplitude, mean in metres and width in metres.
            x     : Z positions in metres, broadcastable against (batch, n_gaussians, n_bins).

        Returns:
            Summed profiles of shape (batch, n_bins), as float32.
        """
        a   = gauss[:, :, 0:1]
        mu  = gauss[:, :, 1:2]
        sig = gauss[:, :, 2:3]

        out = GaussianMixture.kernel(a, mu, sig, x).sum(axis=1)

        return out.astype(np.float32)

    @staticmethod
    def components(params: np.ndarray, z_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        """Returns the individual component profiles of one pixel.

        Args:
            params      : Flat parameter vector of length 3*n_gaussians.
            z_axis      : Sample positions on the z axis of shape (n_bins,).
            n_gaussians : Number of components to evaluate.

        Returns:
            One profile of shape (n_bins,) per component.
        """
        return [GaussianMixture.kernel(float(params[3 * k]), float(params[3 * k + 1]), float(params[3 * k + 2]), z_axis) for k in range(n_gaussians)]


class GaussianActivity:
    """The one definition of when a Gaussian slot counts as active."""
    @staticmethod
    def is_active(amplitude, threshold: float):
        """Returns whether an amplitude lies strictly above the activity threshold.

        Every loss mask, slot sorter, matcher, metric and analysis asks this
        question here, so a threshold of zero means any positive amplitude and a
        slot stored at exactly zero is never active.

        Args:
            amplitude : Physical amplitude as a float, NumPy array or torch tensor of any shape.
            threshold : Physical amplitude above which a slot counts as active, as recorded by the label run.

        Returns:
            Boolean of the same kind and shape as amplitude.
        """
        return amplitude > threshold


class GaussianSlotSorter:
    """Orders the Gaussian slots of a parameter cube."""
    @staticmethod
    def by_mean(parameters_array: np.ndarray, n_gaussians: int, activity_threshold: float) -> np.ndarray:
        """Sorts the slots of every pixel by mean z, pushing inactive slots last.

        The sort is stable, so slots sharing a key, which inactive slots all do,
        keep their stored relative order.

        Args:
            parameters_array   : Parameter cube of shape (3*n_gaussians, rows, cols).
            n_gaussians        : Number of slots held in the cube.
            activity_threshold : Amplitude above which a slot counts as active.

        Returns:
            The cube with slots reordered, of the same shape as the input.
        """
        n_params, n_row, n_col = parameters_array.shape
        reshaped               = parameters_array.reshape(n_gaussians, 3, n_row, n_col)

        amps = reshaped[:, 0, :, :]
        mus  = reshaped[:, 1, :, :]

        sort_keys = np.where(GaussianActivity.is_active(amps, activity_threshold), mus, np.inf)
        order     = np.argsort(sort_keys, axis=0, kind="stable")
        ordered   = np.take_along_axis(reshaped, order[:, np.newaxis, :, :], axis=0)

        return ordered.reshape(n_params, n_row, n_col)


class GaussianHead:
    """Channel arithmetic of a Gaussian prediction head."""
    @staticmethod
    def total_channels(ppg: int, n_gaussians: int) -> int:
        """Returns the number of output channels a Gaussian head needs."""
        return ppg * n_gaussians
