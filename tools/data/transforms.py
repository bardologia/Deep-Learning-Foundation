"""Log1p amplitude compression and its bounded inverse.

Reflectivity spans orders of magnitude, so it is trained in log1p space; the
inverse clamps back into the physical amplitude range, optionally leaking a
gradient outside the bounds.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class Log1pTransform:
    """Log1p compression of non-negative amplitudes and its bounded inverse.

    Attributes:
        CEIL : Default largest amplitude the inverse is clamped to.
    """
    CEIL = 200.0

    @staticmethod
    def compress(x, leaky_slope: float = 0.0):
        """Compresses non-negative amplitudes with log1p.

        Args:
            x           : Amplitudes as a torch tensor or NumPy array, of any shape.
            leaky_slope : Slope applied to the negative part instead of hard clipping it.

        Returns:
            Compressed values of the same type and shape as x.
        """
        if isinstance(x, torch.Tensor):
            out = torch.log1p(torch.clamp(x, min=0.0))
            if leaky_slope > 0.0:
                out = out + leaky_slope * torch.clamp(x, max=0.0)
            return out

        out = np.log1p(np.maximum(x, 0.0))
        if leaky_slope > 0.0:
            out = out + leaky_slope * np.minimum(x, 0.0)
        return out

    @staticmethod
    def decompress(x, floor: float = 0.0, ceil: float = CEIL, enabled: bool = True, leaky_slope: float = 0.0):
        """Inverts the log1p compression, bounded to a physical amplitude range.

        Args:
            x           : Compressed values as a torch tensor or NumPy array, of any shape.
            floor       : Smallest amplitude the result is bounded to.
            ceil        : Largest amplitude the result is bounded to.
            enabled     : Whether the bounds are applied at all.
            leaky_slope : Slope kept outside the bounds instead of a hard clamp.

        Returns:
            Amplitudes of the same type and shape as x.
        """
        lo = math.log1p(floor)
        hi = math.log1p(ceil)

        if isinstance(x, torch.Tensor):
            bounded = torch.clamp(x, min=lo, max=hi) if enabled else x
            if enabled and leaky_slope > 0.0:
                bounded = bounded + leaky_slope * (torch.clamp(x - lo, max=0.0) + torch.clamp(x - hi, min=0.0))
            return torch.expm1(bounded)

        bounded = np.clip(x, lo, hi) if enabled else x
        if enabled and leaky_slope > 0.0:
            bounded = bounded + leaky_slope * (np.minimum(x - lo, 0.0) + np.maximum(x - hi, 0.0))
        return np.expm1(bounded)
