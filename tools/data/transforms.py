from __future__ import annotations

import math

import numpy as np
import torch


class Log1pTransform:
    CEIL = 200.0

    @staticmethod
    def compress(x, leaky_slope: float = 0.0):
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
