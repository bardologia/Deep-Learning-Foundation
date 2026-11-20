from __future__ import annotations

import math
from dataclasses import dataclass
from typing      import Tuple

import numpy as np


@dataclass
class GridInfo:

    n_v                   : int
    n_h                   : int
    pad_top               : int
    pad_bot               : int
    pad_left              : int
    pad_right             : int
    patch_size            : Tuple[int, int]
    stride                : Tuple[int, int]
    spatial_size          : Tuple[int, int]
    use_symmetric_padding : bool = True

    @property
    def padding_vertical(self) -> int:
        return self.pad_top + self.pad_bot

    @property
    def padding_horizontal(self) -> int:
        return self.pad_left + self.pad_right

    @property
    def number_of_patches(self) -> int:
        return self.n_v * self.n_h

    @property
    def padded_size(self) -> Tuple[int, int]:
        H, W = self.spatial_size
        return (H + self.padding_vertical, W + self.padding_horizontal)

    def patch_origin(self, idx: int) -> Tuple[int, int]:
        row, column = divmod(idx, self.n_h)
        return row * self.stride[0], column * self.stride[1]

    @staticmethod
    def _axis_coverage(n_patches: int, patch: int, stride: int, padded: int, pad_before: int, size: int) -> np.ndarray:
        counts = np.zeros(padded, dtype=np.int64)

        for index in range(n_patches):
            counts[index * stride:index * stride + patch] += 1

        return counts[pad_before:pad_before + size]

    def coverage(self) -> np.ndarray:
        H, W               = self.spatial_size
        padded_h, padded_w = self.padded_size

        rows    = self._axis_coverage(self.n_v, self.patch_size[0], self.stride[0], padded_h, self.pad_top,  H)
        columns = self._axis_coverage(self.n_h, self.patch_size[1], self.stride[1], padded_w, self.pad_left, W)

        return np.outer(rows, columns)

    def as_dict(self) -> dict:
        return {
            "n_v"                   : self.n_v,
            "n_h"                   : self.n_h,
            "pad_top"               : self.pad_top,
            "pad_bot"               : self.pad_bot,
            "pad_left"              : self.pad_left,
            "pad_right"             : self.pad_right,
            "patch_size"            : list(self.patch_size),
            "stride"                : list(self.stride),
            "spatial_size"          : list(self.spatial_size),
            "use_symmetric_padding" : self.use_symmetric_padding,
            "number_of_patches"     : self.number_of_patches,
        }


class Patcher:

    def __init__(self, grid: GridInfo, patch_coords: list) -> None:
        self.grid          = grid
        self._patch_coords = patch_coords

    @classmethod
    def build(cls, spatial_size : Tuple[int, int], patch_size : Tuple[int, int], stride : Tuple[int, int], use_symmetric_padding : bool = True) -> "Patcher":
        ph, pw = patch_size
        sv, sh = stride
        H, W   = spatial_size

        n_v = 1 if H <= ph else math.ceil((H - ph) / sv) + 1
        n_h = 1 if W <= pw else math.ceil((W - pw) / sh) + 1

        pad_v = (ph + (n_v - 1) * sv) - H
        pad_h = (pw + (n_h - 1) * sh) - W

        pad_top, pad_bot    = pad_v // 2, pad_v - pad_v // 2
        pad_left, pad_right = pad_h // 2, pad_h - pad_h // 2

        grid = GridInfo(
            n_v                   = n_v,
            n_h                   = n_h,
            pad_top               = pad_top,
            pad_bot               = pad_bot,
            pad_left              = pad_left,
            pad_right             = pad_right,
            patch_size            = (ph, pw),
            stride                = (sv, sh),
            spatial_size          = (H, W),
            use_symmetric_padding = use_symmetric_padding,
        )

        mode         = "symmetric" if use_symmetric_padding else "constant"
        patch_coords = []

        for iv in range(n_v):
            for ih in range(n_h):
                v0 = iv * sv - pad_top
                h0 = ih * sh - pad_left
                v1 = v0 + ph
                h1 = h0 + pw

                pt = max(0, -v0)
                pb = max(0, v1 - H)
                pl = max(0, -h0)
                pr = max(0, h1 - W)

                v0c = max(0, v0)
                v1c = min(H, v1)
                h0c = max(0, h0)
                h1c = min(W, h1)

                pw_spec = None
                if pt or pb or pl or pr:
                    pw_spec = (pt, pb, pl, pr, mode)

                patch_coords.append((v0c, v1c, h0c, h1c, pw_spec))

        return cls(grid, patch_coords)

    def extract(self, array: np.ndarray, idx: int) -> np.ndarray:
        v0c, v1c, h0c, h1c, pw_spec = self._patch_coords[idx]
        sub                         = np.ascontiguousarray(array[..., v0c:v1c, h0c:h1c])

        if pw_spec is not None:
            pt, pb, pl, pr, mode = pw_spec
            pad_width = [(0, 0)] * (sub.ndim - 2) + [(pt, pb), (pl, pr)]
            sub       = np.pad(sub, pad_width, mode=mode)

        return sub

    def weight_map(self, count_once: bool) -> np.ndarray:
        grid = self.grid
        H, W = grid.spatial_size

        weights = np.zeros(grid.padded_size, dtype=np.float32)
        weights[grid.pad_top:grid.pad_top + H, grid.pad_left:grid.pad_left + W] = 1.0 / grid.coverage() if count_once else 1.0

        return weights

    def cut_weights(self, weight_map: np.ndarray, idx: int) -> np.ndarray:
        v0, h0           = self.grid.patch_origin(idx)
        patch_h, patch_w = self.grid.patch_size

        return np.ascontiguousarray(weight_map[None, v0:v0 + patch_h, h0:h0 + patch_w])
