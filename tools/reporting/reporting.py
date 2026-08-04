from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing  import List

from tools.runtime.run_tag import RunTag


class ReportAssets:

    MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}

    def __init__(self, base: Path, embed_images: bool = False, timestamp: str | None = None) -> None:
        self.base         = Path(base)
        self.embed_images = embed_images
        self.timestamp    = timestamp if timestamp is not None else RunTag.timestamp()

    def rel(self, target: Path) -> str:
        return Path(os.path.relpath(Path(target).resolve(), self.base.resolve())).as_posix()

    def src(self, path: Path) -> str:
        path = Path(path)
        if self.embed_images and path.exists():
            mime = self.MIME.get(path.suffix.lower(), "image/png")
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"
        return self.rel(path)

    def image(self, label: str, path: Path) -> List[str]:
        return [f"![{label}]({self.src(path)})", ""]

    def images(self, label: str, paths) -> List[str]:
        if isinstance(paths, (str, Path)):
            return self.image(label, Path(paths))

        out: List[str] = []
        for path in paths:
            out += self.image(Path(path).stem, Path(path))
        return out

    @staticmethod
    def natural_key(name: str) -> list:
        return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)]

    def header(self, title: str) -> List[str]:
        return [f"# {title}", f"\n_Generated {self.timestamp}_\n"]


class MetricSectionGrouper:

    PER_BIN_PATTERN = re.compile(r"_\d+$")

    METRIC_SECTIONS = [
        ("Dataset Statistics",           re.compile(r"^(n_pixels|n_elevation|x_axis_|gt_|pred_)")),
        ("Curve-Level",                  re.compile(r"^(curve_|overall_r2|psnr_)")),
        ("SSIM",                         re.compile(r"^ssim_")),
        ("Per-Pixel MSE and MAE",        re.compile(r"^pixel_(mse|mae)_")),
        ("Per-Pixel R² and Cosine",      re.compile(r"^pixel_(r2|cosine)_")),
        ("Peak Location Error",          re.compile(r"^pixel_peak_")),
        ("Per-Elevation-Bin Aggregates", re.compile(r"^elev_")),
        ("Slot Occupancy",               re.compile(r"^slot_")),
        ("Matched Gaussian (Permutation-Invariant)", re.compile(r"^matched_")),
        ("Label Quality",                re.compile(r"^label_")),
        ("Normalization Health",         re.compile(r"^(norm_in_|clamp_)")),
        ("Flip Consistency",             re.compile(r"^flip_")),
        ("Stratified Errors",            re.compile(r"^strat")),
        ("Failure Modes",                re.compile(r"^failure_")),
        ("Presence Calibration",         re.compile(r"^presence_")),
    ]

    LEFTOVER_TITLE = "Other Metrics"

    @classmethod
    def scalar_keys(cls, records) -> list[str]:
        return sorted({
            key
            for record in records
            for key, value in record.metrics.items()
            if isinstance(value, (int, float)) and not cls.PER_BIN_PATTERN.search(key)
        })

    def group(self, keys: list[str]) -> list[tuple[str, list[str]]]:
        all_keys = list(keys)
        claimed  : set[str]                    = set()
        sections : list[tuple[str, list[str]]] = []

        for title, pattern in self.METRIC_SECTIONS:
            matched = [key for key in all_keys if key not in claimed and pattern.search(key)]
            if not matched:
                continue
            claimed.update(matched)
            sections.append((title, matched))

        leftover = [key for key in all_keys if key not in claimed]
        if leftover:
            sections.append((self.LEFTOVER_TITLE, leftover))

        return sections
