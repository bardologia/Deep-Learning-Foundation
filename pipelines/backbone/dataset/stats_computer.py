from __future__ import annotations

import numpy as np

from configuration.normalization      import ChannelStats, NormalizationConfig, NormMethod
from pipelines.backbone.dataset.stats import Stats
from tools.monitoring.logger          import Logger


class StatsComputer:

    SAMPLE_BUDGET = 1_000_000

    @classmethod
    def _sample(cls, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if values.size <= cls.SAMPLE_BUDGET:
            return values

        return values[rng.choice(values.size, size=cls.SAMPLE_BUDGET, replace=False)]

    @classmethod
    def _fit(cls, label: str, arrays: list[np.ndarray], names: list[str], method: NormMethod, rng: np.random.Generator, logger: Logger) -> ChannelStats:
        locs, scales, rows = [], [], []

        for index, name in enumerate(names):
            values     = cls._sample(np.concatenate([array[index].ravel() for array in arrays]), rng)
            loc, scale = method.fit(values)

            locs.append(loc)
            scales.append(scale)
            rows.append({"Channel": name, "Samples": f"{values.size:,}", "loc": f"{loc:+.6f}", "scale": f"{scale:.6f}"})

        logger.metrics_table(rows, ["Channel", "Samples", "loc", "scale"], title=f"{label} normalization ({method.value}, fitted on the train split)")

        return ChannelStats(names=list(names), method=method, loc=locs, scale=scales)

    @classmethod
    def compute(cls, crops: list[tuple[np.ndarray, np.ndarray]], input_channels: list[str], target_channels: list[str], normalization: NormalizationConfig, seed: int, logger: Logger) -> Stats:
        rng = np.random.default_rng(seed)

        logger.section("[Normalization Statistics]")

        input_stats  = cls._fit("Input",  [inputs for inputs, _targets in crops],  input_channels,  normalization.inputs,  rng, logger)
        target_stats = cls._fit("Target", [targets for _inputs, targets in crops], target_channels, normalization.targets, rng, logger)

        return Stats(input_stats=input_stats, target_stats=target_stats)
