from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.normalization import ChannelStats
from tools.data.io               import FileIO
from tools.monitoring.logger     import Logger


@dataclass
class Stats:

    FILENAME = "normalization_stats.json"

    input_stats  : ChannelStats
    target_stats : ChannelStats

    def payload(self) -> dict:
        return {"input_stats": self.input_stats.as_dict(), "target_stats": self.target_stats.as_dict()}

    def save(self, directory: Path) -> Path:
        return FileIO.save_json(self.payload(), Path(directory) / self.FILENAME, indent=4)

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path    = Path(directory) / cls.FILENAME
        payload = FileIO.load_json(path)
        stats   = cls(input_stats=ChannelStats.from_dict(payload["input_stats"]), target_stats=ChannelStats.from_dict(payload["target_stats"]))

        logger.section("[Normalization stats loaded]")
        logger.kv_table({
            "Stats path"      : path,
            "Input channels"  : f"{stats.input_stats.n_channels} ({stats.input_stats.method.value})",
            "Target channels" : f"{stats.target_stats.n_channels} ({stats.target_stats.method.value})",
        })

        return stats
