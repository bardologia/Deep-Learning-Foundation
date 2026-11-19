
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

from configuration.normalization import ChannelStats, OutputClampConfig
from tools.data.io               import FileIO
from tools.monitoring.logger     import Logger


@dataclass
class Stats:

    FILENAME = "normalization_stats.json"

    input_stats  : Optional[ChannelStats] = None
    output_stats : Optional[ChannelStats] = None
    clamp        : OutputClampConfig      = field(default_factory=OutputClampConfig)

    def payload(self) -> dict:
        return {
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
            "clamp"        : self.clamp.as_dict(),
        }

    def save(self, directory: Path) -> Path:
        out_path = Path(directory) / self.FILENAME

        return FileIO.save_json(self.payload(), out_path, indent=4)

    @staticmethod
    def clamp_rows(clamp: OutputClampConfig) -> dict:
        return {
            "Output clamp" : f"{clamp.floor} to {clamp.ceil}" if clamp.enabled else "disabled",
            "Clamp leak"   : clamp.leaky_slope,
            "Param leak"   : clamp.param_leaky_slope,
            "Amp max"      : clamp.amp_max,
        }

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path = Path(directory) / cls.FILENAME

        payload = FileIO.load_json(path)

        input_stats  = ChannelStats.from_dict(payload["input_stats"])
        output_stats = ChannelStats.from_dict(payload["output_stats"])
        clamp        = OutputClampConfig.from_dict(payload["clamp"])

        logger.section("[Normalization stats loaded]")
        logger.kv_table({
            "Stats path"      : path,
            "Input channels"  : input_stats.n_channels,
            "Output channels" : output_stats.n_channels,
            **cls.clamp_rows(clamp),
        })

        return cls(input_stats = input_stats, output_stats = output_stats, clamp = clamp)

    @classmethod
    def merge(cls, input_only: "Stats", output_only: "Stats", clamp: OutputClampConfig) -> "Stats":
        return cls(
            input_stats  = input_only.input_stats,
            output_stats = output_only.output_stats,
            clamp        = clamp,
        )
