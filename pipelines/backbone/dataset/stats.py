from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

from configuration.normalization import ChannelStats, OutputClampConfig
from tools.data.io               import FileIO
from tools.monitoring.logger     import Logger


@dataclass
class Stats:
    input_stats  : Optional[ChannelStats] = None
    output_stats : Optional[ChannelStats] = None
    clamp        : OutputClampConfig      = field(default_factory=OutputClampConfig)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        out_path  = directory / "normalization_stats.json"

        payload = {
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
            "clamp"        : self.clamp.as_dict(),
        }

        return FileIO.save_json(payload, out_path, indent=4)

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path = Path(directory) / "normalization_stats.json"
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found at '{path}'.")

        payload = FileIO.load_json(path)

        if payload["input_stats"] is None or payload["output_stats"] is None:
            missing = [key for key in ("input_stats", "output_stats") if payload[key] is None]
            raise ValueError(f"Normalization stats at '{path}' lack {missing}; the file holds only a partial half and cannot drive training or inference.")

        input_stats  = ChannelStats.from_dict(payload["input_stats"])
        output_stats = ChannelStats.from_dict(payload["output_stats"])
        clamp        = OutputClampConfig.from_dict(payload["clamp"])

        logger.section("[Normalization stats loaded]")
        logger.kv_table({
            "Stats path"      : path,
            "Input channels"  : input_stats.n_channels,
            "Output channels" : output_stats.n_channels,
            "Output clamp"    : f"{clamp.floor} to {clamp.ceil}" if clamp.enabled else "disabled",
            "Clamp leak"      : clamp.leaky_slope,
            "Param leak"      : clamp.param_leaky_slope,
            "Amp max"         : clamp.amp_max,
        })

        return cls(input_stats = input_stats, output_stats = output_stats, clamp = clamp)

    @classmethod
    def merge(cls, input_only: "Stats", output_only: "Stats", clamp: OutputClampConfig) -> "Stats":
        return cls(
            input_stats  = input_only.input_stats,
            output_stats = output_only.output_stats,
            clamp        = clamp,
        )
