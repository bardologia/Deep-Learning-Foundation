"""Container and persistence of the normalization statistics of a training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

from configuration.normalization import ChannelStats, OutputClampConfig
from tools.data.io               import FileIO
from tools.monitoring.logger     import Logger


@dataclass
class Stats:
    """Fitted input and output channel statistics plus the output clamp of a run.

    Attributes:
        input_stats  : Per-channel statistics of the input stack, or None while only half of the statistics have been fitted.
        output_stats : Per-channel statistics of the Gaussian parameter targets, or None while only half of the statistics have been fitted.
        clamp        : Output clamp bounds and leak slopes applied on denormalization.
    """

    FILENAME = "normalization_stats.json"

    input_stats  : Optional[ChannelStats] = None
    output_stats : Optional[ChannelStats] = None
    clamp        : OutputClampConfig      = field(default_factory=OutputClampConfig)

    def payload(self) -> dict:
        """Returns the statistics and clamp as a JSON-ready dictionary."""
        return {
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
            "clamp"        : self.clamp.as_dict(),
        }

    def save(self, directory: Path) -> Path:
        """Writes the statistics into a directory.

        Args:
            directory : Directory receiving the file, usually the run's meta directory.

        Returns:
            Path of the written normalization_stats.json.
        """
        out_path = Path(directory) / self.FILENAME

        return FileIO.save_json(self.payload(), out_path, indent=4)

    @staticmethod
    def clamp_rows(clamp: OutputClampConfig) -> dict:
        """Returns the key-value rows describing an effective output clamp.

        Args:
            clamp : Output clamp configuration of a run.

        Returns:
            Rows naming the clamped range or "disabled", the curve and parameter
            leak slopes and the amplitude ceiling.
        """
        return {
            "Output clamp" : f"{clamp.floor} to {clamp.ceil}" if clamp.enabled else "disabled",
            "Clamp leak"   : clamp.leaky_slope,
            "Param leak"   : clamp.param_leaky_slope,
            "Amp max"      : clamp.amp_max,
        }

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        """Reads the statistics of a run back and reports them.

        Args:
            directory : Directory holding normalization_stats.json.
            logger    : Logger receiving the summary table.

        Returns:
            The restored statistics.

        Raises:
            FileNotFoundError : If the file is missing.
            ValueError        : If either the input or the output half of the statistics is null.
        """
        path = Path(directory) / cls.FILENAME
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
            **cls.clamp_rows(clamp),
        })

        return cls(input_stats = input_stats, output_stats = output_stats, clamp = clamp)

    @classmethod
    def merge(cls, input_only: "Stats", output_only: "Stats", clamp: OutputClampConfig) -> "Stats":
        """Combines separately fitted input and output statistics under one clamp.

        Args:
            input_only  : Statistics carrying only the input half.
            output_only : Statistics carrying only the output half.
            clamp       : Output clamp configuration of the run.

        Returns:
            The merged statistics.
        """
        return cls(
            input_stats  = input_only.input_stats,
            output_stats = output_only.output_stats,
            clamp        = clamp,
        )
