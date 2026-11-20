from __future__ import annotations

from dataclasses import dataclass, field
from enum        import Enum


class LossKind(str, Enum):

    MSE   = "mse"
    L1    = "l1"
    HUBER = "huber"


@dataclass
class LossConfig:

    kind            : LossKind    = LossKind.MSE
    huber_delta     : float       = 1.0
    channel_weights : list[float] = field(default_factory=list)
