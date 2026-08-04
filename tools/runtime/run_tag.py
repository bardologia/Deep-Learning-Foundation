from __future__ import annotations

from datetime import datetime


class RunTag:
    TAG_FORMAT       = "%Y%m%d_%H%M%S"
    TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def now(cls) -> str:
        return datetime.now().strftime(cls.TAG_FORMAT)

    @classmethod
    def timestamp(cls) -> str:
        return datetime.now().strftime(cls.TIMESTAMP_FORMAT)
