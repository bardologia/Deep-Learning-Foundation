from __future__ import annotations

import optuna


class TuningStorage:

    HEARTBEAT_INTERVAL_S = 60
    GRACE_PERIOD_S       = 600

    @classmethod
    def open(cls, storage_url: str) -> optuna.storages.RDBStorage:
        return optuna.storages.RDBStorage(
            url                = storage_url,
            heartbeat_interval = cls.HEARTBEAT_INTERVAL_S,
            grace_period       = cls.GRACE_PERIOD_S,
        )
