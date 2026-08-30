"""Shared Optuna storage settings for the tuning workers.

Workers of the same study run as separate processes, so the storage is opened
with a heartbeat: a worker that dies leaves its trial detectable as stale once
the grace period elapses.
"""

from __future__ import annotations

import optuna


class TuningStorage:
    """Opens the study database with the heartbeat settings the workers share.

    Attributes:
        HEARTBEAT_INTERVAL_S : Seconds between heartbeats of a running trial.
        GRACE_PERIOD_S       : Seconds without a heartbeat after which a running trial counts as stale.
    """

    HEARTBEAT_INTERVAL_S = 60
    GRACE_PERIOD_S       = 600

    @classmethod
    def open(cls, storage_url: str) -> optuna.storages.RDBStorage:
        """Returns the RDB storage at the URL, configured with the heartbeat."""
        return optuna.storages.RDBStorage(
            url                = storage_url,
            heartbeat_interval = cls.HEARTBEAT_INTERVAL_S,
            grace_period       = cls.GRACE_PERIOD_S,
        )
