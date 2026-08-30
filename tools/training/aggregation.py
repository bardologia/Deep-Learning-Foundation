"""Running means of the loss components and monitors reported over an epoch."""

import torch


class MetricAggregator:
    """Accumulates per-batch loss entries and reduces them to epoch means.

    Four independent groups are tracked: weighted loss components, monitor-only
    values, slot occupancy statistics and physical-domain diagnostics.

    Attributes:
        components_sum : Running sum per loss-component key.
        monitor_sum    : Running sum per monitor key.
        occupancy_sum  : Running sum per occupancy key.
        physical_sum   : Running sum per physical-diagnostic key.
        components_n   : Number of batches contributing to each component key.
        monitor_n      : Number of batches contributing to each monitor key.
        occupancy_n    : Number of batches contributing to each occupancy key.
        physical_n     : Number of batches contributing to each physical key.
    """

    def __init__(self):
        """Initialises empty sums and counts for the four metric groups."""
        self.components_sum : dict = {}
        self.monitor_sum    : dict = {}
        self.occupancy_sum  : dict = {}
        self.physical_sum   : dict = {}

        self.components_n : dict = {}
        self.monitor_n    : dict = {}
        self.occupancy_n  : dict = {}
        self.physical_n   : dict = {}

    @staticmethod
    def _accumulate(store: dict, counts: dict, values: dict) -> None:
        """Adds one batch of values into a sum store, detaching tensors first.

        Args:
            store  : Running sum per key, updated in place.
            counts : Contribution count per key, updated in place.
            values : Scalar or 0-d tensor values of this batch, keyed by metric name.
        """
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            store[k]  = store.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1

    def add(self, loss_dict: dict) -> None:
        """Accumulates one batch of loss output.

        Args:
            loss_dict : Loss output holding the "components", "monitor", "occupancy" and "physical" sub-dictionaries.
        """
        self._accumulate(self.components_sum, self.components_n, loss_dict["components"])
        self._accumulate(self.monitor_sum,    self.monitor_n,    loss_dict["monitor"])
        self._accumulate(self.occupancy_sum,  self.occupancy_n,  loss_dict["occupancy"])
        self._accumulate(self.physical_sum,   self.physical_n,   loss_dict["physical"])

    @staticmethod
    def _reduce(store: dict, counts: dict) -> dict:
        """Returns the per-key mean of a sum store as plain floats."""
        return {k: float(v) / counts[k] for k, v in store.items()}

    def reduce_components(self) -> dict:
        """Returns the epoch mean of every weighted loss component."""
        return self._reduce(self.components_sum, self.components_n)

    def reduce_monitor(self) -> dict:
        """Returns the epoch mean of every monitor-only metric."""
        return self._reduce(self.monitor_sum, self.monitor_n)

    def reduce_occupancy(self) -> dict:
        """Returns the epoch mean of every slot-occupancy statistic."""
        return self._reduce(self.occupancy_sum, self.occupancy_n)

    def reduce_physical(self) -> dict:
        """Returns the epoch mean of every physical-domain diagnostic."""
        return self._reduce(self.physical_sum, self.physical_n)
