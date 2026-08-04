import torch


class MetricAggregator:
    def __init__(self):
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
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            store[k]  = store.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1

    def add(self, loss_dict: dict) -> None:
        self._accumulate(self.components_sum, self.components_n, loss_dict["components"])
        self._accumulate(self.monitor_sum,    self.monitor_n,    loss_dict["monitor"])
        self._accumulate(self.occupancy_sum,  self.occupancy_n,  loss_dict["occupancy"])
        self._accumulate(self.physical_sum,   self.physical_n,   loss_dict["physical"])

    @staticmethod
    def _reduce(store: dict, counts: dict) -> dict:
        return {k: float(v) / counts[k] for k, v in store.items()}

    def reduce_components(self) -> dict:
        return self._reduce(self.components_sum, self.components_n)

    def reduce_monitor(self) -> dict:
        return self._reduce(self.monitor_sum, self.monitor_n)

    def reduce_occupancy(self) -> dict:
        return self._reduce(self.occupancy_sum, self.occupancy_n)

    def reduce_physical(self) -> dict:
        return self._reduce(self.physical_sum, self.physical_n)
