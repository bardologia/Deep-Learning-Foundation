
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


class ScalarsPage:

    STAGES = (("train", "training"), ("val", "validation"))

    def __init__(self) -> None:
        self.categories = {}

    def add(self, category: str, title: str, patterns: list) -> None:
        self.categories.setdefault(category, {})[title] = list(patterns)

    def add_by_stage(self, category: str, label: str, family: str) -> None:
        for stage, word in self.STAGES:
            self.add(category, f"{label}, {word}", [f"{family}/.+/{stage}$"])

    def loss(self) -> "ScalarsPage":
        self.add("Loss", "Total loss by stage", ["loss/train$", "loss/val$", "loss/test$"])
        self.add_by_stage("Loss", "Loss terms", "loss_terms")
        self.add_by_stage("Loss", "Monitors", "loss_monitor")
        return self

    def optimiser(self) -> "ScalarsPage":
        self.add("Optimiser", "Learning rate per group", ["optim/lr/"])
        self.add("Optimiser", "Gradient norm, total and per group", ["optim/grad_norm$", "optim/grad_norm/"])
        return self

    def parameter_errors(self) -> "ScalarsPage":
        self.add_by_stage("Diagnostics", "Parameter errors", "param_error")
        return self

    def occupancy(self) -> "ScalarsPage":
        self.add_by_stage("Diagnostics", "Occupancy", "occupancy")
        return self

    def slots(self) -> "ScalarsPage":
        self.add("Slots", "Active fraction per slot", ["slot_vitals/active_frac/"])
        self.add("Slots", "Mean active amplitude per slot", ["slot_vitals/amp_mean/"])
        self.add("Slots", "Mean width per slot, all and declared pixels (m)", ["slot_vitals/sigma_mean/", "slot_vitals/sigma_mean_declared/"])
        self.add("Slots", "Mean centre per slot, all and declared pixels (m)", ["slot_vitals/mu_mean/", "slot_vitals/mu_mean_declared/"])
        self.add("Slots", "Gradient norm per slot", ["slot_vitals/grad_norm/"])
        return self

    def layout(self) -> dict:
        return {category: {title: ["Multiline", list(patterns)] for title, patterns in charts.items()} for category, charts in self.categories.items()}
