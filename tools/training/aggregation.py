"""Running means of the loss components and monitors reported over an epoch, and the Custom Scalars page that lays the logged families out."""

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


class ScalarsPage:
    """The Custom Scalars page of a run: charts that draw several tags each, so a family of scalars is read on one plot.

    TensorBoard matches the patterns of a chart from the start of the tag, so a
    family is named by its prefix and its stage pinned at the end. The base
    trainer adds the loss and the optimiser charts every family logs; a trainer
    adds the categories of the diagnostics its loss fills.

    Attributes:
        STAGES     : Stage labels a metric family is logged under, each with the word its chart titles use.
        categories : Category title to a mapping of chart title to its patterns, in page order.
    """

    STAGES = (("train", "training"), ("val", "validation"))

    def __init__(self) -> None:
        """Starts with an empty page."""
        self.categories = {}

    def add(self, category: str, title: str, patterns: list) -> None:
        """Adds one chart drawing every tag one of the patterns matches.

        Args:
            category : Category the chart sits in, created on first use.
            title    : Chart title, unique within the category.
            patterns : Tag regexes matched from the start of the tag.
        """
        self.categories.setdefault(category, {})[title] = list(patterns)

    def add_by_stage(self, category: str, label: str, family: str) -> None:
        """Adds one chart per stage drawing every key of a staged family, one line per key.

        Args:
            category : Category the charts sit in.
            label    : Chart label the stage word is appended to.
            family   : Tag prefix of the family, whose tags read family/key/stage.
        """
        for stage, word in self.STAGES:
            self.add(category, f"{label}, {word}", [f"{family}/.+/{stage}$"])

    def loss(self) -> "ScalarsPage":
        """Adds the total loss by stage and the loss terms and monitors by stage."""
        self.add("Loss", "Total loss by stage", ["loss/train$", "loss/val$", "loss/test$"])
        self.add_by_stage("Loss", "Loss terms", "loss_terms")
        self.add_by_stage("Loss", "Monitors", "loss_monitor")
        return self

    def optimiser(self) -> "ScalarsPage":
        """Adds the learning rate per parameter group and the gradient norm, total and per group."""
        self.add("Optimiser", "Learning rate per group", ["optim/lr/"])
        self.add("Optimiser", "Gradient norm, total and per group", ["optim/grad_norm$", "optim/grad_norm/"])
        return self

    def parameter_errors(self) -> "ScalarsPage":
        """Adds the physical parameter errors by stage, one line per parameter."""
        self.add_by_stage("Diagnostics", "Parameter errors", "param_error")
        return self

    def occupancy(self) -> "ScalarsPage":
        """Adds the predicted occupancy statistics by stage, one line per statistic."""
        self.add_by_stage("Diagnostics", "Occupancy", "occupancy")
        return self

    def slots(self) -> "ScalarsPage":
        """Adds the per slot vitals, one line per slot: active fraction, mean active amplitude and gradient norm."""
        self.add("Slots", "Active fraction per slot", ["slot_vitals/active_frac/"])
        self.add("Slots", "Mean active amplitude per slot", ["slot_vitals/amp_mean/"])
        self.add("Slots", "Gradient norm per slot", ["slot_vitals/grad_norm/"])
        return self

    def layout(self) -> dict:
        """Returns the page in the form the writer's add_custom_scalars takes: category to chart title to ["Multiline", patterns]."""
        return {category: {title: ["Multiline", list(patterns)] for title, patterns in charts.items()} for category, charts in self.categories.items()}
