"""Streaming collection of per-layer activation statistics via forward hooks.

Provides an accumulator that folds every forward pass of a leaf module into
scalar summaries and a log-spaced magnitude histogram, plus a recorder that
attaches those accumulators to a model without holding activations in memory.
"""

from __future__ import annotations

import numpy as np
import torch


class LayerActivationStats:
    """Running summary of the activations produced by one leaf module.

    Statistics are folded batch by batch, so memory use is independent of the
    number of batches captured. Channel-wise statistics are only tracked for
    four-dimensional (N, C, H, W) outputs.

    Attributes:
        name            : Dotted module path of the recorded layer.
        module_type     : Class name of the recorded module.
        shape           : Per-sample output shape (all dimensions except the batch axis).
        first_seen      : Forward-call index at which the module first fired.
        n_batches       : Number of forward passes folded in.
        n_elements      : Total number of activation values seen.
        n_zero          : Number of values exactly equal to zero.
        n_nonfinite     : Number of NaN or Inf values.
        value_sum       : Sum of finite activation values.
        value_sumsq     : Sum of squared finite activation values.
        max_abs         : Largest finite absolute activation seen.
        channel_max_abs : Per-channel maximum absolute activation, or None.
        channel_abs_sum : Per-channel sum of absolute activations, or None.
        hist_counts     : Counts per log-spaced |activation| bin of COUNT_EDGES.
    """

    HIST_EDGES       = np.logspace(-8.0, 4.0, 25)
    COUNT_EDGES      = np.concatenate([[0.0], np.logspace(-8.0, 4.0, 25), [np.inf]])
    DEAD_CHANNEL_ABS = 1e-8

    def __init__(self, name: str, module_type: str) -> None:
        """Initialises an empty accumulator for a named module.

        Args:
            name        : Dotted module path of the layer being recorded.
            module_type : Class name of the module.
        """
        self.name        = name
        self.module_type = module_type

        self.shape           = None
        self.first_seen      = None
        self.n_batches       = 0
        self.n_elements      = 0
        self.n_zero          = 0
        self.n_nonfinite     = 0
        self.value_sum       = 0.0
        self.value_sumsq     = 0.0
        self.max_abs         = 0.0
        self.channel_max_abs = None
        self.channel_abs_sum = None
        self.hist_counts     = np.zeros(self.COUNT_EDGES.size - 1, dtype=np.int64)

    def update(self, tensor: torch.Tensor, order: int) -> None:
        """Folds one forward-pass output into the running statistics.

        Args:
            tensor : Module output of shape (batch, ...); four-dimensional tensors (batch, channels, height, width) additionally contribute per-channel magnitude statistics.
            order  : Global forward-call index used to record first_seen.
        """
        t    = tensor.detach()
        flat = t.reshape(-1).float()

        finite_mask = torch.isfinite(flat)
        finite      = flat[finite_mask]

        if self.first_seen is None:
            self.first_seen = order

        self.shape        = tuple(t.shape[1:])
        self.n_batches   += 1
        self.n_elements  += flat.numel()
        self.n_zero      += int((flat == 0.0).sum())
        self.n_nonfinite += int(flat.numel() - finite_mask.sum())

        if finite.numel():
            self.value_sum    += float(finite.sum())
            self.value_sumsq  += float((finite * finite).sum())
            self.max_abs       = max(self.max_abs, float(finite.abs().max()))
            self.hist_counts  += np.histogram(finite.abs().cpu().numpy(), bins=self.COUNT_EDGES)[0]

        if t.ndim == 4:
            channel_abs = t.float().abs().nan_to_num(0.0)
            channel_max = channel_abs.amax(dim=(0, 2, 3)).cpu().numpy()
            channel_sum = channel_abs.sum(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)

            if self.channel_max_abs is None:
                self.channel_max_abs = channel_max
                self.channel_abs_sum = channel_sum
            else:
                self.channel_max_abs = np.maximum(self.channel_max_abs, channel_max)
                self.channel_abs_sum = self.channel_abs_sum + channel_sum

    @staticmethod
    def _gini(values: np.ndarray) -> float | None:
        """Returns the Gini concentration of a non-negative vector, or None when its sum is zero."""
        total = float(values.sum())
        if total <= 0.0:
            return None

        ordered = np.sort(values)
        ranks   = np.arange(1, ordered.size + 1, dtype=np.float64)

        return float(2.0 * (ranks * ordered).sum() / (ordered.size * total) - (ordered.size + 1.0) / ordered.size)

    @staticmethod
    def _effective_channel_frac(channel_abs_sum: np.ndarray) -> float:
        """Returns exp(entropy of the channel mass share) divided by the channel count."""
        total = float(channel_abs_sum.sum())
        if total <= 0.0:
            return 0.0

        share   = channel_abs_sum / total
        nonzero = share[share > 0.0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())

        return float(np.exp(entropy) / channel_abs_sum.size)

    def _abs_percentile(self, q: float) -> float:
        """Returns the upper bin edge holding the q-quantile of the non-zero |activation| values.

        Bin 0 of COUNT_EDGES holds the exact zeros and the values below 1e-8,
        which the zero fraction and the dead-channel count report separately.
        Taking the quantile over the remaining bins keeps a ReLU layer with
        half its outputs at zero from reading a p50 of 1e-8 and a p99/p50
        ratio near 1e8. The value is the bin's upper edge, an upper bound
        quantised to half a decade and capped at the observed maximum.

        Args:
            q : Quantile in [0, 1].

        Returns:
            The bound in activation units, or 0 when no value exceeds 1e-8.
        """
        counts = self.hist_counts[1:]
        total  = int(counts.sum())
        if total == 0:
            return 0.0

        cumulative = np.cumsum(counts)
        idx        = int(np.searchsorted(cumulative, q * total))
        idx        = min(idx, counts.size - 1)

        upper = self.COUNT_EDGES[idx + 2]
        return self.max_abs if not np.isfinite(upper) else float(min(upper, self.max_abs))

    def finalize(self) -> dict:
        """Returns the accumulated statistics as a JSON-serialisable summary.

        Returns:
            Mapping with layer identity, output shape, forward order, element
            and batch counts, non-finite and exact-zero fractions, mean, std,
            peak magnitude, the p50 and p99 of the non-zero magnitudes as
            upper half-decade bin edges, their ratio as dynamic range (None
            when no value exceeds 1e-8), channel counts, dead- and
            effective-channel measures, the channel Gini value, and the
            log-spaced magnitude histogram counts.

        Raises:
            ValueError : If the module never produced an activation.
        """
        if self.n_elements == 0:
            raise ValueError(f"Layer '{self.name}' recorded no activations; the module never fired during capture")

        n_finite = self.n_elements - self.n_nonfinite
        mean     = self.value_sum / n_finite if n_finite else float("nan")
        var      = self.value_sumsq / n_finite - mean * mean if n_finite else float("nan")
        std      = float(np.sqrt(max(var, 0.0))) if n_finite else float("nan")

        dead_channels      = int((self.channel_max_abs < self.DEAD_CHANNEL_ABS).sum()) if self.channel_max_abs is not None else None
        dead_channel_frac  = dead_channels / self.channel_max_abs.size if self.channel_max_abs is not None else None

        abs_p50       = self._abs_percentile(0.50)
        abs_p99       = self._abs_percentile(0.99)
        dynamic_range = abs_p99 / abs_p50 if abs_p50 > 0.0 else None

        effective_channel_frac = self._effective_channel_frac(self.channel_abs_sum) if self.channel_abs_sum is not None else None
        channel_gini           = self._gini(self.channel_abs_sum) if self.channel_abs_sum is not None else None

        return {
            "name"                   : self.name,
            "module_type"            : self.module_type,
            "shape"                  : list(self.shape) if self.shape is not None else None,
            "first_seen"             : self.first_seen,
            "n_batches"              : self.n_batches,
            "n_elements"             : self.n_elements,
            "nonfinite_frac"         : self.n_nonfinite / self.n_elements,
            "zero_frac"              : self.n_zero / self.n_elements,
            "mean"                   : mean,
            "std"                    : std,
            "max_abs"                : self.max_abs,
            "abs_p50"                : abs_p50,
            "abs_p99"                : abs_p99,
            "dynamic_range"          : dynamic_range,
            "n_channels"             : int(self.channel_max_abs.size) if self.channel_max_abs is not None else None,
            "dead_channels"          : dead_channels,
            "dead_channel_frac"      : dead_channel_frac,
            "effective_channel_frac" : effective_channel_frac,
            "channel_gini"           : channel_gini,
            "hist_counts"            : self.hist_counts.tolist(),
        }


class ActivationRecorder:
    """Attaches forward hooks to a model to record activation statistics or tensors.

    Attributes:
        model          : Model whose leaf modules are instrumented.
        include        : Optional whitelist of leaf module names to record; None records every leaf module.
        SITE_SEPARATOR : Separator between a module path and the call-site index in the statistics key of a reused module's later calls.
    """

    SITE_SEPARATOR = "#"

    def __init__(self, model: torch.nn.Module, include: list[str] | None = None) -> None:
        """Initialises the recorder with no hooks attached.

        Args:
            model   : Model to instrument.
            include : Optional leaf module names to restrict recording to.
        """
        self.model   = model
        self.include = list(include) if include is not None else None

        self._handles       = []
        self._stats         = {}
        self._store         = {}
        self._call_index    = 0
        self._forward_calls = {}
        self._call_sites    = {}

    def leaf_names(self) -> list[str]:
        """Returns the names of the leaf modules that will be recorded.

        Returns:
            Dotted names of childless modules, filtered by include when set.

        Raises:
            KeyError   : If a name in include is not a leaf module of the model.
            ValueError : If the model exposes no leaf modules at all.
        """
        names = [name for name, module in self.model.named_modules() if name and not any(module.children())]

        if self.include is not None:
            names = [name for name in names if name in self.include]
            missing = [name for name in self.include if name not in names]
            if missing:
                raise KeyError(f"Modules {missing} do not exist as leaf modules of {type(self.model).__name__}")

        if not names:
            raise ValueError(f"Model {type(self.model).__name__} exposes no leaf modules to record")

        return names

    @staticmethod
    def _first_tensor(output):
        """Returns the tensor a module output carries, or None when it holds none."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        return None

    def _reset_forward_hook(self):
        """Returns a forward pre-hook on the root module that starts a new per-forward call count for every leaf."""
        def hook(module, args):
            self._forward_calls = {}
        return hook

    def _stats_hook(self, name: str, module_type: str):
        """Returns a forward hook that folds the module output into the accumulator of its call site.

        A leaf module that fires several times in one forward pass, such as a
        pooling layer shared by every encoder level, gets one accumulator per
        call site, keyed "<name>#<k>" for its k-th call, so outputs of
        different shapes or depths are never merged into one row.

        Args:
            name        : Dotted module path of the hooked leaf.
            module_type : Class name of the hooked module.
        """
        def hook(module, args, output):
            tensor = self._first_tensor(output)
            if tensor is None:
                return

            site                      = self._forward_calls.get(name, 0)
            self._forward_calls[name] = site + 1
            self._call_sites[name]    = max(self._call_sites.get(name, 0), site + 1)

            key = name if site == 0 else f"{name}{self.SITE_SEPARATOR}{site}"
            if key not in self._stats:
                self._stats[key] = LayerActivationStats(key, module_type)

            self._call_index += 1
            self._stats[key].update(tensor, order=self._call_index)
        return hook

    def _store_hook(self, name: str):
        """Returns a forward hook that keeps a detached CPU copy of the module output."""
        def hook(module, args, output):
            tensor = self._first_tensor(output)
            if tensor is not None:
                self._store[name] = tensor.detach().clone().cpu()
        return hook

    def attach_stats(self) -> None:
        """Registers statistics hooks on every leaf module selected for recording, plus the per-forward reset on the root."""
        modules = dict(self.model.named_modules())
        self._handles.append(self.model.register_forward_pre_hook(self._reset_forward_hook()))

        for name in self.leaf_names():
            module_type       = type(modules[name]).__name__
            self._stats[name] = LayerActivationStats(name, module_type)
            self._handles.append(modules[name].register_forward_hook(self._stats_hook(name, module_type)))

    def attach_store(self, names: list[str]) -> None:
        """Registers hooks that retain the output tensors of the named modules.

        Args:
            names : Dotted module paths whose outputs should be kept.

        Raises:
            KeyError : If any name is not a module of the model.
        """
        modules = dict(self.model.named_modules())
        missing = [name for name in names if name not in modules]
        if missing:
            raise KeyError(f"Modules {missing} do not exist in {type(self.model).__name__}")

        for name in names:
            self._handles.append(modules[name].register_forward_hook(self._store_hook(name)))

    def detach(self) -> None:
        """Removes every hook this recorder registered on the model."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stats(self) -> dict[str, dict]:
        """Returns the finalised statistics of every call site that fired, keyed by module name and call site.

        Raises:
            ValueError : If no hooked module produced a tensor, which means the loader yielded no batch or every module output was non-tensor.
        """
        fired = {name: layer_stats.finalize() for name, layer_stats in self._stats.items() if layer_stats.n_elements > 0}
        if not fired:
            raise ValueError(f"No leaf module of {type(self.model).__name__} produced a tensor during capture; the loader yielded no batch or every module output was non-tensor")
        return fired

    @classmethod
    def module_of(cls, key: str) -> str:
        """Returns the dotted module path behind a statistics key, dropping the call-site suffix a reused module's later calls carry."""
        return key.split(cls.SITE_SEPARATOR, 1)[0]

    def reused_modules(self) -> dict[str, int]:
        """Returns the leaf modules that fired more than once in a forward pass, with the most calls seen in one pass."""
        return {name: calls for name, calls in self._call_sites.items() if calls > 1}

    def silent_layers(self) -> dict[str, str]:
        """Returns the hooked leaf modules that never produced a tensor, mapped to their module type.

        A leaf the eval forward never runs, or whose output holds no tensor,
        has no statistics row; the caller reports these so an unwired branch
        is not mistaken for a clean one.
        """
        return {name: layer_stats.module_type for name, layer_stats in self._stats.items() if layer_stats.n_elements == 0}

    def stored(self) -> dict[str, torch.Tensor]:
        """Returns the retained CPU output tensors, keyed by module name."""
        return dict(self._store)
