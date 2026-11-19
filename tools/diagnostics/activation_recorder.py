
from __future__ import annotations

import numpy as np
import torch


class LayerActivationStats:

    HIST_EDGES       = np.logspace(-8.0, 4.0, 25)
    COUNT_EDGES      = np.concatenate([[0.0], np.logspace(-8.0, 4.0, 25), [np.inf]])
    DEAD_CHANNEL_ABS = 1e-8

    def __init__(self, name: str, module_type: str) -> None:
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
        total = float(values.sum())
        if total <= 0.0:
            return None

        ordered = np.sort(values)
        ranks   = np.arange(1, ordered.size + 1, dtype=np.float64)

        return float(2.0 * (ranks * ordered).sum() / (ordered.size * total) - (ordered.size + 1.0) / ordered.size)

    @staticmethod
    def _effective_channel_frac(channel_abs_sum: np.ndarray) -> float:
        total = float(channel_abs_sum.sum())
        if total <= 0.0:
            return 0.0

        share   = channel_abs_sum / total
        nonzero = share[share > 0.0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())

        return float(np.exp(entropy) / channel_abs_sum.size)

    def _abs_percentile(self, q: float) -> float:
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

    SITE_SEPARATOR = "#"

    def __init__(self, model: torch.nn.Module, include: list[str] | None = None) -> None:
        self.model   = model
        self.include = list(include) if include is not None else None

        self._handles       = []
        self._stats         = {}
        self._store         = {}
        self._call_index    = 0
        self._forward_calls = {}
        self._call_sites    = {}

    def leaf_names(self) -> list[str]:
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
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        return None

    def _reset_forward_hook(self):
        def hook(module, args):
            self._forward_calls = {}
        return hook

    def _stats_hook(self, name: str, module_type: str):
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
        def hook(module, args, output):
            tensor = self._first_tensor(output)
            if tensor is not None:
                self._store[name] = tensor.detach().clone().cpu()
        return hook

    def attach_stats(self) -> None:
        modules = dict(self.model.named_modules())
        self._handles.append(self.model.register_forward_pre_hook(self._reset_forward_hook()))

        for name in self.leaf_names():
            module_type       = type(modules[name]).__name__
            self._stats[name] = LayerActivationStats(name, module_type)
            self._handles.append(modules[name].register_forward_hook(self._stats_hook(name, module_type)))

    def attach_store(self, names: list[str]) -> None:
        modules = dict(self.model.named_modules())
        missing = [name for name in names if name not in modules]
        if missing:
            raise KeyError(f"Modules {missing} do not exist in {type(self.model).__name__}")

        for name in names:
            self._handles.append(modules[name].register_forward_hook(self._store_hook(name)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stats(self) -> dict[str, dict]:
        fired = {name: layer_stats.finalize() for name, layer_stats in self._stats.items() if layer_stats.n_elements > 0}
        if not fired:
            raise ValueError(f"No leaf module of {type(self.model).__name__} produced a tensor during capture; the loader yielded no batch or every module output was non-tensor")
        return fired

    @classmethod
    def module_of(cls, key: str) -> str:
        return key.split(cls.SITE_SEPARATOR, 1)[0]

    def reused_modules(self) -> dict[str, int]:
        return {name: calls for name, calls in self._call_sites.items() if calls > 1}

    def silent_layers(self) -> dict[str, str]:
        return {name: layer_stats.module_type for name, layer_stats in self._stats.items() if layer_stats.n_elements == 0}

    def stored(self) -> dict[str, torch.Tensor]:
        return dict(self._store)
