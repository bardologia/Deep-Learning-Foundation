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
            channel_max = t.float().abs().nan_to_num(0.0).amax(dim=(0, 2, 3)).cpu().numpy()

            if self.channel_max_abs is None:
                self.channel_max_abs = channel_max
            else:
                self.channel_max_abs = np.maximum(self.channel_max_abs, channel_max)

    def _abs_percentile(self, q: float) -> float:
        total = int(self.hist_counts.sum())
        if total == 0:
            return 0.0

        cumulative = np.cumsum(self.hist_counts)
        idx        = int(np.searchsorted(cumulative, q * total))
        idx        = min(idx, self.hist_counts.size - 1)

        upper = self.COUNT_EDGES[idx + 1]
        return self.max_abs if not np.isfinite(upper) else float(upper)

    def finalize(self) -> dict:
        if self.n_elements == 0:
            raise ValueError(f"Layer '{self.name}' recorded no activations; the module never fired during capture")

        n_finite = self.n_elements - self.n_nonfinite
        mean     = self.value_sum / n_finite if n_finite else float("nan")
        var      = self.value_sumsq / n_finite - mean * mean if n_finite else float("nan")
        std      = float(np.sqrt(max(var, 0.0))) if n_finite else float("nan")

        dead_channels      = int((self.channel_max_abs < self.DEAD_CHANNEL_ABS).sum()) if self.channel_max_abs is not None else None
        dead_channel_frac  = dead_channels / self.channel_max_abs.size if self.channel_max_abs is not None else None

        return {
            "name"              : self.name,
            "module_type"       : self.module_type,
            "shape"             : list(self.shape) if self.shape is not None else None,
            "first_seen"        : self.first_seen,
            "n_batches"         : self.n_batches,
            "n_elements"        : self.n_elements,
            "nonfinite_frac"    : self.n_nonfinite / self.n_elements,
            "zero_frac"         : self.n_zero / self.n_elements,
            "mean"              : mean,
            "std"               : std,
            "max_abs"           : self.max_abs,
            "abs_p99"           : self._abs_percentile(0.99),
            "n_channels"        : int(self.channel_max_abs.size) if self.channel_max_abs is not None else None,
            "dead_channels"     : dead_channels,
            "dead_channel_frac" : dead_channel_frac,
            "hist_counts"       : self.hist_counts.tolist(),
        }


class ActivationRecorder:

    def __init__(self, model: torch.nn.Module, include: list[str] | None = None) -> None:
        self.model   = model
        self.include = list(include) if include is not None else None

        self._handles    = []
        self._stats      = {}
        self._store      = {}
        self._call_index = 0

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

    def _stats_hook(self, name: str):
        def hook(module, args, output):
            tensor = self._first_tensor(output)
            if tensor is not None:
                self._call_index += 1
                self._stats[name].update(tensor, order=self._call_index)
        return hook

    def _store_hook(self, name: str):
        def hook(module, args, output):
            tensor = self._first_tensor(output)
            if tensor is not None:
                self._store[name] = tensor.detach().clone().cpu()
        return hook

    def attach_stats(self) -> None:
        modules = dict(self.model.named_modules())

        for name in self.leaf_names():
            self._stats[name] = LayerActivationStats(name, type(modules[name]).__name__)
            self._handles.append(modules[name].register_forward_hook(self._stats_hook(name)))

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
        return {name: layer_stats.finalize() for name, layer_stats in self._stats.items() if layer_stats.n_elements > 0}

    def stored(self) -> dict[str, torch.Tensor]:
        return dict(self._store)
