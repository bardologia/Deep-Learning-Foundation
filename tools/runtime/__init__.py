import importlib

_EXPORTS = {
    "Detacher"           : "detacher",
    "CompletionMarker"   : "completion",
    "ConfigCli"          : "config_cli",
    "CondaEnv"           : "conda_env",
    "CondaJobDispatcher" : "conda_env",
    "Reproducibility"    : "reproducibility",
    "WorkerInitializer"  : "reproducibility",
}

__all__ = [
    "Detacher",
    "CompletionMarker",
    "ConfigCli",
    "CondaEnv",
    "CondaJobDispatcher",
    "Reproducibility",
    "WorkerInitializer",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
