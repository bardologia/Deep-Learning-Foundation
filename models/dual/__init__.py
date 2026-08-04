from configuration.architectures import DualResUNetConfig
from ..registry                  import RegistryFactory
from .dual_resunet               import DualResUNet


DUAL_MODEL_REGISTRY: dict[str, type] = {
    "dual_resunet" : DualResUNet,
}

DUAL_CONFIG_REGISTRY: dict[str, type] = {
    "dual_resunet" : DualResUNetConfig,
}


get_dual = RegistryFactory(DUAL_MODEL_REGISTRY, DUAL_CONFIG_REGISTRY, "dual").build


__all__ = [
    "DualResUNet",
    "DualResUNetConfig",
    "DUAL_MODEL_REGISTRY",
    "DUAL_CONFIG_REGISTRY",
    "get_dual",
]
