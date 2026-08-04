from configuration.architectures import GammaNetConfig
from ..registry                  import RegistryFactory
from .gamma_net                  import GammaNet, ProfileProx, TomoOperator


UNROLLED_MODEL_REGISTRY: dict[str, type] = {
    "gamma_net" : GammaNet,
}

UNROLLED_CONFIG_REGISTRY: dict[str, type] = {
    "gamma_net" : GammaNetConfig,
}


get_unrolled = RegistryFactory(UNROLLED_MODEL_REGISTRY, UNROLLED_CONFIG_REGISTRY, "unrolled").build


__all__ = [
    "GammaNet",
    "GammaNetConfig",
    "ProfileProx",
    "TomoOperator",
    "UNROLLED_MODEL_REGISTRY",
    "UNROLLED_CONFIG_REGISTRY",
    "get_unrolled",
]
