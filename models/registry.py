
from __future__ import annotations


class RegistryFactory:

    def __init__(self, model_registry: dict, config_registry: dict, kind: str) -> None:
        self.model_registry  = model_registry
        self.config_registry = config_registry
        self.kind            = kind

    def normalize(self, name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    def resolve(self, name: str) -> str:
        key = self.normalize(name)
        return key

    def resolve_config(self, key: str, config, overrides: dict):
        if config is None:
            return self.config_registry[key](**overrides)

        for attribute, value in overrides.items():
            setattr(config, attribute, value)
        return config

    def build(self, name: str, config=None, **overrides):
        key    = self.resolve(name)
        config = self.resolve_config(key, config, overrides)
        return self.model_registry[key](config), config
