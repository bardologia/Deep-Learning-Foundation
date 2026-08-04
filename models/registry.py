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
        if key not in self.model_registry:
            raise ValueError(f"Unknown {self.kind} '{name}'. Available: {list(self.model_registry.keys())}")
        return key

    def resolve_config(self, key: str, config, overrides: dict):
        if config is None:
            return self.config_registry[key](**overrides)

        for attribute, value in overrides.items():
            if not hasattr(config, attribute):
                raise AttributeError(f"Unknown {self.kind} config override '{attribute}' for {type(config).__name__}")
            setattr(config, attribute, value)
        return config

    def build(self, name: str, config=None, **overrides):
        key    = self.resolve(name)
        config = self.resolve_config(key, config, overrides)
        return self.model_registry[key](config), config
