"""Name-to-class construction shared by every model family registry.

Each family (backbone, profile autoencoder, image autoencoder, unrolled) pairs a
model registry with a configuration registry and exposes the resulting
``build`` bound method as its ``get_*`` factory function.
"""

from __future__ import annotations


class RegistryFactory:
    """Builds models by registry name, resolving their configuration alongside.

    Attributes:
        model_registry  : Mapping from normalized architecture name to model class.
        config_registry : Mapping from the same names to configuration classes.
        kind            : Family label used in error messages, e.g. 'autoencoder'.
    """

    def __init__(self, model_registry: dict, config_registry: dict, kind: str) -> None:
        """Stores the paired registries and the family label.

        Args:
            model_registry  : Mapping from architecture name to model class.
            config_registry : Mapping from architecture name to config class.
            kind            : Family label used when reporting unknown names or overrides.
        """
        self.model_registry  = model_registry
        self.config_registry = config_registry
        self.kind            = kind

    def normalize(self, name: str) -> str:
        """Returns the registry key for a name, lowercased with separators as underscores."""
        return name.lower().replace("-", "_").replace(" ", "_")

    def resolve(self, name: str) -> str:
        """Returns the registry key for an architecture name.

        Args:
            name : Architecture name in any accepted spelling.

        Returns:
            The normalized key present in the model registry.

        Raises:
            ValueError : If the normalized name is not registered.
        """
        key = self.normalize(name)
        if key not in self.model_registry:
            raise ValueError(f"Unknown {self.kind} '{name}'. Available: {list(self.model_registry.keys())}")
        return key

    def resolve_config(self, key: str, config, overrides: dict):
        """Returns the configuration to build with, applying the overrides.

        Args:
            key       : Normalized registry key of the architecture.
            config    : Existing configuration object, or None to instantiate the registered configuration class from the overrides alone.
            overrides : Field name to value assignments applied to the config.

        Returns:
            The configuration object to pass to the model constructor.

        Raises:
            AttributeError : If an override names a field the configuration does not declare.
        """
        if config is None:
            return self.config_registry[key](**overrides)

        for attribute, value in overrides.items():
            if not hasattr(config, attribute):
                raise AttributeError(f"Unknown {self.kind} config override '{attribute}' for {type(config).__name__}")
            setattr(config, attribute, value)
        return config

    def build(self, name: str, config=None, **overrides):
        """Instantiates a registered model together with its configuration.

        Args:
            name        : Architecture name in any accepted spelling.
            config      : Existing configuration object, or None to build one from the registered configuration class.
            **overrides : Configuration field assignments applied before build.

        Returns:
            Tuple of the instantiated model and the configuration it was built
            with.
        """
        key    = self.resolve(name)
        config = self.resolve_config(key, config, overrides)
        return self.model_registry[key](config), config
