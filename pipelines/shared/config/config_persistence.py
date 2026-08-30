"""Persistence of architecture and dataset configurations alongside a run.

Every trained run stores the dataclass that defined its architecture under meta/,
so a checkpoint can be reconstructed faithfully at inference time. Loading is
strict: an unknown model name or a field the persisted payload does not carry
raises rather than falling back on current defaults.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib     import Path

from configuration.dataset.profile_autoencoder import ProfileAugmentationConfig, ProfileDatasetConfig
from models                                    import BACKBONE_CONFIG_REGISTRY
from models.dual                               import DUAL_CONFIG_REGISTRY
from models.image_autoencoder                  import IMAGE_AE_CONFIG_REGISTRY
from models.profile_autoencoder                import PROFILE_AE_CONFIG_REGISTRY
from tools.data.io                             import FileIO
from tools.data.regions                        import CropRegion, SplitRegions
from pipelines.shared.inference.run_classifier import RunArtifacts


class ConfigIO:
    """Base for the per-family architecture config writers and readers.

    Attributes:
        FILENAME     : Name of the JSON file inside the run's meta/ directory.
        NAME_KEY     : Payload key holding the registry name of the architecture.
        MISSING_NOUN : Family noun used in the "file not found" message.
        UNKNOWN_NOUN : Family noun used in the "unknown architecture" message.
    """

    FILENAME     : str
    NAME_KEY     = "model_name"
    MISSING_NOUN : str
    UNKNOWN_NOUN : str

    @staticmethod
    def _normalize(name: str) -> str:
        """Returns the registry key of a model name: lowercase with underscores."""
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _registry(cls) -> dict:
        """Returns the family's name-to-config-factory registry.

        Raises:
            NotImplementedError : Always; subclasses supply the registry.
        """
        raise NotImplementedError

    @classmethod
    def _serialize(cls, config) -> dict:
        """Returns the JSON-ready field mapping of an architecture config.

        Raises:
            NotImplementedError : Always; subclasses choose the serialisation.
        """
        raise NotImplementedError

    @classmethod
    def _expected_fields(cls, config) -> set:
        """Returns the dataclass field names a persisted payload must carry."""
        return {f.name for f in fields(config)}

    @classmethod
    def exists(cls, meta_directory: Path) -> bool:
        """True when this family's config file is present in the meta directory."""
        return (Path(meta_directory) / cls.FILENAME).is_file()

    @classmethod
    def save(cls, config, model_name: str, meta_directory: Path) -> Path:
        """Writes the architecture config next to the run and returns the file path.

        Args:
            config         : Architecture configuration dataclass instance.
            model_name     : Registry name of the architecture.
            meta_directory : Run's meta/ directory.
        """
        payload = {
            cls.NAME_KEY  : model_name,
            "config_type" : type(config).__name__,
            "config"      : cls._serialize(config),
        }

        return FileIO.save_json(payload, Path(meta_directory) / cls.FILENAME)

    @classmethod
    def load(cls, meta_directory: Path):
        """Reconstructs the persisted architecture config of a run.

        Args:
            meta_directory : Run's meta/ directory.

        Returns:
            Pair of the rebuilt config dataclass and the raw persisted model name.

        Raises:
            FileNotFoundError : The run predates config persistence.
            ValueError        : The persisted name is not in the registry, the payload lacks fields the current dataclass declares, or it carries a field the dataclass no longer has.
        """
        path = Path(meta_directory) / cls.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No {cls.FILENAME} under {meta_directory}; the {cls.MISSING_NOUN} architecture was not persisted at training time and the checkpoint cannot be reconstructed faithfully. Retrain to regenerate it.")

        payload  = FileIO.load_json(path)
        raw_name = payload[cls.NAME_KEY]
        name     = cls._normalize(str(raw_name))

        registry = cls._registry()
        if name not in registry:
            raise ValueError(f"Persisted {cls.NAME_KEY} '{name}' is not a known {cls.UNKNOWN_NOUN}: {list(registry.keys())}")

        config = registry[name]()

        missing = cls._expected_fields(config) - payload["config"].keys()
        if missing:
            raise ValueError(f"Persisted {cls.FILENAME} under {meta_directory} lacks fields {sorted(missing)} of {type(config).__name__}; the architecture definition gained fields since this checkpoint was trained and current defaults cannot be assumed. Retrain to regenerate it.")

        for key, value in payload["config"].items():
            if not hasattr(config, key):
                raise ValueError(f"Persisted field '{key}' is not an attribute of {type(config).__name__}; the architecture definition changed since this checkpoint was trained")

            current = getattr(config, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)

            setattr(config, key, value)

        return config, raw_name


class ModelConfigIO(ConfigIO):
    """Config IO for the families whose dataclasses carry non-serialisable fields.

    Attributes:
        EXCLUDED : Field names dropped from the payload and from the completeness check, because they hold runtime-only shape-logging types.
    """

    EXCLUDED = {"shape_logger_types"}

    @classmethod
    def _serialize(cls, config) -> dict:
        """Returns the config fields as a flat dict, minus the excluded runtime fields."""
        return {f.name: getattr(config, f.name) for f in fields(config) if f.name not in cls.EXCLUDED}

    @classmethod
    def _expected_fields(cls, config) -> set:
        """Returns the required field names, minus the excluded runtime fields."""
        return super()._expected_fields(config) - cls.EXCLUDED


class BackboneModelConfigIO(ModelConfigIO):
    """Persists the backbone architecture config of a spatial run."""

    FILENAME     = RunArtifacts.BACKBONE_CONFIG
    MISSING_NOUN = "backbone"
    UNKNOWN_NOUN = "architecture"

    @classmethod
    def _registry(cls) -> dict:
        """Returns the backbone architecture registry."""
        return BACKBONE_CONFIG_REGISTRY


class ProfileAutoencoderConfigIO(ConfigIO):
    """Persists the architecture config of a 1-D profile autoencoder."""

    FILENAME     = RunArtifacts.PROFILE_AE_CONFIG
    MISSING_NOUN = "profile autoencoder"
    UNKNOWN_NOUN = "profile autoencoder"

    @classmethod
    def _registry(cls) -> dict:
        """Returns the profile-autoencoder architecture registry."""
        return PROFILE_AE_CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        """Returns the config as a recursively converted dict."""
        return asdict(config)


class ImageAutoencoderConfigIO(ConfigIO):
    """Persists the architecture config of a 2-D image autoencoder."""

    FILENAME     = RunArtifacts.IMAGE_AE_CONFIG
    MISSING_NOUN = "image autoencoder"
    UNKNOWN_NOUN = "image autoencoder"

    @classmethod
    def _registry(cls) -> dict:
        """Returns the image-autoencoder architecture registry."""
        return IMAGE_AE_CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        """Returns the config as a recursively converted dict."""
        return asdict(config)


class DualModelConfigIO(ModelConfigIO):
    """Persists the architecture config of a dual-trunk model."""

    FILENAME     = RunArtifacts.DUAL_CONFIG
    MISSING_NOUN = "dual-input"
    UNKNOWN_NOUN = "dual-input model"

    @classmethod
    def _registry(cls) -> dict:
        """Returns the dual-model architecture registry."""
        return DUAL_CONFIG_REGISTRY


class ProfileDatasetConfigIO:
    """Persists the dataset configuration of a profile-autoencoder run.

    Attributes:
        FILENAME : Name of the JSON file inside the run's meta/ directory.
    """

    FILENAME = "profile_dataset_config.json"

    @staticmethod
    def exists(meta_directory: Path) -> bool:
        """True when the profile dataset config is present in the meta directory."""
        return (Path(meta_directory) / ProfileDatasetConfigIO.FILENAME).is_file()

    @staticmethod
    def save(config, meta_directory: Path) -> Path:
        """Writes the profile dataset config, stringifying its paths, and returns the file path.

        Args:
            config         : Profile dataset configuration to persist.
            meta_directory : Run's meta/ directory.
        """
        payload = asdict(config)

        payload["preprocessing_run_directory"] = str(config.preprocessing_run_directory)
        payload["parameters_path"]             = str(config.parameters_path) if config.parameters_path is not None else None

        return FileIO.save_json(payload, Path(meta_directory) / ProfileDatasetConfigIO.FILENAME)

    @staticmethod
    def _parse_split(value):
        """Returns one split entry as a CropRegion, or a list of them when disjoint."""
        if isinstance(value, list):
            return [CropRegion(**region) for region in value]
        return CropRegion(**value)

    @staticmethod
    def load(meta_directory: Path):
        """Reconstructs the persisted profile dataset configuration of a run.

        Args:
            meta_directory : Run's meta/ directory.

        Returns:
            The rebuilt profile dataset configuration, with its split regions and
            augmentation block restored.

        Raises:
            FileNotFoundError : The run predates dataset-config persistence and must be backfilled or retrained.
        """
        path = Path(meta_directory) / ProfileDatasetConfigIO.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No {ProfileDatasetConfigIO.FILENAME} under {meta_directory}; this profile-autoencoder run predates dataset-config persistence and is not self-describing. Retrain the run.")

        payload = FileIO.load_json(path)
        splits  = payload["split_regions"]

        split_regions = SplitRegions(
            train = ProfileDatasetConfigIO._parse_split(splits["train"]),
            val   = ProfileDatasetConfigIO._parse_split(splits["val"]),
            test  = ProfileDatasetConfigIO._parse_split(splits["test"]),
        )

        return ProfileDatasetConfig(
            preprocessing_run_directory = Path(payload["preprocessing_run_directory"]),
            split_regions               = split_regions,
            parameters_path             = Path(payload["parameters_path"]) if payload["parameters_path"] is not None else None,
            n_gaussians                 = int(payload["n_gaussians"]),
            x_min                       = float(payload["x_min"]),
            x_max                       = float(payload["x_max"]),
            pixel_subsample             = float(payload["pixel_subsample"]),
            keep_empty_frac             = float(payload["keep_empty_frac"]),
            amp_zero_thr                = float(payload["amp_zero_thr"]),
            batch_size                  = int(payload["batch_size"]),
            num_workers                 = int(payload["num_workers"]),
            prefetch_factor             = int(payload["prefetch_factor"]),
            pin_memory                  = bool(payload["pin_memory"]),
            shuffle_train               = bool(payload["shuffle_train"]),
            stats_max_samples           = int(payload["stats_max_samples"]),
            augmentation                = ProfileAugmentationConfig(**payload["augmentation"]),
        )
