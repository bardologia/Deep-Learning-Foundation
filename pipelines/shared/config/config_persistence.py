from __future__ import annotations

from dataclasses import asdict, fields
from pathlib     import Path

from configuration.dataset.profile_autoencoder import ProfileAugmentationConfig, ProfileDatasetConfig
from models                                    import BACKBONE_CONFIG_REGISTRY
from models.dual                               import DUAL_CONFIG_REGISTRY
from models.image_autoencoder                  import IMAGE_AE_CONFIG_REGISTRY
from models.profile_autoencoder                import PROFILE_AE_CONFIG_REGISTRY
from models.unrolled                           import UNROLLED_CONFIG_REGISTRY
from tools.data.io                             import FileIO
from tools.data.regions                        import CropRegion, SplitRegions
from pipelines.shared.inference.run_classifier import RunArtifacts


class ConfigIO:
    FILENAME     : str
    NAME_KEY     = "model_name"
    MISSING_NOUN : str
    UNKNOWN_NOUN : str

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _registry(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def _serialize(cls, config) -> dict:
        raise NotImplementedError

    @classmethod
    def _expected_fields(cls, config) -> set:
        return {f.name for f in fields(config)}

    @classmethod
    def exists(cls, meta_directory: Path) -> bool:
        return (Path(meta_directory) / cls.FILENAME).is_file()

    @classmethod
    def save(cls, config, model_name: str, meta_directory: Path) -> Path:
        payload = {
            cls.NAME_KEY  : model_name,
            "config_type" : type(config).__name__,
            "config"      : cls._serialize(config),
        }

        return FileIO.save_json(payload, Path(meta_directory) / cls.FILENAME)

    @classmethod
    def load(cls, meta_directory: Path):
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
    EXCLUDED = {"shape_logger_types"}

    @classmethod
    def _serialize(cls, config) -> dict:
        return {f.name: getattr(config, f.name) for f in fields(config) if f.name not in cls.EXCLUDED}

    @classmethod
    def _expected_fields(cls, config) -> set:
        return super()._expected_fields(config) - cls.EXCLUDED


class BackboneModelConfigIO(ModelConfigIO):
    FILENAME     = RunArtifacts.BACKBONE_CONFIG
    MISSING_NOUN = "backbone"
    UNKNOWN_NOUN = "architecture"

    @classmethod
    def _registry(cls) -> dict:
        return BACKBONE_CONFIG_REGISTRY


class ProfileAutoencoderConfigIO(ConfigIO):
    FILENAME     = RunArtifacts.PROFILE_AE_CONFIG
    MISSING_NOUN = "profile autoencoder"
    UNKNOWN_NOUN = "profile autoencoder"

    @classmethod
    def _registry(cls) -> dict:
        return PROFILE_AE_CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        return asdict(config)


class ImageAutoencoderConfigIO(ConfigIO):
    FILENAME     = RunArtifacts.IMAGE_AE_CONFIG
    MISSING_NOUN = "image autoencoder"
    UNKNOWN_NOUN = "image autoencoder"

    @classmethod
    def _registry(cls) -> dict:
        return IMAGE_AE_CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        return asdict(config)


class UnrolledModelConfigIO(ModelConfigIO):
    FILENAME     = RunArtifacts.UNROLLED_CONFIG
    MISSING_NOUN = "unrolled"
    UNKNOWN_NOUN = "unrolled model"

    @classmethod
    def _registry(cls) -> dict:
        return UNROLLED_CONFIG_REGISTRY


class DualModelConfigIO(ModelConfigIO):
    FILENAME     = RunArtifacts.DUAL_CONFIG
    MISSING_NOUN = "dual-input"
    UNKNOWN_NOUN = "dual-input model"

    @classmethod
    def _registry(cls) -> dict:
        return DUAL_CONFIG_REGISTRY


class ProfileDatasetConfigIO:
    FILENAME = "profile_dataset_config.json"

    @staticmethod
    def exists(meta_directory: Path) -> bool:
        return (Path(meta_directory) / ProfileDatasetConfigIO.FILENAME).is_file()

    @staticmethod
    def save(config, meta_directory: Path) -> Path:
        payload = asdict(config)

        payload["preprocessing_run_directory"] = str(config.preprocessing_run_directory)
        payload["parameters_path"]             = str(config.parameters_path) if config.parameters_path is not None else None

        return FileIO.save_json(payload, Path(meta_directory) / ProfileDatasetConfigIO.FILENAME)

    @staticmethod
    def _parse_split(value):
        if isinstance(value, list):
            return [CropRegion(**region) for region in value]
        return CropRegion(**value)

    @staticmethod
    def load(meta_directory: Path):
        path = Path(meta_directory) / ProfileDatasetConfigIO.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No {ProfileDatasetConfigIO.FILENAME} under {meta_directory}; this profile-autoencoder run predates dataset-config persistence and is not self-describing. Regenerate it with 'python scripts/backfill_profile_dataset_config.py <run_directory>' (or retrain).")

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
