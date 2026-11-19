from __future__ import annotations

from dataclasses import fields
from pathlib     import Path

from models                                    import BACKBONE_CONFIG_REGISTRY
from tools.data.io                             import FileIO
from pipelines.shared.inference.run_classifier import RunArtifacts


class BackboneModelConfigIO:

    FILENAME = RunArtifacts.BACKBONE_CONFIG
    NAME_KEY = "model_name"
    EXCLUDED = {"shape_logger_types"}

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _serialize(cls, config) -> dict:
        return {f.name: getattr(config, f.name) for f in fields(config) if f.name not in cls.EXCLUDED}

    @classmethod
    def _expected_fields(cls, config) -> set:
        return {f.name for f in fields(config)} - cls.EXCLUDED

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
        payload  = FileIO.load_json(Path(meta_directory) / cls.FILENAME)
        raw_name = payload[cls.NAME_KEY]
        config   = BACKBONE_CONFIG_REGISTRY[cls._normalize(str(raw_name))]()
        values   = payload["config"]

        for key in cls._expected_fields(config) | values.keys():
            current = getattr(config, key)
            value   = values[key]

            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)

            setattr(config, key, value)

        return config, raw_name
