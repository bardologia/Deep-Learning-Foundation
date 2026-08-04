from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path
from typing      import Literal, Tuple

from configuration.sar.processing_config import ProcessingConfig, TomogramConfig
from tools.data.io                       import FileIO
from tools.monitoring.logger             import Logger
from tools.baselines                     import TrackProfiles


ArtifactType = Literal["tomogram_full", "dem_full", "primary", "secondaries", "interferograms", "track_profiles"]


class ArtifactRegistry:
    def __init__(self, config: ProcessingConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def ensure_directory_structure(self) -> None:
        paths = self.config.paths
        self.logger.section("[Directory Validation]")
        for directory in (paths.data_directory, paths.metadata_directory, paths.temporary_directory):
            target_dir = FileIO.ensure_dir(directory.resolve())
            self.logger.subsection(f"Ensured path : {target_dir}")

    def artifact_filenames(self) -> dict[str, str]:
        return {
            "tomogram_full"  : "tomogram_full.npy",
            "dem_full"       : "dem_full.npy",
            "primary"        : "primary.npy",
            "secondaries"    : "secondaries.npy",
            "interferograms" : "interferograms.npy",
            "track_profiles" : TrackProfiles.FILENAME,
        }

    def artifact_path(self, artifact_type: ArtifactType) -> Path:
        filenames = self.artifact_filenames()
        return self.config.paths.data_directory / filenames[artifact_type]


class MetadataManager:
    def __init__(self, config: ProcessingConfig, logger: Logger) -> None:
        self.config   = config
        self.logger   = logger
        self.registry = ArtifactRegistry(config, logger)

        self.logger.section("[MetadataManager Initialization]")
        self.logger.subsection(f"Run Directory : {config.paths.run_directory}")
        self.logger.subsection(f"Tomogram Tag  : {config.tomogram_tag}")
        self.logger.subsection(f"Parameter Tag : {config.parameter_tag}")

    def build_tomogram_metadata(self, output_path: Path, stack_identifier: str, cfg: TomogramConfig) -> dict[str, str]:
        return {
            "tomo_full"    : str(output_path),
            "crop"         : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
            "FuSARproject" : cfg.fusar_project_path,
            "id"           : stack_identifier,
            "basedir"      : cfg.base_directory,
            "polarisation" : cfg.polarisation,
            "select"       : cfg.track_selection,
            "range"        : f"[{', '.join(str(v) for v in cfg.height_range)}]",
            "filter"       : cfg.filter_method,
            "method"       : cfg.beamforming_method,
            "win"          : f"[{', '.join(str(v) for v in cfg.filter_arguments['win'])}]",
        }

    def build_inputs_metadata(self, primary_path: Path, secondaries_path: Path, interferograms_path: Path, primary_shape: Tuple[int, ...], secondaries_shape: Tuple[int, ...], interferograms_shape: Tuple[int, ...]) -> dict[str, str]:
        cfg = self.config.tomogram_config
        return {
            "primary_path"         : str(primary_path),
            "secondaries_path"     : str(secondaries_path),
            "interferograms_path"  : str(interferograms_path),
            "primary_shape"        : f"[{', '.join(str(v) for v in primary_shape)}]",
            "secondaries_shape"    : f"[{', '.join(str(v) for v in secondaries_shape)}]",
            "interferograms_shape" : f"[{', '.join(str(v) for v in interferograms_shape)}]",
            "crop"                 : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
            "FuSARproject"         : cfg.fusar_project_path,
            "id"                   : self.config.stack_identifier,
            "basedir"              : cfg.base_directory,
            "polarisation"         : cfg.polarisation,
            "select"               : cfg.track_selection,
            "data_type"            : self.config.dataset_type,
        }

    def save_stage_metadata(self, stage_name: str, metadata_entries: dict[str, str]) -> Path:
        self.registry.ensure_directory_structure()
        meta_filename = f"meta_{stage_name}.txt"
        meta_path     = self.config.paths.metadata_directory / meta_filename

        self.logger.section(f"[Saving Metadata] Stage: {stage_name}")
        FileIO.save_text_metadata(metadata_entries, meta_path)

        self.logger.subsection(f"Metadata written: {meta_path}")
        return meta_path

    def save_pipeline_configuration(self) -> Path:
        self.registry.ensure_directory_structure()

        dump_path = self.config.paths.metadata_directory / "config_state.json"

        config_dict = asdict(self.config)

        self.logger.section("[Saving Configuration State]")
        FileIO.save_json(config_dict, dump_path)

        self.logger.subsection(f"Configuration preserved at: {dump_path}")
        return dump_path

    def save_dataset_layout(self, pass_labels: list | None = None) -> Path:
        self.registry.ensure_directory_structure()

        layout = {
            "global_crop"        : list(self.config.crop.as_tuple()),
            "dataset_type"       : self.config.dataset_type,
            "tomogram_tag"       : self.config.tomogram_tag,
            "parameter_tag"      : self.config.parameter_tag,
            "max_amplitude_clip" : self.config.tomogram_config.max_amplitude_clip,
            "pass_labels"        : list(pass_labels) if pass_labels is not None else None,
            "artifacts"          : self.registry.artifact_filenames(),
        }

        out_path = self.config.paths.data_directory / "dataset.json"
        FileIO.save_json(layout, out_path, indent=2)

        self.logger.section(f"[Dataset Layout Saved] {out_path}")
        return out_path
