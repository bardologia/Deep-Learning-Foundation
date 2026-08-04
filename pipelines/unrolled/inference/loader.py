from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np

from configuration.inference.unrolled           import UnrolledInferenceConfig
from configuration.sar.gaussian_config          import GaussianConfig
from configuration.training                     import UnrolledEntryConfig
from models.unrolled                            import get_unrolled
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.stats           import Stats
from pipelines.profile_autoencoder.dataset      import ParameterCropper
from pipelines.shared.config.config_persistence import UnrolledModelConfigIO
from pipelines.shared.inference.run_artifacts   import RunArtifactLoader
from pipelines.shared.dataset.dataset_spatial   import Layout
from tools.data.gaussians                       import GaussianAxis
from tools.data.io                              import FileIO
from tools.data.regions                         import CropRegion
from tools.runtime.config_cli                   import ConfigCli
from tools.sar                                  import GeometryField


@dataclass
class UnrolledRun:
    model            : object
    model_name       : str
    model_config     : object
    gt_parameters    : np.ndarray
    kz_field         : np.ndarray
    x_axis           : np.ndarray
    ppg              : int
    n_gaussians      : int
    split_name       : str
    split_region     : CropRegion
    curve_loss       : str
    power_floor      : float
    noise_std        : float
    checkpoint_path  : Path
    training_summary : dict


class UnrolledRunLoader(RunArtifactLoader):
    def _load_entry_config(self) -> UnrolledEntryConfig:
        path = self.run_directory / "docs" / "resolved_entry_config.json"

        return ConfigCli.load_resolved(UnrolledEntryConfig(), path)

    def _load_training_summary(self) -> dict:
        path = self.run_directory / "training_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"No training_summary.json under {self.run_directory}; the run never finished training, re-train it before running inference.")

        return FileIO.load_json(path)

    def _build_unrolled_model(self, device: str, checkpoint_name: str):
        model_config, model_name = UnrolledModelConfigIO.load(self.meta_directory)
        model, _                 = get_unrolled(model_name, config=model_config)

        checkpoint_path = self.run_directory / checkpoint_name
        if not checkpoint_path.is_file():
            available = sorted(entry.name for entry in self.run_directory.glob("*.pt"))
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found; available checkpoints: {available}")

        checkpoint, _x_axis, checkpoint_meta = self._load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["params"])

        model = model.to(device)
        model.eval()

        return model, model_name, model_config, checkpoint_path, checkpoint_meta

    def _region_ground_truth(self, dataset_config, split: str, normalizer: Normalizer):
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = ParameterCropper(layout, dataset_config.split_regions, logger=self.logger)

        regions = dataset_config.split_regions.regions(split)
        if len(regions) != 1:
            raise ValueError(f"Unrolled inference requires a single contiguous region for split '{split}'; found {len(regions)} disjoint regions.")

        parameters = cropper.load_split(split)[0]
        indices    = dataset_config.output_config.selected_indices(n_gaussians=dataset_config.n_gaussians)

        raw      = np.ascontiguousarray(parameters[indices], dtype=np.float32)
        physical = normalizer.denormalize_output(normalizer.normalize_output(raw))

        return np.ascontiguousarray(physical, dtype=np.float32), regions[0], layout

    def _region_kz(self, dataset_config, layout: Layout, region: CropRegion, height_axis_convention: str) -> np.ndarray:
        path = Path(dataset_config.preprocessing_run_directory) / "meta" / GeometryField.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Unrolled inference requires the per-pixel geometry field but {path} is missing; re-run preprocessing to generate it.")

        field = GeometryField.load(path).subset(dataset_config.secondary_labels)
        field.validate_extent(layout.global_crop)

        azimuth_slice, range_slice = region.local_slices(layout.global_crop)

        return field.slice(azimuth_slice, range_slice).kz(height_axis_convention).astype(np.float32)

    def load(self, *, config: UnrolledInferenceConfig, device: str) -> UnrolledRun:
        self.logger.section("[Unrolled Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        entry            = self._load_entry_config()
        training_summary = self._load_training_summary()

        dataset_config = self._build_dataset_config(
            payload     = self._read_json("dataset_creation_config.json"),
            batch_size  = None,
            num_workers = 0,
        )

        model, model_name, model_config, checkpoint_path, checkpoint_meta = self._build_unrolled_model(device, config.checkpoint_name)

        norm_stats = Stats.load(self.meta_directory, self.logger)
        normalizer = Normalizer(norm_stats)

        gt_parameters, region, layout = self._region_ground_truth(dataset_config, config.split, normalizer)
        kz_field                      = self._region_kz(dataset_config, layout, region, entry.geometry.height_axis_convention)

        gaussian = GaussianConfig.from_dataset(dataset_config.preprocessing_run_directory, dataset_config.parameters_path)
        x_axis   = GaussianAxis.build(gaussian.x_min, gaussian.x_max, layout.profile_length)

        noise_std = entry.measurement_noise_std if config.measurement_noise_std is None else float(config.measurement_noise_std)

        self.logger.section(f"[Unrolled Model] : '{model_name}'")
        self.logger.kv_table({
            "Checkpoint"   : checkpoint_path,
            "Best epoch"   : f"{checkpoint_meta['best_epoch'] + 1} (val loss {checkpoint_meta['best_val_loss']:.6f})",
            "Iterations"   : model_config.n_iterations,
            "Parameters"   : sum(parameter.numel() for parameter in model.parameters()),
            "Split"        : config.split,
            "Azimuth size" : region.azimuth_size,
            "Range size"   : region.range_size,
            "Tracks"       : kz_field.shape[0],
            "X-axis bins"  : x_axis.size,
            "Noise std"    : noise_std,
        })

        return UnrolledRun(
            model            = model,
            model_name       = model_name,
            model_config     = model_config,
            gt_parameters    = gt_parameters,
            kz_field         = kz_field,
            x_axis           = x_axis,
            ppg              = gaussian.params_per_gaussian,
            n_gaussians      = dataset_config.n_gaussians,
            split_name       = config.split,
            split_region     = region,
            curve_loss       = entry.curve_loss,
            power_floor      = entry.power_floor,
            noise_std        = noise_std,
            checkpoint_path  = checkpoint_path,
            training_summary = training_summary,
        )
