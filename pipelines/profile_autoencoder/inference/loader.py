from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.inference.profile_autoencoder         import ProfileAeInferenceConfig
from models.profile_autoencoder                          import get_profile_autoencoder
from pipelines.shared.dataset.dataset_spatial            import Layout
from pipelines.profile_autoencoder.dataset.datasets      import ProfileDataset
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from tools.data.gaussians                                import GaussianAxis
from pipelines.profile_autoencoder.dataset.splitting     import ParameterCropper
from pipelines.shared.config.config_persistence          import ProfileAutoencoderConfigIO, ProfileDatasetConfigIO
from tools.data.io                                       import FileIO
from tools.data.regions                                  import CropRegion
from tools.monitoring.logger                             import Logger


@dataclass
class ProfileAeRun:
    model                       : object
    ae_name                     : str
    embedding_dim               : int
    x_axis                      : np.ndarray
    normalizer                  : ProfileNormalizer
    dataset                     : ProfileDataset
    loader                      : DataLoader
    split_name                  : str
    checkpoint_meta             : dict
    preprocessing_run_directory : Path
    split_regions               : list[CropRegion]
    amp_zero_thr                : float


class ProfileAeRunLoader:
    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_run_summary(self) -> dict:
        summary = FileIO.load_json(self.meta_directory / "run_summary.json")

        if str(summary.get("model_name")) != "profile_ae":
            raise ValueError(f"Run '{self.run_directory}' is not a standalone profile-autoencoder run (model_name='{summary.get('model_name')}'). Use 'infer' for backbone and JEPA runs.")

        return summary

    def _build_dataset(self, config: ProfileAeInferenceConfig, dataset_config, normalizer: ProfileNormalizer):
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = ParameterCropper(layout, dataset_config.split_regions, logger=self.logger)

        param_arrays = cropper.load_split(config.split)
        x_len        = cropper.profile_length()
        x_axis       = GaussianAxis.build(dataset_config.x_min, dataset_config.x_max, x_len)

        dataset = ProfileDataset(
            param_arrays    = param_arrays,
            x_axis          = x_axis,
            n_gaussians     = dataset_config.n_gaussians,
            split_name      = config.split,
            amp_zero_thr    = dataset_config.amp_zero_thr,
            pixel_subsample = config.pixel_subsample,
            keep_empty_frac = config.keep_empty_frac,
            seed            = config.seed,
            normalizer      = normalizer,
            augmenter       = None,
            logger          = self.logger,
        )

        return dataset, x_axis, x_len

    def _build_model(self, x_len: int, device: str):
        ae_cfg, ae_name       = ProfileAutoencoderConfigIO.load(self.meta_directory)
        ae_cfg.profile_length = x_len

        model, _ = get_profile_autoencoder(ae_name, ae_cfg)

        return model.to(device), ae_name

    def _load_checkpoint(self, ckpt_path: Path, device: str):
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for inference: {ckpt_path}")

        ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        x_axis = np.asarray(ckpt["x_axis"], dtype=np.float32)

        meta = {
            "epoch"         : int(ckpt["epoch"]),
            "best_val_loss" : float(ckpt["best_val_loss"]),
            "best_epoch"    : int(ckpt["best_epoch"]),
        }

        return ckpt, x_axis, meta

    def load(self, *, config: ProfileAeInferenceConfig, device: str) -> ProfileAeRun:
        self.logger.section("[Profile AE Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        summary        = self._read_run_summary()
        embedding_dim  = int(summary["out_channels"])
        dataset_config = ProfileDatasetConfigIO.load(self.meta_directory)
        normalizer     = ProfileNormalizer(ProfileStats.load(self.meta_directory, self.logger))

        dataset, x_axis, x_len = self._build_dataset(config, dataset_config, normalizer)

        if x_len != int(summary["x_axis_length"]):
            raise ValueError(f"Profile length mismatch: dataset gives {x_len} bins but the run was trained on {summary['x_axis_length']}. The persisted dataset no longer matches the training dataset.")

        model, ae_name = self._build_model(x_len, device)
        ckpt_path      = self.run_directory / config.checkpoint_name

        ckpt, ckpt_x_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)

        model.load_state_dict(ckpt["params"])
        model.eval()

        if ckpt_x_axis.shape[0] != x_axis.shape[0]:
            raise ValueError(f"Checkpoint x-axis length {ckpt_x_axis.shape[0]} does not match dataset profile length {x_axis.shape[0]}.")

        batch_size = config.batch_size if config.batch_size is not None else len(dataset)
        loader     = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        regions = dataset_config.split_regions.regions(config.split)

        self.logger.section(f"[Profile AE Model]  : '{ae_name}'")
        self.logger.kv_table({
            "Checkpoint"     : ckpt_path,
            "Embedding dim"  : embedding_dim,
            "Profile length" : x_len,
            "Curves"         : len(dataset),
            "Split"          : config.split,
        })

        return ProfileAeRun(
            model                       = model,
            ae_name                     = ae_name,
            embedding_dim               = embedding_dim,
            x_axis                      = x_axis,
            normalizer                  = normalizer,
            dataset                     = dataset,
            loader                      = loader,
            split_name                  = config.split,
            checkpoint_meta             = ckpt_meta,
            preprocessing_run_directory = Path(dataset_config.preprocessing_run_directory),
            split_regions               = regions,
            amp_zero_thr                = dataset_config.amp_zero_thr,
        )
