from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Tuple

from torch.utils.data import DataLoader

from configuration.inference.image_autoencoder  import ImageAeInferenceConfig
from models.image_autoencoder                   import get_image_autoencoder
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.stats           import Stats
from pipelines.shared.config.config_persistence import ImageAutoencoderConfigIO
from pipelines.shared.inference.run_artifacts   import RunArtifactLoader
from tools.data.regions                         import CropRegion


@dataclass
class ImageAeRun:
    model                       : object
    ae_name                     : str
    embedding_dim               : int
    in_channels                 : int
    normalizer                  : Normalizer
    loader                      : DataLoader
    split_name                  : str
    patch_size                  : Tuple[int, int]
    checkpoint_meta             : dict
    preprocessing_run_directory : Path
    split_region                : CropRegion


class ImageAeRunLoader(RunArtifactLoader):
    def _read_run_summary(self) -> dict:
        summary = self._read_json("run_summary.json")

        if str(summary.get("model_name")) != "image_ae":
            raise ValueError(f"Run '{self.run_directory}' is not a standalone image-autoencoder run (model_name='{summary.get('model_name')}'). Use 'infer' for backbone and JEPA runs.")

        return summary

    def _build_model(self, device: str):
        ae_cfg, ae_name = ImageAutoencoderConfigIO.load(self.meta_directory)
        model, _        = get_image_autoencoder(ae_name, ae_cfg)

        return model.to(device), ae_name, ae_cfg

    def load(self, *, config: ImageAeInferenceConfig, device: str) -> ImageAeRun:
        self.logger.section("[Image AE Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        summary       = self._read_run_summary()
        embedding_dim = int(summary["out_channels"])

        dataset_config = self._build_dataset_config(
            payload     = self._read_json("dataset_creation_config.json"),
            batch_size  = config.batch_size,
            num_workers = config.num_workers,
        )

        norm_stats             = Stats.load(self.meta_directory, self.logger)
        model, ae_name, ae_cfg = self._build_model(device)

        ckpt_path               = self.run_directory / config.checkpoint_name
        ckpt, x_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)

        model.load_state_dict(ckpt["params"])
        model.eval()

        dataset, grid, region, _global_crop, _arrays = self._build_dataset(
            dataset_config = dataset_config,
            split_name     = config.split,
            x_axis         = x_axis,
            n_gaussians    = dataset_config.n_gaussians,
            norm_stats     = norm_stats,
            load_tomogram  = False,
        )

        if dataset.input_channels != int(summary["in_channels"]):
            raise ValueError(f"Input-channel mismatch: dataset yields {dataset.input_channels} channels but the run was trained on {summary['in_channels']}. The persisted dataset no longer matches the training dataset.")

        loader = DataLoader(
            dataset,
            batch_size  = dataset_config.batch_size,
            shuffle     = False,
            num_workers = dataset_config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        self.logger.section(f"[Image AE Model]  : '{ae_name}'")
        self.logger.kv_table({
            "Checkpoint"    : ckpt_path,
            "Embedding dim" : embedding_dim,
            "In channels"   : dataset.input_channels,
            "Patch size"    : tuple(dataset_config.patch.size),
            "Patches"       : grid.grid.number_of_patches,
            "Split"         : config.split,
        })

        return ImageAeRun(
            model                       = model,
            ae_name                     = ae_name,
            embedding_dim               = embedding_dim,
            in_channels                 = dataset.input_channels,
            normalizer                  = Normalizer(norm_stats),
            loader                      = loader,
            split_name                  = config.split,
            patch_size                  = tuple(dataset_config.patch.size),
            checkpoint_meta             = ckpt_meta,
            preprocessing_run_directory = Path(dataset_config.preprocessing_run_directory),
            split_region                = region,
        )
