"""Loading of a trained image-autoencoder run into an inference-ready bundle.

Rebuilds the autoencoder from its persisted configuration, restores the
checkpoint weights, and rebuilds the patch dataset and loader of the requested
split from the run's own dataset-creation configuration.
"""

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
    """Everything inference needs from a trained image-autoencoder run.

    Attributes:
        model                       : Restored autoencoder in eval mode, on the inference device.
        ae_name                     : Registered image-autoencoder architecture name.
        embedding_dim               : Width of the latent embedding.
        in_channels                 : Width of the input patch stack the dataset yields.
        normalizer                  : Normalizer holding the training input statistics.
        loader                      : Unshuffled loader over the requested split's patches.
        split_name                  : Name of the split being inferred.
        patch_size                  : Patch extent as (rows, cols) in pixels.
        checkpoint_meta             : Checkpoint metadata, including best epoch and loss.
        preprocessing_run_directory : Directory of the preprocessing run the dataset was built from.
        split_region                : Crop region of the split in the preprocessed scene.
    """

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
    """Restores a standalone image-autoencoder run from its artifact directory."""

    def _read_run_summary(self) -> dict:
        """Returns the run summary, rejecting runs of any other model family.

        Raises:
            ValueError : If the run summary does not name the `image_ae` model.
        """
        summary = self._read_json("run_summary.json")

        if str(summary.get("model_name")) != "image_ae":
            raise ValueError(f"Run '{self.run_directory}' is not a standalone image-autoencoder run (model_name='{summary.get('model_name')}'). Use 'infer' for backbone and JEPA runs.")

        return summary

    def _build_model(self, device: str):
        """Builds the autoencoder from the persisted architecture configuration.

        Args:
            device : Torch device string the model is moved to.

        Returns:
            The model on the device, its registered architecture name and its
            resolved configuration.
        """
        ae_cfg, ae_name = ImageAutoencoderConfigIO.load(self.meta_directory)
        model, _        = get_image_autoencoder(ae_name, ae_cfg)

        return model.to(device), ae_name, ae_cfg

    def load(self, *, config: ImageAeInferenceConfig, device: str) -> ImageAeRun:
        """Restores model, weights, normalization statistics and split loader of the run.

        Args:
            config : Inference configuration naming the checkpoint, split, batch size and worker count.
            device : Torch device string the model is loaded onto.

        Returns:
            The assembled `ImageAeRun` bundle.

        Raises:
            ValueError : If the run is not an image-autoencoder run, or the rebuilt dataset yields a different input width than the run was trained on.
        """
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

        ckpt_path                = self.run_directory / config.checkpoint_name
        ckpt, _x_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)

        model.load_state_dict(ckpt["params"])
        model.eval()

        dataset, grid, region, _global_crop, _arrays = self._build_dataset(
            dataset_config = dataset_config,
            split_name     = config.split,
            n_gaussians    = dataset_config.n_gaussians,
            norm_stats     = norm_stats,
            load_cube      = False,
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
