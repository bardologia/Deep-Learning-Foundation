from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configuration.training.general.run         import TrainingQueueConfig
from configuration.dataset.profile_autoencoder  import ProfileAugmentationConfig, ProfileDatasetConfig
from pipelines.shared.config.config_persistence import ProfileDatasetConfigIO
from tools.data.regions                         import CropRegion, SplitRegions
from tools.monitoring.logger                    import Logger


class ProfileDatasetConfigBackfiller:
    def __init__(self, run_directory, parameters_path=None, preprocessing_directory=None, train_azimuth=None, val_azimuth=None, test_azimuth=None) -> None:
        self.run_directory  = Path(run_directory)
        self.meta_directory = self.run_directory / "meta"
        self.docs_directory = self.run_directory / "docs"

        self.parameters_override    = Path(parameters_path) if parameters_path else None
        self.preprocessing_override = Path(preprocessing_directory) if preprocessing_directory else None

        defaults = TrainingQueueConfig()
        self.train_azimuth = tuple(train_azimuth) if train_azimuth else defaults.train_azimuth
        self.val_azimuth   = tuple(val_azimuth)   if val_azimuth   else defaults.val_azimuth
        self.test_azimuth  = tuple(test_azimuth)  if test_azimuth  else defaults.test_azimuth

        self.logger = Logger(log_dir="logs", name="backfill_profile_dataset_config")

    def _load_metadata(self):
        summary = json.loads((self.meta_directory / "run_summary.json").read_text(encoding="utf-8"))
        if str(summary.get("model_name")) != "profile_ae":
            raise ValueError(f"Run '{self.run_directory}' is not a profile-autoencoder run (model_name='{summary.get('model_name')}').")

        trainer = json.loads((self.docs_directory / "trainer_config.json").read_text(encoding="utf-8"))

        return summary, trainer

    def _recover_preprocessing_dir(self, trainer: dict) -> Path:
        if self.preprocessing_override:
            return self.preprocessing_override

        origin = trainer.get("geometry", {}).get("baselines_origin")
        if not origin or origin == "config":
            raise ValueError("Could not recover the preprocessing directory from geometry.baselines_origin; pass --preprocessing-dir.")

        return Path(origin).parent.parent

    def _discover_parameters_path(self, preprocessing_dir: Path) -> Path:
        if self.parameters_override:
            if not self.parameters_override.is_file():
                raise FileNotFoundError(f"--parameters-path does not exist: {self.parameters_override}")
            return self.parameters_override

        params_root = preprocessing_dir / "params"
        candidates  = sorted(params_root.glob("*/parameters*.npy"))

        if not candidates:
            raise FileNotFoundError(f"No parameters npy found under {params_root}; pass --parameters-path.")
        if len(candidates) > 1:
            listing = ", ".join(str(c) for c in candidates)
            raise ValueError(f"Multiple parameter sets under {params_root}: {listing}. Disambiguate with --parameters-path.")

        return candidates[0]

    def _build_split_regions(self, preprocessing_dir: Path) -> SplitRegions:
        layout = json.loads((preprocessing_dir / "data" / "dataset.json").read_text(encoding="utf-8"))
        crop   = layout["global_crop"]

        az_start, az_end, rg_start, rg_end = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])

        def region(azimuth) -> CropRegion:
            a0, a1 = int(azimuth[0]), int(azimuth[1])
            if a0 < az_start or a1 > az_end:
                raise ValueError(f"Azimuth split {azimuth} lies outside the dataset crop azimuth [{az_start}, {az_end}]; pass --train/--val/--test-azimuth matching the training run.")
            return CropRegion(a0, a1, rg_start, rg_end)

        return SplitRegions(train=region(self.train_azimuth), val=region(self.val_azimuth), test=region(self.test_azimuth))

    def _build_config(self, trainer: dict, preprocessing_dir: Path, parameters_path: Path, split_regions: SplitRegions) -> ProfileDatasetConfig:
        gaussian = trainer["gaussian"]

        return ProfileDatasetConfig(
            preprocessing_run_directory = preprocessing_dir,
            split_regions               = split_regions,
            parameters_path             = parameters_path,
            n_gaussians                 = int(gaussian["n_default_gaussians"]),
            x_min                       = float(gaussian["x_min"]),
            x_max                       = float(gaussian["x_max"]),
            augmentation                = ProfileAugmentationConfig(),
        )

    def run(self) -> None:
        if ProfileDatasetConfigIO.exists(self.meta_directory):
            self.logger.ok(f"{self.meta_directory / ProfileDatasetConfigIO.FILENAME} already present; nothing to do.")
            return

        _summary, trainer = self._load_metadata()

        preprocessing_dir = self._recover_preprocessing_dir(trainer)
        parameters_path   = self._discover_parameters_path(preprocessing_dir)
        split_regions     = self._build_split_regions(preprocessing_dir)
        config            = self._build_config(trainer, preprocessing_dir, parameters_path, split_regions)

        out_path = ProfileDatasetConfigIO.save(config, self.meta_directory)

        self.logger.info(f"preprocessing : {preprocessing_dir}")
        self.logger.info(f"parameters    : {parameters_path}")
        self.logger.info(f"gaussians     : Ng={config.n_gaussians}  x=[{config.x_min}, {config.x_max}]")
        self.logger.info(f"test split    : {config.split_regions.test.as_tuple()}  (override with --test-azimuth if the run used a custom split)")
        self.logger.ok(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate meta/profile_dataset_config.json for a profile-autoencoder run trained before dataset-config persistence, so it becomes self-describing for inference.")
    parser.add_argument("run_directory", type=str)
    parser.add_argument("--parameters-path",   type=str, default=None)
    parser.add_argument("--preprocessing-dir", type=str, default=None)
    parser.add_argument("--train-azimuth", type=int, nargs=2, default=None)
    parser.add_argument("--val-azimuth",   type=int, nargs=2, default=None)
    parser.add_argument("--test-azimuth",  type=int, nargs=2, default=None)
    args = parser.parse_args()

    ProfileDatasetConfigBackfiller(
        run_directory           = args.run_directory,
        parameters_path         = args.parameters_path,
        preprocessing_directory = args.preprocessing_dir,
        train_azimuth           = args.train_azimuth,
        val_azimuth             = args.val_azimuth,
        test_azimuth            = args.test_azimuth,
    ).run()


if __name__ == "__main__":
    main()
