
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.dataset               import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.dataset.gaussian      import GaussianConfig
from configuration.normalization.general import ChannelStats, ChannelStrategy, NormMethod
from configuration.synthesis             import SynthesisConfig
from pipelines.backbone.dataset.pipeline import DatasetPipeline
from pipelines.backbone.dataset.stats    import Stats
from tools.data.io                       import FileIO
from tools.data.regions                  import CropRegion
from tools.monitoring.logger             import Logger


def _logger(tmp_path, name) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name=name, level="ERROR")


def _activity_threshold(test_data_dir, params_dir) -> float:
    return GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy").activity_threshold


def _backbone_config(test_data_dir, params_dir) -> DatasetConfig:
    ic = InputConfig(
        use_primary=True,        primary_representation=Representation.MAG_ONLY,
        use_secondaries=True,    secondaries_representation=Representation.MAG_ONLY,
        use_derived=True, derived_representation=Representation.ANGLE_ONLY,
    )

    splits = SplitRegions(
        train = CropRegion(1000, 1024, 500, 524),
        val   = CropRegion(1024, 1040, 500, 524),
        test  = CropRegion(1040, 1056, 500, 524),
    )

    return DatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = splits,
        secondary_labels            = ("CH01", "CH02"),
        patch                       = PatchConfig(size=(8, 8), stride=(8, 8)),
        input_config                = ic,
        batch_size                  = 2,
        num_workers                 = 0,
        prefetch_factor             = 8,
        shuffle_train               = True,
        n_gaussians                 = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy").n_default_gaussians,
    )


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_writes_metadata_and_stats(test_data_dir, params_dir, tmp_path):
    config   = _backbone_config(test_data_dir, params_dir)
    pipeline = DatasetPipeline(config, tmp_path, _activity_threshold(test_data_dir, params_dir), logger=_logger(tmp_path, "bb_pipe"), seed=0)

    train_loader, val_loader, test_loader, datasets = pipeline.run()

    meta = tmp_path / "meta"

    assert (meta / "normalization_stats.json").is_file()
    assert (meta / "dataset_creation_config.json").is_file()
    assert (meta / "crop.json").is_file()
    assert (meta / "patch.json").is_file()

    assert pipeline.profile_length == SynthesisConfig().profile_length
    assert datasets["train"].normalizer is not None
    assert len(train_loader) >= 1


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_train_order_reproducible(test_data_dir, params_dir, tmp_path):
    torch.use_deterministic_algorithms(False)

    def _first_batch(run_dir):
        config   = _backbone_config(test_data_dir, params_dir)
        pipeline = DatasetPipeline(config, run_dir, _activity_threshold(test_data_dir, params_dir), logger=_logger(run_dir, "bb_repro"), seed=7)
        loaders  = pipeline.run()
        x, _     = next(iter(loaders[0]))
        return x.numpy()

    a = _first_batch(tmp_path / "a")
    b = _first_batch(tmp_path / "b")

    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_rectangular_patches_yield_rectangular_batches(test_data_dir, params_dir, tmp_path):
    config       = _backbone_config(test_data_dir, params_dir)
    config.patch = PatchConfig(size=(8, 16), stride=(4, 8))

    pipeline = DatasetPipeline(config, tmp_path, _activity_threshold(test_data_dir, params_dir), logger=_logger(tmp_path, "bb_rect"), seed=0)

    train_loader, _val_loader, _test_loader, _datasets = pipeline.run()
    x, y = next(iter(train_loader))

    assert tuple(x.shape[-2:]) == (8, 16)
    assert tuple(y.shape[-2:]) == (8, 16)


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_announces_each_crop_read_by_split_and_region(test_data_dir, params_dir, tmp_path):
    config               = _backbone_config(test_data_dir, params_dir)
    config.split_regions = SplitRegions(
        train = [CropRegion(1000, 1012, 500, 524), CropRegion(1012, 1024, 500, 524)],
        val   = CropRegion(1024, 1040, 500, 524),
        test  = CropRegion(1040, 1056, 500, 524),
    )

    logger = _logger(tmp_path, "bb_crop_log")
    DatasetPipeline(config, tmp_path, _activity_threshold(test_data_dir, params_dir), logger=logger, seed=0).run()
    logger.close()

    text = (tmp_path / "logs" / "bb_crop_log.log").read_text(encoding="utf-8")

    announce = text.index("Loading split 'train' region 2/2: row 1012:1024, col 500:524")
    loaded   = text.index("[CROP LOADED: TRAIN 2/2]")

    assert "Loading split 'train' region 1/2: row 1000:1012, col 500:524" in text
    assert announce < loaded
    assert "Region           : row 1012:1024, col 500:524" in text[loaded:]
    assert "[CROP LOADED: VAL 1/1]" in text
    assert "[CROP LOADED: TEST 1/1]" in text


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_crops_only_the_enabled_input_groups(test_data_dir, params_dir, tmp_path):
    full    = _backbone_config(test_data_dir, params_dir)
    trimmed = _backbone_config(test_data_dir, params_dir)

    trimmed.input_config = InputConfig(
        use_primary=False,
        use_secondaries=False,
        use_derived=True, derived_representation=Representation.ANGLE_ONLY,
    )

    threshold        = _activity_threshold(test_data_dir, params_dir)
    full_datasets    = DatasetPipeline(full,    tmp_path / "full",    threshold, logger=_logger(tmp_path, "bb_full"),    seed=0).run()[3]
    trimmed_datasets = DatasetPipeline(trimmed, tmp_path / "trimmed", threshold, logger=_logger(tmp_path, "bb_trimmed"), seed=0).run()[3]

    full_train    = full_datasets["train"]
    trimmed_train = trimmed_datasets["train"]

    assert (full_train.n_primary, full_train.n_secondaries, full_train.n_derived)          == (1, 2, 2)
    assert (trimmed_train.n_primary, trimmed_train.n_secondaries, trimmed_train.n_derived) == (0, 0, 2)

    assert full_train.inputs.shape[0]    == 5
    assert trimmed_train.inputs.shape[0] == 2
    assert trimmed_train.input_channels  == 2

    np.testing.assert_array_equal(trimmed_train.inputs, full_train.inputs[3:])


def _stats(scale: float) -> Stats:
    channels = ChannelStats(
        loc        = [0.0],
        scale      = [scale],
        names      = ["pass/mag"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE)],
        clampable  = [False],
    )

    return Stats(input_stats=channels, output_stats=channels)


def _resumable_pipeline(run_dir, resume: bool) -> DatasetPipeline:
    pipeline                        = DatasetPipeline.__new__(DatasetPipeline)
    pipeline.training_run_directory = run_dir
    pipeline.resume                 = resume
    pipeline.logger                 = _logger(run_dir, "persist_stats")

    (run_dir / "meta").mkdir(parents=True, exist_ok=True)

    return pipeline


def test_resumed_run_keeps_matching_normalization_stats(tmp_path):
    pipeline = _resumable_pipeline(tmp_path, resume=True)

    _stats(2.0).save(tmp_path / "meta")
    pipeline._persist_stats(_stats(2.0))

    stored = FileIO.load_json(tmp_path / "meta" / Stats.FILENAME)

    assert stored["input_stats"]["channels"][0]["scale"] == 2.0


def test_fresh_run_overwrites_normalization_stats(tmp_path):
    pipeline = _resumable_pipeline(tmp_path, resume=False)

    _stats(2.0).save(tmp_path / "meta")
    pipeline._persist_stats(_stats(5.0))

    stored = FileIO.load_json(tmp_path / "meta" / Stats.FILENAME)

    assert stored["input_stats"]["channels"][0]["scale"] == 5.0


class _CapturingDatasetPipeline:

    captured = {}

    def __init__(self, *args, **kwargs):
        type(self).captured = kwargs
        self.profile_length = 16
        self.layout         = SimpleNamespace(profile_length=16)

    def run(self):
        return None, None, None, {}


def test_backbone_preparation_threads_the_training_resume_flag(tmp_path, monkeypatch):
    import pipelines.backbone.dataset.pipeline as module
    monkeypatch.setattr(module, "DatasetPipeline", _CapturingDatasetPipeline)

    gaussian       = SimpleNamespace(n_default_gaussians=3, x_min=-5.0, x_max=25.0, activity_threshold=2e-3)
    trainer_config = SimpleNamespace(gaussian=gaussian, training=SimpleNamespace(resume=True))
    dataset_config = SimpleNamespace(n_gaussians=None, x_axis=None)

    module.BackboneDatasetPreparation(dataset_config, trainer_config, tmp_path, _logger(tmp_path, "prep_resume"), seed=0).run()

    assert _CapturingDatasetPipeline.captured["resume"]             is True
    assert _CapturingDatasetPipeline.captured["activity_threshold"] == 2e-3
