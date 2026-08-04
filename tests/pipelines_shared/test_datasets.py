from __future__ import annotations

import numpy as np
import pytest

from configuration.dataset                          import InputConfig, OutputConfig, Representation
from configuration.normalization                    import ChannelStats, ChannelStrategy, NormMethod
from pipelines.backbone.dataset.datasets            import MultiRegionDataset, PatchDataset
from pipelines.backbone.dataset.normalizer          import Normalizer
from pipelines.backbone.dataset.spatial             import Patcher
from pipelines.backbone.dataset.stats               import Stats
from pipelines.profile_autoencoder.dataset.datasets import ProfileDataset
from tools.monitoring.logger                        import Logger


def _identity_channel_stats(n: int) -> ChannelStats:
    strategy = ChannelStrategy(NormMethod.ZSCORE)
    return ChannelStats(loc=[0.0] * n, scale=[1.0] * n, names=[f"c{i}" for i in range(n)], strategies=[strategy] * n, clampable=[False] * n)


def _identity_normalizer(n_inputs: int, n_outputs: int) -> Normalizer:
    return Normalizer(Stats(input_stats=_identity_channel_stats(n_inputs), output_stats=_identity_channel_stats(n_outputs)))


def _backbone_dataset(derived, parameters, split_name="train", n_derived=3):
    derived = np.ascontiguousarray(np.asarray(derived[:n_derived, :24, :24]))
    primary = derived[:1]
    inputs  = np.concatenate([primary, derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :24, :24]))

    patcher = Patcher.build(spatial_size=(24, 24), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=False,
                          use_derived=True, derived_representation=Representation.ANGLE_ONLY)

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=OutputConfig(), split_name=split_name,
        n_secondaries=0, n_derived=n_derived, n_gaussians=params.shape[0] // 3,
    )
    ds.normalizer = _identity_normalizer(ds.input_channels, ds.gt_channels)

    return ds, patcher


@pytest.mark.dataset
def test_patchdataset_len_matches_grid(derived, parameters):
    ds, patcher = _backbone_dataset(derived, parameters)

    assert len(ds) == patcher.grid.number_of_patches
    assert len(ds) == 9


@pytest.mark.dataset
def test_patchdataset_sample_shapes_and_dtypes(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    x, y = ds[0]

    assert x.shape == (4, 8, 8)
    assert y.shape == (parameters.shape[0], 8, 8)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


@pytest.mark.dataset
def test_patchdataset_channel_counts(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    assert ds.input_channels == 4
    assert ds.gt_channels    == parameters.shape[0]
    assert ds.output_channel_indices == list(range(parameters.shape[0]))


@pytest.mark.dataset
def test_patchdataset_no_nan_in_any_sample(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    for idx in range(len(ds)):
        x, y = ds[idx]
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))


@pytest.mark.dataset
def test_patchdataset_angle_channels_in_pi_range(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    x, _ = ds[0]

    assert np.all(np.abs(x) <= np.pi + 1e-4)


@pytest.mark.dataset
def test_patchdataset_layer_mismatch_raises(derived, parameters):
    derived = np.ascontiguousarray(np.asarray(derived[:3, :16, :16]))
    inputs  = np.concatenate([derived[:1], derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :16, :16]))
    patcher = Patcher.build(spatial_size=(16, 16), patch_size=(8, 8), stride=(8, 8))

    with pytest.raises(ValueError):
        PatchDataset(
            inputs=inputs, gt_parameters=params, grid=patcher,
            input_config=InputConfig(), output_config=OutputConfig(), split_name="train",
            n_secondaries=0, n_derived=99, n_gaussians=params.shape[0] // 3,
        )


@pytest.mark.dataset
def test_patchdataset_use_dem_without_dem_raises(derived, parameters):
    derived = np.ascontiguousarray(np.asarray(derived[:3, :16, :16]))
    inputs  = np.concatenate([derived[:1], derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :16, :16]))
    patcher = Patcher.build(spatial_size=(16, 16), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_aux=True)

    with pytest.raises(ValueError, match="Aux"):
        PatchDataset(
            inputs=inputs, gt_parameters=params, grid=patcher,
            input_config=ic, output_config=OutputConfig(), split_name="train",
            n_secondaries=0, n_derived=3, n_gaussians=params.shape[0] // 3,
        )


@pytest.mark.dataset
def test_patchdataset_dem_fills_last_channel(derived, parameters):
    derived = np.ascontiguousarray(np.asarray(derived[:3, :24, :24]))
    inputs  = np.concatenate([derived[:1], derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :24, :24]))
    aux     = np.random.default_rng(1).standard_normal((24, 24)).astype(np.float32)
    patcher = Patcher.build(spatial_size=(24, 24), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=False,
                          use_derived=True, derived_representation=Representation.ANGLE_ONLY,
                          use_aux=True)

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=OutputConfig(), split_name="val",
        n_secondaries=0, n_derived=3, n_gaussians=params.shape[0] // 3, aux=aux,
    )
    ds.normalizer = _identity_normalizer(ds.input_channels, ds.gt_channels)

    x, _ = ds[0]

    assert x.shape == (5, 8, 8)
    np.testing.assert_array_equal(x[-1], patcher.extract(aux, 0))


@pytest.mark.dataset
def test_multiregion_dataset_concatenates_parts(derived, parameters):
    part_a, _ = _backbone_dataset(derived, parameters)
    part_b, _ = _backbone_dataset(derived, parameters)

    multi = MultiRegionDataset([part_a, part_b])

    assert len(multi) == len(part_a) + len(part_b)

    first = multi[0]
    last  = multi[len(multi) - 1]

    assert first[0].shape == (4, 8, 8)
    assert last[0].shape  == (4, 8, 8)


@pytest.mark.dataset
def test_multiregion_index_out_of_range_raises(derived, parameters):
    part, _ = _backbone_dataset(derived, parameters)
    multi   = MultiRegionDataset([part])

    with pytest.raises(IndexError):
        multi[len(multi)]


def _profile_dataset(parameters, tmp_path, **kw):
    params = [np.ascontiguousarray(np.asarray(parameters[:, :40, :40]))]
    z_axis = np.linspace(0.0, 1.0, 150, dtype=np.float32)
    logger = Logger(log_dir=str(tmp_path / "logs"), name="prof_ds", level="ERROR")

    defaults = dict(n_gaussians=params[0].shape[0] // 3, split_name="train", amp_zero_thr=1e-3, keep_empty_frac=0.05, pixel_subsample=1.0, seed=0)
    defaults.update(kw)

    return ProfileDataset(param_arrays=params, z_axis=z_axis, logger=logger, **defaults)


@pytest.mark.dataset
def test_profiledataset_sample_shape_and_dtype(parameters, tmp_path):
    ds    = _profile_dataset(parameters, tmp_path)
    curve = ds[0]

    assert curve.shape == (150,)
    assert curve.dtype == np.float32


@pytest.mark.dataset
def test_profiledataset_no_nan_and_nonnegative(parameters, tmp_path):
    ds = _profile_dataset(parameters, tmp_path)

    for i in range(min(len(ds), 50)):
        curve = ds[i]
        assert np.all(np.isfinite(curve))
        assert np.all(curve >= 0.0)


@pytest.mark.dataset
def test_profiledataset_len_consistent_with_index(parameters, tmp_path):
    ds = _profile_dataset(parameters, tmp_path)

    assert len(ds) == ds.index.shape[0]
    assert len(ds) <= ds.amps.shape[0]
    assert ds.amps.shape[0] == 40 * 40


@pytest.mark.dataset
def test_profiledataset_full_subsample_keeps_all_active(parameters, tmp_path):
    ds = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=1.0)

    assert len(ds) == ds.n_active


@pytest.mark.dataset
def test_profiledataset_subsample_reduces_active(parameters, tmp_path):
    full = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=1.0)
    half = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=0.5)

    assert len(half) < len(full)


@pytest.mark.dataset
def test_profiledataset_index_deterministic_for_seed(parameters, tmp_path):
    a = _profile_dataset(parameters, tmp_path, seed=11)
    b = _profile_dataset(parameters, tmp_path, seed=11)

    np.testing.assert_array_equal(a.index, b.index)


@pytest.mark.dataset
def test_profiledataset_index_changes_with_seed(parameters, tmp_path):
    a = _profile_dataset(parameters, tmp_path, seed=1, pixel_subsample=0.5, keep_empty_frac=0.0)
    b = _profile_dataset(parameters, tmp_path, seed=2, pixel_subsample=0.5, keep_empty_frac=0.0)

    assert not np.array_equal(a.index, b.index)


@pytest.mark.parametrize("pixel_subsample", [0.5, 1.0])
def test_profiledataset_without_active_pixels_names_the_split_and_threshold(tmp_path, pixel_subsample):
    empty = np.zeros((15, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="amp_zero_thr"):
        _profile_dataset(empty, tmp_path, pixel_subsample=pixel_subsample)
