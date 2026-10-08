"""Tests for HeadPriorBias, the training-label prior written into the conv head bias.

Covers the active-pixel means in sorted_gt slot order, their independence from the
region split and the row chunking, the loud failures on a never-active slot, a
non-finite label and a head without a single output convolution, and the guarantee
that only the conv head bias changes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models                               import get_backbone
from pipelines.backbone.dataset.datasets  import MultiRegionDataset
from pipelines.backbone.training.pipeline import HeadPriorBias

from tests.conftest import SilentLogger


pytestmark = pytest.mark.usefixtures("force_cpu")


THRESHOLD = 1e-3


class _AffineNormalizer:
    """Normalizer stub applying a per-channel affine map, (x - loc) / scale.

    Attributes:
        loc   : Per-channel location of shape (channels,).
        scale : Per-channel scale of shape (channels,).
    """

    def __init__(self, loc, scale) -> None:
        """Stores the per-channel location and scale."""
        self.loc   = np.asarray(loc,   dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    def normalize_output(self, tensor: np.ndarray) -> np.ndarray:
        """Returns the labels of shape (channels, h, w) in normalized units."""
        return (tensor - self.loc[:, None, None]) / self.scale[:, None, None]


class _Region:
    """Training-region stub carrying the attributes HeadPriorBias and MultiRegionDataset read.

    Attributes:
        gt_selected : Physical labels of shape (3 * n_gaussians, rows, cols).
        normalizer  : Normalizer shared by the split.
        gt_channels : Number of label channels.
        n_gaussians : Number of Gaussian slots.
        split_name  : Split name, always "train".
    """

    def __init__(self, labels: np.ndarray, normalizer) -> None:
        """Stores the labels and the normalizer, and fills the channel-structure fields MultiRegionDataset compares."""
        self.gt_selected            = labels
        self.normalizer             = normalizer
        self.gt_channels            = labels.shape[0]
        self.n_gaussians            = labels.shape[0] // 3
        self.split_name             = "train"
        self.input_channels         = 1
        self.n_primary              = 0
        self.n_secondaries          = 1
        self.n_derived              = 0
        self.input_layers           = 1
        self.input_config           = None
        self.output_config          = None
        self.output_channel_indices = list(range(labels.shape[0]))

    def __len__(self) -> int:
        """Returns one patch, enough for MultiRegionDataset's index."""
        return 1


def _pixel_labels() -> np.ndarray:
    """Returns two-slot labels of shape (6, 2, 2) covering every sorting case.

    Pixel (0, 0) is stored lowest first, (0, 1) stores the higher component in
    slot 0, (1, 0) has an inactive slot 0 and an active slot 1, and (1, 1) has
    both slots inactive.
    """
    pixels = {
        (0, 0) : [(1.0,  0.0, 1.0), (2.0, 10.0, 3.0)],
        (0, 1) : [(3.0, 12.0, 5.0), (0.5, -2.0, 0.5)],
        (1, 0) : [(0.0, 99.0, 9.0), (1.5,  4.0, 2.0)],
        (1, 1) : [(1e-4, 7.0, 1.0), (5e-4, 8.0, 1.0)],
    }

    labels = np.zeros((6, 2, 2), dtype=np.float32)
    for (row, col), slots in pixels.items():
        labels[:, row, col] = np.concatenate(slots)

    return labels


def _normalizer() -> _AffineNormalizer:
    """Returns a normalizer whose six channels all have distinct location and scale."""
    return _AffineNormalizer(loc=[0.1, -5.0, 0.2, 0.3, 1.0, 0.4], scale=[2.0, 10.0, 4.0, 3.0, 20.0, 5.0])


def test_prior_is_the_active_pixel_mean_in_sorted_slot_order():
    """Each channel's prior is the mean of its normalized label over the pixels where its sorted slot is active."""
    normalizer = _normalizer()
    prior, counts, n_pixels = HeadPriorBias(_Region(_pixel_labels(), normalizer), 2, THRESHOLD, SilentLogger()).prior()

    slot_0 = np.array([(1.0, 0.0, 1.0), (0.5, -2.0, 0.5), (1.5, 4.0, 2.0)])
    slot_1 = np.array([(2.0, 10.0, 3.0), (3.0, 12.0, 5.0)])
    loc    = normalizer.loc.astype(np.float64)
    scale  = normalizer.scale.astype(np.float64)

    expected = np.concatenate([(slot_0.mean(axis=0) - loc[:3]) / scale[:3], (slot_1.mean(axis=0) - loc[3:]) / scale[3:]])

    np.testing.assert_allclose(prior, expected, rtol=1e-6)
    assert counts.tolist() == [3, 2]
    assert n_pixels == 4


def test_prior_ignores_the_region_split_and_the_row_chunking(monkeypatch):
    """Two training regions give the same prior as one region holding all their pixels, at any row chunk size."""
    normalizer = _normalizer()
    rng        = np.random.default_rng(0)
    labels     = rng.uniform(0.0, 2.0, size=(6, 7, 5)).astype(np.float32)

    labels[0::3][:, rng.uniform(size=(7, 5)) < 0.3] = 0.0

    whole = HeadPriorBias(_Region(labels, normalizer), 2, THRESHOLD, SilentLogger()).prior()

    monkeypatch.setattr(HeadPriorBias, "ROW_CHUNK", 2)
    split = MultiRegionDataset([_Region(labels[:, :3], normalizer), _Region(labels[:, 3:], normalizer)])
    parts = HeadPriorBias(split, 2, THRESHOLD, SilentLogger()).prior()

    np.testing.assert_allclose(parts[0], whole[0], rtol=1e-6)
    assert parts[1].tolist() == whole[1].tolist()
    assert parts[2] == whole[2] == 35


def test_prior_rejects_a_slot_that_is_never_active():
    """A slot without a single active training pixel has no prior, so the computation raises."""
    labels    = _pixel_labels()
    labels[0] = 1.0
    labels[3] = 0.0

    with pytest.raises(ValueError, match=r"Slots \[1\] are active on no training pixel"):
        HeadPriorBias(_Region(labels, _normalizer()), 2, THRESHOLD, SilentLogger()).prior()


def test_prior_rejects_a_non_finite_label_on_an_active_pixel():
    """A NaN width on an active pixel makes the prior non-finite, which raises and names the channel."""
    labels          = _pixel_labels()
    labels[2, 0, 0] = np.nan

    with pytest.raises(ValueError, match="slot_0_sigma"):
        HeadPriorBias(_Region(labels, _normalizer()), 2, THRESHOLD, SilentLogger()).prior()


def test_apply_overwrites_only_the_conv_head_bias():
    """Applying the prior sets the conv head bias to it and leaves every other parameter and buffer as built."""
    torch.manual_seed(0)
    model, _ = get_backbone("unet", head="conv", in_channels=2, out_channels=6, features=[8, 16])
    before   = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    prior_bias  = HeadPriorBias(_Region(_pixel_labels(), _normalizer()), 2, THRESHOLD, SilentLogger())
    record      = prior_bias.apply(model)
    prior, _, _ = prior_bias.prior()

    after = model.state_dict()
    torch.testing.assert_close(after["output_head.bias"], torch.as_tensor(prior, dtype=torch.float32))

    for name, tensor in before.items():
        if name != "output_head.bias":
            assert torch.equal(after[name], tensor), name

    np.testing.assert_allclose(record["default_bias"], before["output_head.bias"].numpy().astype(np.float64))
    np.testing.assert_allclose(record["prior_bias"], prior)
    assert record["channels"]           == ["slot_0_amplitude", "slot_0_mu", "slot_0_sigma", "slot_1_amplitude", "slot_1_mu", "slot_1_sigma"]
    assert record["active_pixels"]      == [3, 2]
    assert record["training_pixels"]    == 4
    assert record["activity_threshold"] == THRESHOLD


def test_apply_rejects_a_head_without_a_single_output_convolution():
    """The set-prediction head has no single output bias, so applying the prior raises before touching the model."""
    model, _ = get_backbone("unet", head="set_pred", in_channels=2, out_channels=6, features=[8, 16])

    with pytest.raises(ValueError, match="'set_pred' head"):
        HeadPriorBias(_Region(_pixel_labels(), _normalizer()), 2, THRESHOLD, SilentLogger()).apply(model)
