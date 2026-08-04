from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.training.diagnostics import ExampleSelector, ParamSampler, ReconstructionFigures, SlotVitals
from tests.backbone_training._helpers        import build_loss, valid_param_tensor


class FigureTracker:
    def __init__(self):
        self.figures    = []
        self.histograms = []

    def log_figure(self, tag, figure, step=None):
        self.figures.append((tag, figure, step))

    def log_histogram(self, tag, values, step=None):
        self.histograms.append((tag, values, step))


def _phys_params(batch=2, n_gaussians=2, hw=5, seed=0):
    return valid_param_tensor(batch, n_gaussians, hw, hw, seed=seed)


def test_sampler_inactive_until_begin():
    sampler = ParamSampler(params_per_gaussian=3)

    assert sampler.active is False

    sampler.begin(1e-3)
    assert sampler.active is True

    sampler.end()
    assert sampler.active is False


def test_sampler_collects_active_values_per_param():
    sampler = ParamSampler(params_per_gaussian=3)
    sampler.begin(1e-3)
    sampler.observe(_phys_params())

    hists = sampler.histograms()

    assert set(hists.keys()) == {"amp", "mu_m", "sigma_m"}
    assert all(isinstance(v, np.ndarray) for v in hists.values())
    assert all(v.size == 2 * 2 * 5 * 5 for v in hists.values())


def test_sampler_masks_inactive_slots():
    sampler = ParamSampler(params_per_gaussian=3)
    params  = _phys_params()

    params[:, 0:3] = 0.0

    sampler.begin(1e-3)
    sampler.observe(params)

    hists = sampler.histograms()

    assert all(v.size == 2 * 1 * 5 * 5 for v in hists.values())


def test_sampler_empty_when_all_slots_inactive():
    sampler = ParamSampler(params_per_gaussian=3)

    sampler.begin(1e-3)
    sampler.observe(torch.zeros(2, 6, 5, 5))

    assert sampler.histograms() == {}


def test_sampler_subsamples_to_batch_cap():
    sampler = ParamSampler(params_per_gaussian=3)
    sampler.begin(1e-3)
    sampler.observe(_phys_params(batch=4, n_gaussians=2, hw=32))

    hists = sampler.histograms()

    assert all(v.size == ParamSampler.MAX_VALUES_PER_BATCH for v in hists.values())


def test_sampler_stops_at_total_cap():
    sampler = ParamSampler(params_per_gaussian=3)
    sampler.begin(1e-3)

    n_batches = ParamSampler.MAX_VALUES_TOTAL // ParamSampler.MAX_VALUES_PER_BATCH + 2
    for i in range(n_batches):
        sampler.observe(_phys_params(batch=4, n_gaussians=2, hw=32, seed=i))

    hists = sampler.histograms()

    assert all(v.size <= ParamSampler.MAX_VALUES_TOTAL for v in hists.values())


def test_sampler_end_clears_store():
    sampler = ParamSampler(params_per_gaussian=3)
    sampler.begin(1e-3)
    sampler.observe(_phys_params())
    sampler.end()

    assert sampler.histograms() == {}


def test_loss_feeds_active_sampler():
    sampler = ParamSampler(params_per_gaussian=3)
    loss    = build_loss(n_gaussians=2, sampler=sampler)
    pred    = valid_param_tensor(2, 2, 5, 5, seed=1)
    gt      = valid_param_tensor(2, 2, 5, 5, seed=2)

    loss(pred, gt)
    assert sampler.histograms() == {}

    sampler.begin(1e-3)
    loss(pred, gt)
    assert set(sampler.histograms().keys()) == {"amp", "mu_m", "sigma_m"}


def _set_gaussian(tgt, sample, h, w, slot, amp, mu, sigma):
    tgt[sample, 3 * slot + 0, h, w] = amp
    tgt[sample, 3 * slot + 1, h, w] = mu
    tgt[sample, 3 * slot + 2, h, w] = sigma


def _category_loader(hw=8):
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(4, 2, hw, hw, generator=gen)
    tgt  = torch.zeros(4, 6, hw, hw)

    _set_gaussian(tgt, 0, 1, 1, 0, 1.0,  30.0, 5.0)
    _set_gaussian(tgt, 0, 2, 2, 0, 5.0,  30.0, 5.0)
    _set_gaussian(tgt, 0, 3, 3, 0, 10.0, 30.0, 5.0)

    _set_gaussian(tgt, 1, 4, 4, 0, 3.0, 20.0, 4.0)
    _set_gaussian(tgt, 1, 4, 4, 1, 3.0, 26.0, 4.0)
    _set_gaussian(tgt, 1, 5, 5, 0, 3.0, 20.0, 4.0)
    _set_gaussian(tgt, 1, 5, 5, 1, 3.0, 28.0, 4.0)
    _set_gaussian(tgt, 1, 6, 6, 0, 3.0, 18.0, 4.0)
    _set_gaussian(tgt, 1, 6, 6, 1, 3.0, 29.0, 4.0)

    _set_gaussian(tgt, 2, 1, 1, 0, 2.0, 0.0,  3.0)
    _set_gaussian(tgt, 2, 1, 1, 1, 5.0, 60.0, 3.0)
    _set_gaussian(tgt, 2, 2, 2, 0, 2.0, 0.0,  3.0)
    _set_gaussian(tgt, 2, 2, 2, 1, 8.0, 60.0, 3.0)
    _set_gaussian(tgt, 2, 3, 3, 0, 1.0, 0.0,  3.0)
    _set_gaussian(tgt, 2, 3, 3, 1, 9.0, 60.0, 3.0)

    _set_gaussian(tgt, 3, 1, 1, 0, 3.0, 10.0, 3.0)
    _set_gaussian(tgt, 3, 1, 1, 1, 3.0, 22.0, 3.0)
    _set_gaussian(tgt, 3, 2, 2, 0, 3.0, 10.0, 3.0)
    _set_gaussian(tgt, 3, 2, 2, 1, 3.0, 34.0, 3.0)
    _set_gaussian(tgt, 3, 3, 3, 0, 3.0, 5.0,  3.0)
    _set_gaussian(tgt, 3, 3, 3, 1, 3.0, 53.0, 3.0)

    return [(imgs[:2], tgt[:2]), (imgs[2:], tgt[2:])]


def _single_only_loader(hw=8):
    gen  = torch.Generator().manual_seed(1)
    imgs = torch.randn(1, 2, hw, hw, generator=gen)
    tgt  = torch.zeros(1, 6, hw, hw)

    _set_gaussian(tgt, 0, 1, 1, 0, 1.0, 30.0, 5.0)
    _set_gaussian(tgt, 0, 2, 2, 0, 4.0, 30.0, 5.0)
    _set_gaussian(tgt, 0, 3, 3, 0, 9.0, 30.0, 5.0)

    return [(imgs, tgt)]


def test_selector_finds_all_categories():
    criterion = build_loss(n_gaussians=2)
    selector  = ExampleSelector(params_per_gaussian=3, amp_zero_thr=1e-3)
    selected  = selector.select(_category_loader(), criterion.norm_stats.denormalize_output)

    assert selected["single_gaussian"] == [(0, 1, 1), (0, 2, 2), (0, 3, 3)]
    assert selected["two_overlapping"] == [(1, 4, 4), (1, 5, 5), (1, 6, 6)]
    assert selected["two_separated"]   == [(3, 1, 1), (3, 2, 2), (3, 3, 3)]
    assert selected["two_distant"]     == [(2, 1, 1), (2, 2, 2), (2, 3, 3)]


def test_figures_capture_reference_once():
    figs      = ReconstructionFigures(FigureTracker(), torch.device("cpu"))
    criterion = build_loss(n_gaussians=2)

    loader = _category_loader()
    figs.capture_reference(loader, criterion)
    first = figs._images

    figs.capture_reference(_single_only_loader(), criterion)

    assert figs._images is first
    assert len(figs._pixels) == 12


def test_figures_log_noop_without_reference():
    tracker = FigureTracker()
    figs    = ReconstructionFigures(tracker, torch.device("cpu"))

    figs.log(model=None, criterion=None, epoch=0)

    assert tracker.figures == []


def test_figures_log_emits_one_figure_per_example():
    tracker   = FigureTracker()
    figs      = ReconstructionFigures(tracker, torch.device("cpu"))
    criterion = build_loss(n_gaussians=2)
    figs.capture_reference(_category_loader(), criterion)

    model = lambda images: valid_param_tensor(images.shape[0], 2, images.shape[2], images.shape[3], seed=4)

    figs.log(model, criterion, epoch=5)

    expected_tags = [f"reconstruction/{name}_{rank}/val" for name in ExampleSelector.CATEGORIES for rank in (1, 2, 3)]

    assert [tag for tag, _, _ in tracker.figures] == expected_tags
    assert all(step == 5 for _, _, step in tracker.figures)


def test_figures_ylim_fixed_across_epochs():
    tracker   = FigureTracker()
    figs      = ReconstructionFigures(tracker, torch.device("cpu"))
    criterion = build_loss(n_gaussians=2)
    figs.capture_reference(_category_loader(), criterion)

    small = lambda images: valid_param_tensor(images.shape[0], 2, images.shape[2], images.shape[3], seed=4)
    large = lambda images: 20.0 * valid_param_tensor(images.shape[0], 2, images.shape[2], images.shape[3], seed=4)

    figs.log(small, criterion, epoch=0)
    figs.log(large, criterion, epoch=1)

    first  = [fig.axes[0].get_ylim() for _, fig, step in tracker.figures if step == 0]
    second = [fig.axes[0].get_ylim() for _, fig, step in tracker.figures if step == 1]

    assert first == second
    assert all(lo == 0.0 for lo, _ in first)


def test_figures_partial_categories_keep_available_examples():
    tracker   = FigureTracker()
    figs      = ReconstructionFigures(tracker, torch.device("cpu"))
    criterion = build_loss(n_gaussians=2)
    figs.capture_reference(_single_only_loader(), criterion)

    model = lambda images: valid_param_tensor(images.shape[0], 2, images.shape[2], images.shape[3], seed=4)
    figs.log(model, criterion, epoch=0)

    assert len(tracker.figures) == 3
    assert all(tag.startswith("reconstruction/single_gaussian_") for tag, _, _ in tracker.figures)


def test_figures_disable_when_no_category_matches():
    tracker   = FigureTracker()
    figs      = ReconstructionFigures(tracker, torch.device("cpu"))
    criterion = build_loss(n_gaussians=2)

    gen    = torch.Generator().manual_seed(2)
    loader = [(torch.randn(2, 2, 8, 8, generator=gen), torch.zeros(2, 6, 8, 8))]

    figs.capture_reference(loader, criterion)
    figs.log(model=None, criterion=criterion, epoch=0)

    assert figs._disabled is True
    assert tracker.figures == []


def _vitals(n_k: int = 2) -> "SlotVitals":
    return SlotVitals(params_per_gaussian=3)


def _phys(amps: list[float], H: int = 2, W: int = 2) -> torch.Tensor:
    n_k  = len(amps)
    phys = torch.zeros(1, n_k * 3, H, W)

    for k, amp in enumerate(amps):
        phys[:, 3 * k]     = amp
        phys[:, 3 * k + 1] = 10.0
        phys[:, 3 * k + 2] = 3.0

    return phys


def test_slot_vitals_inactive_until_begin():
    vitals = _vitals()
    vitals.observe(_phys([1.0, 1.0]))

    assert vitals.scalars() == {}


def test_slot_vitals_active_fractions_and_amp_means():
    vitals = _vitals()
    vitals.begin(1e-3)
    vitals.observe(_phys([2.0, 0.0]))

    scalars = vitals.scalars()

    assert scalars["slot_vitals/active_frac/slot_0"] == pytest.approx(1.0)
    assert scalars["slot_vitals/active_frac/slot_1"] == pytest.approx(0.0)
    assert scalars["slot_vitals/amp_mean/slot_0"]    == pytest.approx(2.0)
    assert scalars["slot_vitals/amp_mean/slot_1"]    == pytest.approx(0.0)


def test_slot_vitals_entropy_is_low_when_one_slot_dominates():
    balanced = _vitals()
    balanced.begin(1e-3)
    balanced.observe(_phys([1.0, 1.0]))

    collapsed = _vitals()
    collapsed.begin(1e-3)
    collapsed.observe(_phys([1.0, 0.0]))

    assert balanced.scalars()["slot_vitals/usage_entropy"]  == pytest.approx(1.0)
    assert collapsed.scalars()["slot_vitals/usage_entropy"] == pytest.approx(0.0, abs=1e-6)


def test_slot_vitals_accumulates_gradient_norms_per_slot():
    vitals = _vitals()

    grad          = torch.zeros(1, 6, 2, 2)
    grad[:, 0:3]  = 3.0

    vitals.observe_gradient(grad)
    vitals.observe_gradient(grad)

    scalars = vitals.gradient_scalars()

    assert scalars["slot_vitals/grad_norm/slot_0"] == pytest.approx((9.0 * 12) ** 0.5)
    assert scalars["slot_vitals/grad_norm/slot_1"] == pytest.approx(0.0)
    assert vitals.gradient_scalars() == {}


def test_slot_vitals_end_resets_validation_state():
    vitals = _vitals()
    vitals.begin(1e-3)
    vitals.observe(_phys([1.0, 1.0]))
    vitals.end()

    assert vitals.scalars() == {}
