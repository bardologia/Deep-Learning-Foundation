from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.training.gradients import GradientClipper


def _grad_config(mode="fixed", max_grad_norm=1.0, window=5, percentile=90.0, mean_std_k=2.0, epsilon=1e-6, hist_freq=10):
    return SimpleNamespace(
        gradient_clipper=SimpleNamespace(
            clip_mode           = mode,
            max_grad_norm       = max_grad_norm,
            adaptive_window     = window,
            adaptive_percentile = percentile,
            adaptive_mean_std_k = mean_std_k,
            clip_epsilon        = epsilon,
            log_histogram_freq  = hist_freq,
        )
    )


def _model_with_grads(grad_values):
    model = torch.nn.Linear(len(grad_values), 1, bias=False)
    with torch.no_grad():
        model.weight.grad = torch.tensor([grad_values], dtype=torch.float32)
    return model


def test_global_norm_matches_l2(logger, tracker):
    model = _model_with_grads([3.0, 4.0])
    norm  = GradientClipper.global_norm(model)
    assert norm == pytest.approx(5.0)


def test_global_norm_combines_multiple_params():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=True))
    with torch.no_grad():
        model[0].weight.grad = torch.tensor([[3.0]])
        model[0].bias.grad   = torch.tensor([4.0])

    assert GradientClipper.global_norm(model) == pytest.approx(5.0)


def test_global_norm_zero_when_no_grads():
    model = torch.nn.Linear(2, 1)
    assert GradientClipper.global_norm(model) == 0.0


def test_fixed_clip_scales_gradients_to_threshold(logger, tracker):
    config = _grad_config(mode="fixed", max_grad_norm=2.5, epsilon=0.0)
    clip   = GradientClipper(config, logger, tracker)

    model     = _model_with_grads([3.0, 4.0])
    norm_back = clip.maybe_clip(model, global_step=0)

    assert norm_back == pytest.approx(5.0)
    assert GradientClipper.global_norm(model) == pytest.approx(2.5)


def test_fixed_clip_leaves_small_gradients_untouched(logger, tracker):
    config = _grad_config(mode="fixed", max_grad_norm=100.0, epsilon=0.0)
    clip   = GradientClipper(config, logger, tracker)

    model = _model_with_grads([0.3, 0.4])
    clip.maybe_clip(model, global_step=0)

    assert GradientClipper.global_norm(model) == pytest.approx(0.5)


def test_disabled_mode_returns_norm_without_clipping(logger, tracker):
    config = _grad_config(mode="disabled")
    clip   = GradientClipper(config, logger, tracker)

    model = _model_with_grads([3.0, 4.0])
    out   = clip.maybe_clip(model, global_step=0)

    assert out == pytest.approx(5.0)
    assert GradientClipper.global_norm(model) == pytest.approx(5.0)


def test_adaptive_percentile_returns_none_before_window_full(logger, tracker):
    config = _grad_config(mode="adaptive_percentile", window=5)
    clip   = GradientClipper(config, logger, tracker)

    clip.history = [1.0, 2.0]
    assert clip._compute_adaptive_threshold() is None


def test_adaptive_percentile_threshold_value(logger, tracker):
    config = _grad_config(mode="adaptive_percentile", window=5, percentile=50.0)
    clip   = GradientClipper(config, logger, tracker)

    clip.history = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected     = float(np.percentile(np.float32([10, 20, 30, 40, 50]), 50.0))

    assert clip._compute_adaptive_threshold() == pytest.approx(expected)


def test_adaptive_mean_std_threshold_value(logger, tracker):
    config = _grad_config(mode="adaptive_mean_std", window=4, mean_std_k=2.0)
    clip   = GradientClipper(config, logger, tracker)

    data         = np.float32([1.0, 2.0, 3.0, 4.0])
    clip.history = [1.0, 2.0, 3.0, 4.0]
    expected     = float(data.mean() + 2.0 * data.std())

    assert clip._compute_adaptive_threshold() == pytest.approx(expected)


def test_adaptive_uses_only_last_window(logger, tracker):
    config = _grad_config(mode="adaptive_mean_std", window=2, mean_std_k=0.0)
    clip   = GradientClipper(config, logger, tracker)

    clip.history = [1000.0, 5.0, 7.0]
    expected     = float(np.float32([5.0, 7.0]).mean())

    assert clip._compute_adaptive_threshold() == pytest.approx(expected)


def test_maybe_clip_no_threshold_returns_norm_unchanged(logger, tracker):
    config = _grad_config(mode="adaptive_percentile", window=10)
    clip   = GradientClipper(config, logger, tracker)

    model = _model_with_grads([3.0, 4.0])
    out   = clip.maybe_clip(model, global_step=0)

    assert out == pytest.approx(5.0)
    assert GradientClipper.global_norm(model) == pytest.approx(5.0)


def test_record_appends_history(logger, tracker):
    config = _grad_config()
    clip   = GradientClipper(config, logger, tracker)

    clip.record(1.5, global_step=1)
    clip.record(2.5, global_step=2)

    assert clip.history == [1.5, 2.5]


def test_record_logs_histogram_on_freq(logger, tracker):
    config = _grad_config(hist_freq=3)
    clip   = GradientClipper(config, logger, tracker)

    for step in range(1, 4):
        clip.record(float(step), global_step=step)

    assert any(tag == "optim/grad_norm_hist" for tag, _, _ in tracker.histograms)


def test_check_gradients_detects_nan(logger, tracker):
    config = _grad_config()
    clip   = GradientClipper(config, logger, tracker)

    model = _model_with_grads([float("nan"), 1.0])
    assert clip.check_gradients(model, global_step=0) is True


def test_check_gradients_clean_returns_false(logger, tracker):
    config = _grad_config()
    clip   = GradientClipper(config, logger, tracker)

    model = _model_with_grads([1.0, 2.0])
    assert clip.check_gradients(model, global_step=0) is False


def test_fixed_clip_logs_norm_and_ratio(logger, tracker):
    config = _grad_config(mode="fixed", max_grad_norm=2.5, epsilon=0.0)
    clip   = GradientClipper(config, logger, tracker)

    model = _model_with_grads([3.0, 4.0])
    clip.maybe_clip(model, global_step=0)

    tags = {tag for tag, _, _ in tracker.scalars}
    assert "optim/grad_norm"           in tags
    assert "optim/grad_clip_ratio"     in tags
    assert "optim/grad_clip_threshold" not in tags


def test_per_group_norms_logged_for_multiple_groups(logger, tracker):
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        model.bias.grad   = torch.tensor([12.0])

    param_groups = [
        {"params": [model.weight], "name": "encoder"},
        {"params": [model.bias],   "name": "head"},
    ]

    clip = GradientClipper(_grad_config(mode="disabled"), logger, tracker, param_groups=param_groups)
    clip.maybe_clip(model, global_step=0)

    recorded = {tag: val for tag, val, _ in tracker.scalars}

    assert recorded["optim/grad_norm"]         == pytest.approx(13.0)
    assert recorded["optim/grad_norm/encoder"] == pytest.approx(5.0)
    assert recorded["optim/grad_norm/head"]    == pytest.approx(12.0)


def test_per_group_norms_skipped_for_single_group(logger, tracker):
    model = _model_with_grads([3.0, 4.0])

    param_groups = [{"params": list(model.parameters()), "name": "main"}]

    clip = GradientClipper(_grad_config(mode="disabled"), logger, tracker, param_groups=param_groups)
    clip.maybe_clip(model, global_step=0)

    tags = {tag for tag, _, _ in tracker.scalars}

    assert "optim/grad_norm"      in tags
    assert "optim/grad_norm/main" not in tags


def test_adaptive_clip_logs_threshold(logger, tracker):
    config = _grad_config(mode="adaptive_percentile", window=2, percentile=50.0, epsilon=0.0)
    clip   = GradientClipper(config, logger, tracker)

    clip.history = [1.0, 1.0]
    model = _model_with_grads([3.0, 4.0])
    clip.maybe_clip(model, global_step=0)

    tags = {tag for tag, _, _ in tracker.scalars}
    assert "optim/grad_clip_threshold" in tags
