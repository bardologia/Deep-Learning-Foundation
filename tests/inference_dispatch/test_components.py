"""Tests that InferenceComponentsResolver picks loader, predictor and evaluator classes per run.

Covers the plain backbone default, the JEPA curve and parameter component sets selected by the
autoencoder configs persisted next to the backbone config, and the refusal to run inference on a
standalone autoencoder run through this resolver."""

from __future__ import annotations

import pytest

from pipelines.backbone.inference.loader             import RunLoader
from pipelines.backbone.inference.predictor          import Predictor
from pipelines.jepa.inference.pipeline               import JEPA_INFERENCE_COMPONENTS, JEPA_PARAM_INFERENCE_COMPONENTS
from pipelines.shared.inference.inference_components import InferenceComponentsResolver


def _make_run(tmp_path, *filenames):
    """Creates a run directory whose meta/ holds the named placeholder config files."""
    meta = tmp_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (meta / name).write_text("{}")
    return tmp_path


def test_backbone_run_resolves_default_components(tmp_path):
    """A backbone-only run resolves to the plain RunLoader, Predictor and parameter space."""
    run = _make_run(tmp_path, "model_config.json")

    components = InferenceComponentsResolver.for_run(run)

    assert components.loader_cls              is RunLoader
    assert components.predictor_cls           is Predictor
    assert components.param_space             is True
    assert components.embedding_evaluator_cls is None


def test_profile_jepa_run_resolves_curve_components(tmp_path):
    """A backbone paired with a profile autoencoder resolves to the JEPA curve components."""
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json")

    assert InferenceComponentsResolver.for_run(run) is JEPA_INFERENCE_COMPONENTS


def test_image_jepa_run_resolves_param_components(tmp_path):
    """A backbone paired with an image autoencoder resolves to the JEPA parameter components."""
    run = _make_run(tmp_path, "model_config.json", "image_autoencoder_config.json")

    assert InferenceComponentsResolver.for_run(run) is JEPA_PARAM_INFERENCE_COMPONENTS


def test_dual_autoencoder_jepa_run_resolves_profile_curve_components(tmp_path):
    """With both autoencoders present the profile branch decides, giving the curve components."""
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json", "image_autoencoder_config.json")

    assert InferenceComponentsResolver.for_run(run) is JEPA_INFERENCE_COMPONENTS


def test_standalone_profile_autoencoder_run_raises(tmp_path):
    """A profile-autoencoder run without a backbone is rejected with an explanatory error."""
    run = _make_run(tmp_path, "profile_autoencoder_config.json")

    with pytest.raises(ValueError, match="standalone profile-autoencoder"):
        InferenceComponentsResolver.for_run(run)


def test_standalone_image_autoencoder_run_raises(tmp_path):
    """An image-autoencoder run without a backbone is rejected with an explanatory error."""
    run = _make_run(tmp_path, "image_autoencoder_config.json")

    with pytest.raises(ValueError, match="standalone image-autoencoder"):
        InferenceComponentsResolver.for_run(run)
