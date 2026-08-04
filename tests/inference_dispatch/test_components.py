from __future__ import annotations

import pytest

from pipelines.backbone.inference.loader             import RunLoader
from pipelines.backbone.inference.predictor          import Predictor
from pipelines.jepa.inference.pipeline               import JEPA_INFERENCE_COMPONENTS, JEPA_PARAM_INFERENCE_COMPONENTS
from pipelines.shared.inference.inference_components import InferenceComponentsResolver


def _make_run(tmp_path, *filenames):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (meta / name).write_text("{}")
    return tmp_path


def test_backbone_run_resolves_default_components(tmp_path):
    run = _make_run(tmp_path, "model_config.json")

    components = InferenceComponentsResolver.for_run(run)

    assert components.loader_cls              is RunLoader
    assert components.predictor_cls           is Predictor
    assert components.param_space             is True
    assert components.embedding_evaluator_cls is None


def test_profile_jepa_run_resolves_curve_components(tmp_path):
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json")

    assert InferenceComponentsResolver.for_run(run) is JEPA_INFERENCE_COMPONENTS


def test_image_jepa_run_resolves_param_components(tmp_path):
    run = _make_run(tmp_path, "model_config.json", "image_autoencoder_config.json")

    assert InferenceComponentsResolver.for_run(run) is JEPA_PARAM_INFERENCE_COMPONENTS


def test_dual_autoencoder_jepa_run_resolves_profile_curve_components(tmp_path):
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json", "image_autoencoder_config.json")

    assert InferenceComponentsResolver.for_run(run) is JEPA_INFERENCE_COMPONENTS


def test_standalone_profile_autoencoder_run_raises(tmp_path):
    run = _make_run(tmp_path, "profile_autoencoder_config.json")

    with pytest.raises(ValueError, match="standalone profile-autoencoder"):
        InferenceComponentsResolver.for_run(run)


def test_standalone_image_autoencoder_run_raises(tmp_path):
    run = _make_run(tmp_path, "image_autoencoder_config.json")

    with pytest.raises(ValueError, match="standalone image-autoencoder"):
        InferenceComponentsResolver.for_run(run)
