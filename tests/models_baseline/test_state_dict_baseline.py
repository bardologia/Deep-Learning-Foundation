"""Tests guarding checkpoint compatibility across every model family.

Each registered backbone, autoencoder, unrolled and dual model is rebuilt and its
state-dict key/shape signature compared against the committed baseline, so that any
architecture change that would invalidate existing checkpoints fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from models import (
    BACKBONE_HEADS,
    BACKBONE_MODEL_REGISTRY,
    DUAL_MODEL_REGISTRY,
    IMAGE_AE_MODEL_REGISTRY,
    PROFILE_AE_MODEL_REGISTRY,
    get_backbone,
    get_dual,
    get_image_autoencoder,
    get_profile_autoencoder,
)

BASELINE_PATH = Path(__file__).resolve().parents[1] / "state_dict_baseline.json"


def _backbone_cases() -> dict:
    """Returns the backbone build cases keyed by model name and non-default head."""
    cases = {}
    for name in BACKBONE_MODEL_REGISTRY:
        for head in BACKBONE_HEADS:
            key        = name if head == "conv" else f"{name}-{head}"
            cases[key] = (name, {"head": head})

    return cases


def _plain_cases(registry: dict) -> dict:
    """Returns build cases with no overrides for every name in a model registry."""
    return {name: (name, {}) for name in registry}


def _build_case(factory, case: tuple):
    """Returns the model and config produced by a factory for one build case."""
    name, overrides = case
    return factory(name, **overrides)


MODEL_FAMILIES = {
    "backbone"            : (_backbone_cases(),                       get_backbone),
    "profile_autoencoder" : (_plain_cases(PROFILE_AE_MODEL_REGISTRY), get_profile_autoencoder),
    "image_autoencoder"   : (_plain_cases(IMAGE_AE_MODEL_REGISTRY),   get_image_autoencoder),
    "dual"                : (_plain_cases(DUAL_MODEL_REGISTRY),       get_dual),
}

FAMILY_MODEL_CASES = [(family, key) for family, (cases, _factory) in MODEL_FAMILIES.items() for key in sorted(cases)]


class StateDictSignature:
    """Fingerprints a module by the keys and shapes of its state dict."""
    @staticmethod
    def compute(model: torch.nn.Module) -> dict:
        """Returns the state-dict digest, key count and parameter count of a module."""
        state_dict = model.state_dict()
        lines      = [f"{key} {tuple(tensor.shape)}" for key, tensor in state_dict.items()]
        digest     = hashlib.sha256("\n".join(lines).encode()).hexdigest()

        return {
            "signature"      : digest,
            "num_keys"       : len(state_dict),
            "num_parameters" : sum(p.numel() for p in model.parameters()),
        }


class TestStateDictCompatibility:
    """Compares every registered model against the committed state-dict baseline."""
    @pytest.fixture(scope="class")
    def baseline(self) -> dict:
        """Returns the parsed baseline file, failing the test when it is missing."""
        if not BASELINE_PATH.exists():
            pytest.fail(f"Missing baseline {BASELINE_PATH}. Generate it with scripts/generate_state_dict_baseline.py")
        return json.loads(BASELINE_PATH.read_text())

    @pytest.mark.parametrize("family, key", FAMILY_MODEL_CASES)
    def test_state_dict_matches_baseline(self, family: str, key: str, baseline: dict):
        """Verifies a model's state-dict keys, shapes and parameter count still match the baseline."""
        if key not in baseline.get(family, {}):
            pytest.fail(f"Model '{family}/{key}' missing from baseline. Regenerate with scripts/generate_state_dict_baseline.py")

        cases, factory = MODEL_FAMILIES[family]
        model, _config = _build_case(factory, cases[key])
        signature      = StateDictSignature.compute(model)
        expected       = baseline[family][key]

        assert signature["num_keys"] == expected["num_keys"], f"state_dict key count changed for '{family}/{key}'"
        assert signature["num_parameters"] == expected["num_parameters"], f"parameter count changed for '{family}/{key}'"
        assert signature["signature"] == expected["signature"], f"state_dict keys or shapes changed for '{family}/{key}': checkpoints trained before this change will not load"

    def test_baseline_covers_full_registries(self, baseline: dict):
        """Verifies the baseline holds exactly the models the registries currently expose."""
        expected = {family: set(cases) for family, (cases, _factory) in MODEL_FAMILIES.items()}
        actual   = {family: set(models) for family, models in baseline.items()}
        assert actual == expected
