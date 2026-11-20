from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from models import BACKBONE_MODEL_REGISTRY, get_backbone

BASELINE_PATH = Path(__file__).resolve().parents[1] / "state_dict_baseline.json"


class StateDictSignature:
    @staticmethod
    def compute(model: torch.nn.Module) -> dict:
        state_dict = model.state_dict()
        lines      = [f"{key} {tuple(tensor.shape)}" for key, tensor in state_dict.items()]
        digest     = hashlib.sha256("\n".join(lines).encode()).hexdigest()

        return {
            "signature"      : digest,
            "num_keys"       : len(state_dict),
            "num_parameters" : sum(p.numel() for p in model.parameters()),
        }


class TestStateDictCompatibility:
    @pytest.fixture(scope="class")
    def baseline(self) -> dict:
        return json.loads(BASELINE_PATH.read_text())

    @pytest.mark.parametrize("key", sorted(BACKBONE_MODEL_REGISTRY))
    def test_state_dict_matches_baseline(self, key: str, baseline: dict):
        model, _config = get_backbone(key)
        signature      = StateDictSignature.compute(model)
        expected       = baseline[key]

        assert signature["num_keys"] == expected["num_keys"], f"state_dict key count changed for '{key}'"
        assert signature["num_parameters"] == expected["num_parameters"], f"parameter count changed for '{key}'"
        assert signature["signature"] == expected["signature"], f"state_dict keys or shapes changed for '{key}': checkpoints trained before this change will not load"

    def test_baseline_covers_the_registry(self, baseline: dict):
        assert set(baseline) == set(BACKBONE_MODEL_REGISTRY)
