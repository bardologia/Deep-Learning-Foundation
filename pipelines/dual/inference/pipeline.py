from __future__ import annotations

from pipelines.backbone.inference.pipeline import InferenceComponents
from pipelines.dual.inference.loader       import DualRunLoader


DUAL_INFERENCE_COMPONENTS = InferenceComponents(
    loader_cls = DualRunLoader,
)
