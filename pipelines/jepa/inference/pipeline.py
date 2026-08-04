from __future__ import annotations

from pipelines.backbone.inference.pipeline  import InferenceComponents
from pipelines.backbone.inference.predictor import Predictor
from pipelines.jepa.inference.embedding     import JepaEmbeddingEvaluator
from pipelines.jepa.inference.loader        import JepaParamRunLoader, JepaRunLoader
from pipelines.jepa.inference.predictor     import JepaCurvePredictor

JEPA_INFERENCE_COMPONENTS = InferenceComponents(
    loader_cls              = JepaRunLoader,
    predictor_cls           = JepaCurvePredictor,
    param_space             = False,
    embedding_evaluator_cls = JepaEmbeddingEvaluator,
)

JEPA_PARAM_INFERENCE_COMPONENTS = InferenceComponents(
    loader_cls    = JepaParamRunLoader,
    predictor_cls = Predictor,
    param_space   = True,
)
