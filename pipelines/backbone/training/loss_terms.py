from __future__ import annotations

import copy
from collections import namedtuple

from configuration.training import LossConfig, LossCurriculumConfig


LossTerm = namedtuple("LossTerm", ["name", "use_flag", "weight_key", "space"])

LOSS_TERMS = (
    LossTerm("mse_curve",          "use_mse_curve",          "weight_mse_curve",          "denorm"),
    LossTerm("l1_curve",           "use_l1_curve",           "weight_l1_curve",           "denorm"),
    LossTerm("huber_curve",        "use_huber_curve",        "weight_huber_curve",        "denorm"),
    LossTerm("charbonnier_curve",  "use_charbonnier_curve",  "weight_charbonnier_curve",  "denorm"),
    LossTerm("cosine_curve",       "use_cosine_curve",       "weight_cosine_curve",       "denorm"),
    LossTerm("total_power_relerr", "use_total_power",        "weight_total_power",        "denorm"),
    LossTerm("moments",            "use_moments",            "weight_moments",            "denorm"),
    LossTerm("coherence_resyn",    "use_coherence_resyn",    "weight_coherence_resyn",    "denorm"),
    LossTerm("covariance_match",   "use_covariance_match",   "weight_covariance_match",   "denorm"),
    LossTerm("capon_cycle",        "use_capon_cycle",        "weight_capon_cycle",        "denorm"),
    LossTerm("param_huber",        "use_param_huber",        "weight_param_huber",        "norm"),
    LossTerm("param_mse",          "use_param_mse",          "weight_param_mse",          "norm"),
    LossTerm("param_legacy",       "use_param_legacy",       "weight_param_legacy",       "denorm"),
    LossTerm("smoothness_tv",      "use_smoothness_tv",      "weight_smoothness_tv",      "norm"),
    LossTerm("param_l1",           "use_param_l1",           "weight_param_l1",           "norm"),
)


class LossComponentCatalog:
    _TERMS = {term.name: term for term in LOSS_TERMS}

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(term.name for term in LOSS_TERMS)

    @classmethod
    def standalone(cls, name: str, base: LossConfig | None = None) -> LossConfig:
        return cls.combined([name], base=base)

    @classmethod
    def combined(cls, names, base: LossConfig | None = None) -> LossConfig:
        if not names:
            raise ValueError("names must list at least one loss component")

        unknown = [name for name in names if name not in cls._TERMS]
        if unknown:
            raise KeyError(f"unknown loss component '{unknown[0]}'; valid: {', '.join(cls.names())}")

        cfg = copy.deepcopy(base) if base is not None else LossConfig()

        for other in LOSS_TERMS:
            setattr(cfg, other.use_flag, False)

        for name in names:
            term = cls._TERMS[name]
            setattr(cfg, term.use_flag,   True)
            setattr(cfg, term.weight_key, 1.0)

        return cfg

    @classmethod
    def curriculum(cls, name: str, base: LossConfig | None = None) -> LossCurriculumConfig:
        cfg = cls.standalone(name, base=base)
        return LossCurriculumConfig(enabled=False, warmup=cfg, complete=cfg)

    @classmethod
    def combined_curriculum(cls, names, base: LossConfig | None = None) -> LossCurriculumConfig:
        cfg = cls.combined(names, base=base)
        return LossCurriculumConfig(enabled=False, warmup=cfg, complete=cfg)

    @classmethod
    def probe_union(cls, curriculum: LossCurriculumConfig) -> LossConfig:
        stages = curriculum.active_stages()
        union  = copy.deepcopy(stages[-1])

        for stage in stages[:-1]:
            for term in LOSS_TERMS:
                if getattr(stage, term.use_flag) and not getattr(union, term.use_flag):
                    setattr(union, term.use_flag,   True)
                    setattr(union, term.weight_key, getattr(stage, term.weight_key))

        return union
