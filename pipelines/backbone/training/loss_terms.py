"""Registry of the backbone loss terms and helpers building loss configurations.

`LOSS_TERMS` is the single source of truth linking a term name to its enable
flag, weight field and evaluation space; the catalog uses it to compose
single-term, multi-term and curriculum loss configurations.
"""

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
    LossTerm("log_energy_curve",   "use_log_energy_curve",   "weight_log_energy_curve",   "denorm"),
    LossTerm("total_power_relerr", "use_total_power",        "weight_total_power",        "denorm"),
    LossTerm("moments",            "use_moments",            "weight_moments",            "denorm"),
    LossTerm("param_huber",        "use_param_huber",        "weight_param_huber",        "norm"),
    LossTerm("param_mse",          "use_param_mse",          "weight_param_mse",          "norm"),
    LossTerm("smoothness_tv",      "use_smoothness_tv",      "weight_smoothness_tv",      "norm"),
    LossTerm("param_l1",           "use_param_l1",           "weight_param_l1",           "norm"),
)


class LossComponentCatalog:
    """Builds loss configurations that enable a chosen subset of the registered terms."""

    _TERMS = {term.name: term for term in LOSS_TERMS}

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Returns the names of every registered loss term."""
        return tuple(term.name for term in LOSS_TERMS)

    @classmethod
    def standalone(cls, name: str, base: LossConfig | None = None) -> LossConfig:
        """Returns a loss config with only `name` enabled at weight 1.0."""
        return cls.combined([name], base=base)

    @classmethod
    def combined(cls, names, base: LossConfig | None = None) -> LossConfig:
        """Returns a loss config with exactly the named terms enabled at weight 1.0.

        Args:
            names : Loss component names to enable.
            base  : Config whose other settings are copied; a default LossConfig when omitted.

        Returns:
            A new loss configuration; every unnamed term is disabled.

        Raises:
            ValueError : If no name is given.
            KeyError   : If a name is not a registered loss component.
        """
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
        """Returns a curriculum-disabled config using only `name` in both stages."""
        cfg = cls.standalone(name, base=base)
        return LossCurriculumConfig(enabled=False, warmup=cfg, complete=cfg)

    @classmethod
    def combined_curriculum(cls, names, base: LossConfig | None = None) -> LossCurriculumConfig:
        """Returns a curriculum-disabled config using the named terms in both stages."""
        cfg = cls.combined(names, base=base)
        return LossCurriculumConfig(enabled=False, warmup=cfg, complete=cfg)

    @classmethod
    def probe_union(cls, curriculum: LossCurriculumConfig) -> LossConfig:
        """Returns the last stage widened with every term any earlier stage enables.

        Pretraining probes must exercise every term the run will ever compute, so
        memory and throughput are measured against the heaviest composition.

        Args:
            curriculum : Curriculum whose active stages are merged.

        Returns:
            A copy of the final stage carrying the union of the enabled terms,
            each keeping the weight of the earliest stage that enabled it.
        """
        stages = curriculum.active_stages()
        union  = copy.deepcopy(stages[-1])

        for stage in stages[:-1]:
            for term in LOSS_TERMS:
                if getattr(stage, term.use_flag) and not getattr(union, term.use_flag):
                    setattr(union, term.use_flag,   True)
                    setattr(union, term.weight_key, getattr(stage, term.weight_key))

        return union
