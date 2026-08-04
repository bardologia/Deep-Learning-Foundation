from __future__ import annotations

from configuration.training.general.loss import ParamMatching


class AblationCatalog:

    COMPLETE_PREFIX = "curriculum.complete."

    CURRICULUM_SWAP_EPOCH = 15

    PARAM_MATCH_FULL = ParamMatching.HUNGARIAN

    GROUP_LR_DEFAULTS = (
        ("encoder_lr",     3e-4),
        ("bottleneck_lr",  3e-4),
        ("decoder_lr",     3e-4),
        ("output_head_lr", 1e-3),
    )
    SINGLE_GROUP_LR = 3e-4

    FULL_ARCHITECTURE     = "resunet"
    BASELINE_ARCHITECTURE = "unet"

    DUAL_GROUP_LR_DEFAULTS = (
        ("params_trunk_lr",    3e-4),
        ("existence_trunk_lr", 3e-4),
        ("output_head_lr",     1e-3),
    )

    DUAL_FULL_ARCHITECTURE     = "unet_skip"
    DUAL_BASELINE_ARCHITECTURE = "unet"

    CHANNEL_NORMS = (
        ("out_amp",   "out_amp",   "robust_iqr_log1p", "zscore"),
        ("out_sigma", "out_sigma", "robust_iqr_log1p", "zscore"),
        ("pass_mag",  "pass_mag",  "robust_iqr_log1p", "zscore_log1p"),
        ("ifg_phase", "ifg_phase", "fixed_div_pi",     "zscore"),
        ("ifg_mag",   "ifg_mag",   "robust_iqr_log1p", "zscore"),
        ("dem",       "dem",       "robust_iqr_log1p", "zscore"),
    )

    AUGMENTATION_DEFAULTS = (
        ("p_flip_h", 0.5),
        ("p_flip_v", 0.5),
    )

    PHYSICS_WEIGHT = 0.05
    COSINE_WEIGHT  = 0.05

    LOSS_TERMS = (
        ("smoothness_tv", "use_smoothness_tv", "weight_smoothness_tv", 1e-4),
    )

    @classmethod
    def _channel_norm_features(cls) -> list[dict]:
        return [
            {
                "label"   : label,
                "group"   : "channel normalization",
                "enable"  : {f"normalization.{field}": full},
                "degrade" : {f"normalization.{field}": degraded},
            }
            for label, field, full, degraded in cls.CHANNEL_NORMS
        ]

    @classmethod
    def _normalization_features(cls) -> list[dict]:
        return [
            {
                "label"   : "normalization",
                "group"   : "normalization",
                "enable"  : {"normalization.input_strategy": "per_slot", "normalization.output_strategy": "per_slot"},
                "degrade" : {"normalization.input_strategy": "zscore",   "normalization.output_strategy": "zscore"},
            },
            {
                "label"   : "output_clamp",
                "group"   : "normalization",
                "enable"  : {"normalization.clamp_output": True},
                "degrade" : {"normalization.clamp_output": False},
            },
        ]

    @classmethod
    def _augmentation_features(cls) -> list[dict]:
        return [
            {
                "label"   : "augmentation",
                "group"   : "augmentation",
                "enable"  : {f"augmentation.{key}": value for key, value in cls.AUGMENTATION_DEFAULTS},
                "degrade" : {f"augmentation.{key}": 0.0 for key, _ in cls.AUGMENTATION_DEFAULTS},
            },
        ]

    @classmethod
    def _schedule_features(cls) -> list[dict]:
        return [
            {
                "label"   : "physics_curriculum",
                "group"   : "schedule",
                "enable"  : {"curriculum.enabled": True, "curriculum.swap_epoch": cls.CURRICULUM_SWAP_EPOCH},
                "degrade" : {"curriculum.enabled": False},
            },
            {
                "label"   : "lr_warmup",
                "group"   : "schedule",
                "enable"  : {"training.warmup_enabled": True},
                "degrade" : {"training.warmup_enabled": False},
            },
            {
                "label"   : "lr_per_group",
                "group"   : "optimizer",
                "enable"  : {"model_overrides": {key: lr                  for key, lr in cls.GROUP_LR_DEFAULTS}},
                "degrade" : {"model_overrides": {key: cls.SINGLE_GROUP_LR for key, _  in cls.GROUP_LR_DEFAULTS}},
            },
        ]

    @classmethod
    def _physics_features(cls) -> list[dict]:
        complete = cls.COMPLETE_PREFIX

        return [
            {
                "label"   : "coherence_resyn",
                "group"   : "physics",
                "enable"  : {f"{complete}use_coherence_resyn": True,  f"{complete}weight_coherence_resyn": cls.PHYSICS_WEIGHT},
                "degrade" : {f"{complete}use_coherence_resyn": False, f"{complete}weight_coherence_resyn": 0.0},
            },
            {
                "label"   : "covariance_match",
                "group"   : "physics",
                "enable"  : {f"{complete}use_covariance_match": True,  f"{complete}weight_covariance_match": cls.PHYSICS_WEIGHT},
                "degrade" : {f"{complete}use_covariance_match": False, f"{complete}weight_covariance_match": 0.0},
            },
        ]

    @classmethod
    def _loss_component_features(cls) -> list[dict]:
        complete = cls.COMPLETE_PREFIX

        return [
            {
                "label"   : "cosine_curve",
                "group"   : "loss components",
                "enable"  : {f"{complete}use_cosine_curve": True,  f"{complete}weight_cosine_curve": cls.COSINE_WEIGHT},
                "degrade" : {f"{complete}use_cosine_curve": False, f"{complete}weight_cosine_curve": 0.0},
            },
        ]

    @classmethod
    def _slot_features(cls) -> list[dict]:
        prefix = cls.COMPLETE_PREFIX

        return [
            {
                "label"   : "focal",
                "group"   : "slot presence",
                "enable"  : {f"{prefix}amp_focal_gamma": 2.0},
                "degrade" : {f"{prefix}amp_focal_gamma": 0.0},
            },
            {
                "label"   : "active_norm",
                "group"   : "slot presence",
                "enable"  : {f"{prefix}use_active_normalization": True},
                "degrade" : {f"{prefix}use_active_normalization": False},
            },
            {
                "label"   : "presence_balance",
                "group"   : "slot presence",
                "enable"  : {f"{prefix}presence_balance": True},
                "degrade" : {f"{prefix}presence_balance": False},
            },
        ]

    @classmethod
    def _architecture_features(cls) -> list[dict]:
        complete = cls.COMPLETE_PREFIX

        return [
            {
                "label"   : "architecture_param_loss",
                "group"   : "architecture",
                "enable"  : {
                    "backbone_name"               : cls.FULL_ARCHITECTURE,
                    f"{complete}use_param_l1"     : True,
                    f"{complete}weight_param_l1"  : 1.0,
                    f"{complete}use_param_mse"    : False,
                    f"{complete}weight_param_mse" : 0.0,
                },
                "degrade" : {
                    "backbone_name"               : cls.BASELINE_ARCHITECTURE,
                    f"{complete}use_param_l1"     : False,
                    f"{complete}weight_param_l1"  : 0.0,
                    f"{complete}use_param_mse"    : True,
                    f"{complete}weight_param_mse" : 1.0,
                },
            },
        ]

    @classmethod
    def _loss_toggle(cls, label: str, use_key: str, weight_key: str, weight: float) -> dict:
        prefix = cls.COMPLETE_PREFIX

        return {
            "label"   : label,
            "group"   : "loss terms",
            "enable"  : {f"{prefix}{use_key}": True,  f"{prefix}{weight_key}": weight},
            "degrade" : {f"{prefix}{use_key}": False, f"{prefix}{weight_key}": 0.0},
        }

    @classmethod
    def features(cls) -> list[dict]:
        loss_terms = [cls._loss_toggle(label, use_key, weight_key, weight) for label, use_key, weight_key, weight in cls.LOSS_TERMS]

        return (
            cls._channel_norm_features()
            + cls._normalization_features()
            + cls._augmentation_features()
            + cls._schedule_features()
            + cls._physics_features()
            + cls._loss_component_features()
            + cls._slot_features()
            + cls._architecture_features()
            + loss_terms
        )

    @classmethod
    def as_dict(cls) -> dict:
        return {feature["label"]: feature for feature in cls.features()}

    DEFAULT_ORDER = (
        "covariance_match", "physics_curriculum", "coherence_resyn",
        "cosine_curve", "architecture_param_loss", "augmentation",
        "active_norm", "lr_per_group", "lr_warmup",
        "out_sigma", "out_amp", "ifg_phase", "pass_mag",
        "output_clamp",
    )

    @classmethod
    def default_features(cls) -> list[dict]:
        catalog = cls.as_dict()
        return [catalog[label] for label in cls.DEFAULT_ORDER]

    @classmethod
    def _dual_lr_feature(cls) -> dict:
        return {
            "label"   : "lr_per_group",
            "group"   : "optimizer",
            "enable"  : {"model_overrides": {key: lr                  for key, lr in cls.DUAL_GROUP_LR_DEFAULTS}},
            "degrade" : {"model_overrides": {key: cls.SINGLE_GROUP_LR for key, _  in cls.DUAL_GROUP_LR_DEFAULTS}},
        }

    @classmethod
    def _dual_architecture_feature(cls) -> dict:
        complete = cls.COMPLETE_PREFIX

        return {
            "label"   : "architecture_param_loss",
            "group"   : "architecture",
            "enable"  : {
                "params_backbone"             : cls.DUAL_FULL_ARCHITECTURE,
                "existence_backbone"          : cls.DUAL_FULL_ARCHITECTURE,
                f"{complete}use_param_l1"     : True,
                f"{complete}weight_param_l1"  : 1.0,
                f"{complete}use_param_mse"    : False,
                f"{complete}weight_param_mse" : 0.0,
            },
            "degrade" : {
                "params_backbone"             : cls.DUAL_BASELINE_ARCHITECTURE,
                "existence_backbone"          : cls.DUAL_BASELINE_ARCHITECTURE,
                f"{complete}use_param_l1"     : False,
                f"{complete}weight_param_l1"  : 0.0,
                f"{complete}use_param_mse"    : True,
                f"{complete}weight_param_mse" : 1.0,
            },
        }

    @classmethod
    def dual_default_features(cls) -> list[dict]:
        catalog = cls.as_dict()
        catalog.update({feature["label"]: feature for feature in (cls._dual_lr_feature(), cls._dual_architecture_feature())})
        return [catalog[label] for label in cls.DEFAULT_ORDER]
