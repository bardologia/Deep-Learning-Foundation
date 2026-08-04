from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

import pytest

from configuration.dataset                import AugmentationConfig
from configuration.training               import LossConfig, ParamMatching
from pipelines.shared.training.run_naming import RunNaming
from scripts.rename_runs                  import RunRenamer


def _loss() -> LossConfig:
    return LossConfig(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True, param_matching=ParamMatching.HUNGARIAN)


def _augmentation() -> AugmentationConfig:
    return AugmentationConfig(p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.0, p_noise=0.0)


def _dual_config(params_backbone: str, existence_backbone: str, params_input: list, existence_input: list) -> dict:
    return {
        "head"               : "set_pred",
        "params_backbone"    : params_backbone,
        "existence_backbone" : existence_backbone,
        "params_input"       : params_input,
        "existence_input"    : existence_input,
    }


def _write_metadata(run_dir: Path, config: dict, n_gaussians: int) -> None:
    loss_payload                   = asdict(_loss())
    loss_payload["param_matching"] = str(_loss().param_matching)

    (run_dir / "meta").mkdir(parents=True)
    (run_dir / "docs").mkdir()

    (run_dir / "meta" / "run_summary.json").write_text(json.dumps({"model_name": "dual_resunet", "n_gaussians": n_gaussians, "run_directory": str(run_dir)}))
    (run_dir / "meta" / "dual_model_config.json").write_text(json.dumps({"model_name": "dual_resunet", "config_type": "DualResUNetConfig", "config": config}))
    (run_dir / "meta" / "dataset_creation_config.json").write_text(json.dumps({"augmentation": asdict(_augmentation())}))
    (run_dir / "docs" / "trainer_config.json").write_text(json.dumps({"curriculum": {"complete": loss_payload}, "gaussian": {"n_default_gaussians": n_gaussians}}))


def _renamer(root: Path, apply: bool, monkeypatch, tmp_path: Path) -> RunRenamer:
    monkeypatch.chdir(tmp_path)
    return RunRenamer([root], apply=apply)


def test_pre_trunk_tag_dual_run_is_renamed_from_its_persisted_trunks(monkeypatch, tmp_path):
    root     = tmp_path / "routing"
    old_name = RunNaming.tag("dual_resunet", "set_pred", _loss(), 5, _augmentation()) + "_di-full-ifg"
    _write_metadata(root / old_name, _dual_config("unet_skip", "unet_skip", ["pass", "ifg"], ["ifg"]), n_gaussians=5)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    new_name = RunNaming.tag("dual_unet_skip", "set_pred", _loss(), 5, _augmentation(), extras=("full.ifg",)) + "_di-full-ifg"
    assert not (root / old_name).exists()
    assert (root / new_name).is_dir()

    summary = json.loads((root / new_name / "meta" / "run_summary.json").read_text())
    assert new_name in summary["run_directory"]
    assert old_name not in summary["run_directory"]


def test_input_tagged_dual_run_swaps_the_model_name_for_the_trunk_tag(monkeypatch, tmp_path):
    root     = tmp_path / "ratio"
    old_name = RunNaming.tag("dual_resunet", "set_pred", _loss(), 2, _augmentation(), extras=("full.full",)) + "_dr-90-10"
    _write_metadata(root / old_name, _dual_config("unet_skip", "local_cnn", ["pass", "ifg"], ["pass", "ifg"]), n_gaussians=2)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    new_name = RunNaming.tag("dual_unet_skip.local_cnn", "set_pred", _loss(), 2, _augmentation(), extras=("full.full",)) + "_dr-90-10"
    assert not (root / old_name).exists()
    assert (root / new_name).is_dir()


def test_dual_run_already_in_the_current_naming_is_left_alone(monkeypatch, tmp_path):
    root = tmp_path / "runs"
    name = RunNaming.tag("dual_unet_skip", "set_pred", _loss(), 2, _augmentation(), extras=("full.full",)) + "_dr-90-10"
    _write_metadata(root / name, _dual_config("unet_skip", "unet_skip", ["pass", "ifg"], ["pass", "ifg"]), n_gaussians=2)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    assert (root / name).is_dir()


def test_seed_fanout_base_is_renamed_as_one_unit(monkeypatch, tmp_path):
    root     = tmp_path / "context"
    old_name = RunNaming.tag("dual_resunet", "set_pred", _loss(), 5, _augmentation()) + "_ctx-cnn29"
    for seed in (0, 1):
        _write_metadata(root / old_name / f"seed{seed}", _dual_config("local_cnn", "local_cnn", ["pass", "ifg"], ["pass", "ifg"]), n_gaussians=5)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    new_name = RunNaming.tag("dual_local_cnn", "set_pred", _loss(), 5, _augmentation(), extras=("full.full",)) + "_ctx-cnn29"
    assert not (root / old_name).exists()
    assert (root / new_name / "seed0" / "meta" / "run_summary.json").is_file()
    assert (root / new_name / "seed1" / "meta" / "run_summary.json").is_file()

    summary = json.loads((root / new_name / "seed1" / "meta" / "run_summary.json").read_text())
    assert old_name not in summary["run_directory"]


def test_retired_input_groups_skip_the_run_instead_of_crashing(monkeypatch, tmp_path):
    root     = tmp_path / "runs"
    old_name = RunNaming.tag("dual_resunet", "set_pred", _loss(), 2, _augmentation()) + "_old"
    _write_metadata(root / old_name, _dual_config("resunet", "resunet", ["pass", "ifg", "dem"], ["pass", "ifg"]), n_gaussians=2)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    assert (root / old_name).is_dir()


def test_dry_run_plans_without_touching_directories(monkeypatch, tmp_path):
    root     = tmp_path / "routing"
    old_name = RunNaming.tag("dual_resunet", "set_pred", _loss(), 5, _augmentation()) + "_di-full-full"
    _write_metadata(root / old_name, _dual_config("unet_skip", "unet_skip", ["pass", "ifg"], ["pass", "ifg"]), n_gaussians=5)

    _renamer(root, apply=False, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    assert (root / old_name).is_dir()
