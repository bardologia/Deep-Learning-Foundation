"""End-to-end test of ModelProbe against a freshly trained dual run on real data.

A short dual training produces a run directory, which the probe then loads and drives
through every analysis it offers: prediction, attribution, what-if, field maps,
ablation, occlusion, flips, sweeps, vitals and image rendering.
"""

from __future__ import annotations

import pytest

from configuration.training           import OverfitCheckConfig
from pipelines.dual.training.pipeline import DualTrainingPipeline

from catalog_roots import CatalogRoots
from model_probe   import ModelProbe

from tests.conftest               import SilentLogger
from tests.dual_training._helpers import N_GAUSSIANS, dual_dataset_config, dual_model_config, dual_trainer_config


pytestmark = [pytest.mark.dataset, pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _train_dual_run(test_data_dir, params_dir, tmp_path):
    """Trains a short dual run on the test data and returns its run directory."""
    pipeline = DualTrainingPipeline(
        trainer_config = dual_trainer_config(test_data_dir, params_dir, tmp_path),
        dataset_config = dual_dataset_config(test_data_dir, params_dir),
        backbone_name  = "dual_resunet",
        model_config   = dual_model_config(),
        seed           = 0,
        run_name       = "probe_dual",
        overfit_check  = OverfitCheckConfig(enabled=False),
    )

    pipeline.run(probe_config=None)
    return pipeline.run_metadata.run_directory


def test_probe_loads_and_probes_a_dual_run_end_to_end(test_data_dir, params_dir, tmp_path):
    """The probe accepts a catalogued dual run, reports its trunks, channels and layers, and answers every analysis request over it."""
    run_directory = _train_dual_run(test_data_dir, params_dir, tmp_path)

    probe = ModelProbe(SilentLogger())
    probe.scanner.roots.add(run_directory.parent.resolve())
    assert probe._reject_unloadable(str(run_directory), probe.scanner.roots) == ""
    assert "outside every catalogued runs root" in probe._reject_unloadable(str(run_directory), CatalogRoots())

    probe._load_worker(str(run_directory), "test", "cpu")

    status = probe.load_status()
    assert status["state"] == "ready", status["error"]

    info = status["info"]
    assert info["backbone"]    == "dual_resunet (unet_skip + unet_skip)"
    assert info["n_gaussians"] == N_GAUSSIANS
    assert info["in_channels"] == 5

    layers = probe.layers()
    assert layers["ok"] is True
    assert any(layer["name"].startswith("trunk_params")    for layer in layers["layers"])
    assert any(layer["name"].startswith("trunk_existence") for layer in layers["layers"])

    assert probe.map_png()[:8] == PNG_MAGIC

    prediction = probe.predict({"row": 3, "col": 4})
    assert prediction["ok"] is True, prediction.get("error")
    assert len(prediction["slots"]) == N_GAUSSIANS
    assert len(prediction["curve"]) == len(prediction["z_axis"])

    attribution = probe.attribution({"row": 3, "col": 4})
    assert attribution["ok"] is True, attribution.get("error")
    assert len(attribution["channels"]) == 5
    assert {payload["family"] for payload in attribution["families"]} == {"amp", "mu", "sigma"}

    whatif = probe.whatif({"row": 3, "col": 4, "perturbation": {"kind": "drop_channel", "channel": 0}})
    assert whatif["ok"] is True, whatif.get("error")
    assert len(whatif["perturbed_slots"]) == N_GAUSSIANS

    fields = probe.fields({"row": 3, "col": 4})
    assert fields["ok"] is True, fields.get("error")
    assert len(fields["slots"]) == N_GAUSSIANS
    assert all(len(slot["families"]) == 3 for slot in fields["slots"])

    ablation = probe.ablation({"row": 3, "col": 4})
    assert ablation["ok"] is True, ablation.get("error")
    assert len(ablation["channels"]) == 5

    occlusion = probe.occlusion({"row": 3, "col": 4})
    assert occlusion["ok"] is True, occlusion.get("error")
    assert occlusion["grid"][0] >= 2 and occlusion["grid"][1] >= 2

    flips = probe.flips({"row": 3, "col": 4})
    assert flips["ok"] is True, flips.get("error")
    assert [entry["flip"] for entry in flips["entries"]] == ["none", "row", "col", "both"]
    assert flips["entries"][0]["delta_mse"] == 0.0

    sweep = probe.sweep({"row": 3, "col": 4, "kind": "noise"})
    assert sweep["ok"] is True, sweep.get("error")
    assert len(sweep["points"]) == 9

    vitals = probe.vitals({"row": 3, "col": 4})
    assert vitals["ok"] is True, vitals.get("error")
    assert vitals["summary"]["n_layers"] > 0
    assert vitals["summary"]["total_params"] > 0

    conv = next(layer["name"] for layer in layers["layers"] if layer["type"] == "Conv2d")
    assert probe.features_png(3, 4, conv)[:8] == PNG_MAGIC
    assert probe.kernels_png(conv)[:8] == PNG_MAGIC
