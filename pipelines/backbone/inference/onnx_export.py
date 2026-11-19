
from __future__ import annotations

from pathlib import Path

import torch

from models                                     import get_backbone
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.model.model_builder       import ModelBuilder
from tools.data.io                              import DatasetFingerprint, FileIO
from tools.monitoring.logger                    import Logger


class OnnxRunExporter:

    CHECKPOINT = "best_model.pt"

    def __init__(self, entry, logger: Logger) -> None:
        self.entry  = entry
        self.logger = logger

    def _discover(self) -> list[Path]:
        root = Path(self.entry.runs_dir)

        run_dirs = []
        for checkpoint in sorted(root.rglob(self.CHECKPOINT)):
            run_dir = checkpoint.parent
            name    = run_dir.relative_to(root).as_posix()

            if self.entry.run_filter and not any(token in name for token in self.entry.run_filter):
                continue
            if not (run_dir / "meta" / "run_summary.json").is_file():
                self.logger.warning(f"Skipping {name}: no meta/run_summary.json beside the checkpoint")
                continue

            run_dirs.append(run_dir)

        return run_dirs

    def _rebuild_model(self, run_dir: Path) -> tuple[torch.nn.Module, dict, tuple[int, int]]:
        meta_dir = run_dir / "meta"
        summary  = FileIO.load_json(meta_dir / "run_summary.json")
        creation = FileIO.load_json(meta_dir / "dataset_creation_config.json")

        patch_size = tuple(int(v) for v in creation["patch"]["size"])

        model_config, _ = BackboneModelConfigIO.load(meta_dir)

        overrides = {"in_channels": int(summary["in_channels"]), "out_channels": int(summary["out_channels"])}
        overrides.update(ModelBuilder.image_size_override(summary["model_name"], patch_size))

        model, _ = get_backbone(summary["model_name"], config=model_config, **overrides)

        checkpoint = torch.load(str(run_dir / self.CHECKPOINT), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["params"])
        model.eval()

        summary["checkpoint_epoch"]    = int(checkpoint["epoch"])
        summary["checkpoint_val_loss"] = float(checkpoint["best_val_loss"])

        return model, summary, patch_size

    def _export_run(self, run_dir: Path) -> Path:
        model, summary, patch_size = self._rebuild_model(run_dir)

        export_dir = FileIO.ensure_dir(run_dir / "export")
        onnx_path  = export_dir / "model.onnx"
        dummy      = torch.zeros(1, int(summary["in_channels"]), *patch_size)

        dynamic_axes = {"input": {0: "batch"}, "params": {0: "batch"}} if self.entry.dynamic_batch else None

        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names   = ["input"],
            output_names  = ["params"],
            dynamic_axes  = dynamic_axes,
            opset_version = int(self.entry.opset),
            dynamo        = False,
        )

        manifest = {
            "model_name"          : summary["model_name"],
            "opset"               : int(self.entry.opset),
            "dynamic_batch"       : bool(self.entry.dynamic_batch),
            "input_shape"         : [1, int(summary["in_channels"]), *patch_size],
            "out_channels"        : int(summary["out_channels"]),
            "checkpoint"          : self.CHECKPOINT,
            "checkpoint_epoch"    : summary["checkpoint_epoch"],
            "checkpoint_val_loss" : summary["checkpoint_val_loss"],
            "onnx"                : DatasetFingerprint.digest(onnx_path),
        }
        FileIO.save_json(manifest, export_dir / "manifest.json")

        return onnx_path

    def run(self) -> list[Path]:
        run_dirs = self._discover()

        paths = []
        for run_dir in run_dirs:
            path = self._export_run(run_dir)
            paths.append(path)
            self.logger.info(f"Exported {run_dir.name} -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

        self.logger.section("[ONNX export]")
        self.logger.kv_table({"Runs exported": len(paths), "Opset": self.entry.opset})

        return paths
