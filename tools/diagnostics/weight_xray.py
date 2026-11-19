
from __future__ import annotations

from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path
from typing      import Optional

import torch

from configuration.diagnostics                  import WeightXrayConfig, WeightXrayEntryConfig
from models                                     import get_backbone
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from tools.diagnostics.weight_xray_plots         import WeightXrayPlots
from tools.diagnostics.weight_xray_report        import WeightXrayReport
from tools.runtime.config_cli                    import ConfigCli
from tools.runtime.git_state                     import RunProvenance
from tools.runtime.run_selector                  import RunSelector

from .weight_xray_analysis import IssueDetector, LayerReport, ModuleTypes, WeightAnalyzer, XraySummarizer


@dataclass
class ResolvedCheckpoint:
    path       : Path
    key        : str
    state_dict : dict
    epoch      : Optional[int]
    best_epoch : Optional[int]


class StateDictResolver:

    TOP_LEVEL = "(top level)"

    def __init__(self, config: WeightXrayConfig) -> None:
        self.config = config

    def _load_object(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint file at {path}; point --checkpoint-path at a .pt file or a run directory containing {self.config.checkpoint_filename}")

        return torch.load(str(path), map_location="cpu", weights_only=False)

    def _is_tensor_mapping(self, candidate) -> bool:
        if not isinstance(candidate, dict) or not candidate:
            return False

        tensor_values = sum(1 for value in candidate.values() if torch.is_tensor(value))
        return tensor_values >= max(1, int(0.5 * len(candidate)))

    def _select_mapping(self, obj, path: Path) -> tuple[str, dict]:
        if self._is_tensor_mapping(obj):
            return self.TOP_LEVEL, obj

        if isinstance(obj, dict):
            for key in self.config.state_dict_keys:
                if key in obj and self._is_tensor_mapping(obj[key]):
                    return key, obj[key]

            candidates = [key for key, value in obj.items() if self._is_tensor_mapping(value)]
            if candidates:
                raise ValueError(f"Checkpoint {path} holds no tensor mapping under the configured state_dict_keys {self.config.state_dict_keys}, but does under {candidates}; name the entry to x-ray in state_dict_keys instead of letting the x-ray guess between them")

            raise ValueError(f"Checkpoint {path} holds no parameter tensor mapping at the top level, under state_dict_keys {self.config.state_dict_keys} or under any other entry; its entries are {list(obj)}")

        raise ValueError(f"Checkpoint {path} is a {type(obj).__name__}, not a tensor mapping or a dict holding one under state_dict_keys {self.config.state_dict_keys}")

    @staticmethod
    def _stored_epoch(obj, name: str) -> Optional[int]:
        if isinstance(obj, dict) and name in obj:
            return int(obj[name])
        return None

    def resolve(self) -> ResolvedCheckpoint:
        path            = self.config.resolved_checkpoint_path
        obj             = self._load_object(path)
        key, state_dict = self._select_mapping(obj, path)

        return ResolvedCheckpoint(
            path       = path,
            key        = key,
            state_dict = state_dict,
            epoch      = self._stored_epoch(obj, "epoch"),
            best_epoch = self._stored_epoch(obj, "best_epoch"),
        )


class ModuleTypeResolver:

    TRANSPOSED = (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)

    def __init__(self, checkpoint_path: Path) -> None:
        self.meta_directory = Path(checkpoint_path).parent / "meta"

    def available(self) -> bool:
        return BackboneModelConfigIO.exists(self.meta_directory)

    def resolve(self) -> ModuleTypes:
        config, name = BackboneModelConfigIO.load(self.meta_directory)
        model, _     = get_backbone(name, config=config)

        by_key     = {}
        transposed = set()

        for module_name, module in model.named_modules():
            owned = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))

            for tensor_name, _tensor in owned:
                key         = f"{module_name}.{tensor_name}" if module_name else tensor_name
                by_key[key] = type(module).__name__

                if isinstance(module, self.TRANSPOSED) and tensor_name == "weight":
                    transposed.add(key)

        return ModuleTypes(architecture=str(name), by_key=by_key, transposed=transposed)


class WeightXray:

    def __init__(self, config: WeightXrayConfig, logger) -> None:
        self.config = config
        self.logger = logger

    @staticmethod
    def _epoch_text(resolved: ResolvedCheckpoint) -> str:
        if resolved.epoch is None and resolved.best_epoch is None:
            return "no epoch stored"
        return f"epoch index {resolved.epoch}, best epoch index {resolved.best_epoch}"

    def _resolve(self) -> ResolvedCheckpoint:
        self.logger.subsection(f"Loading checkpoint: {self.config.resolved_checkpoint_path}")
        resolved = StateDictResolver(self.config).resolve()
        self.logger.ok(f"Resolved {len(resolved.state_dict)} tensors from checkpoint entry '{resolved.key}' ({self._epoch_text(resolved)})")
        return resolved

    def _module_types(self, resolved: ResolvedCheckpoint) -> ModuleTypes:
        resolver = ModuleTypeResolver(resolved.path)

        if not resolver.available():
            self.logger.warning(f"No {BackboneModelConfigIO.FILENAME} under {resolver.meta_directory}; module types are unknown, so every weight is read with its output units on dim 0 and a transposed convolution reports its input channels as output units")
            return ModuleTypes.unavailable()

        types   = resolver.resolve()
        missing = [key for key in resolved.state_dict if key not in types.by_key]

        if missing:
            self.logger.warning(f"{len(missing)} checkpoint tensors have no owning module in the rebuilt {types.architecture} and are read on dim 0: {', '.join(missing)}")

        self.logger.ok(f"Resolved module types from the persisted {types.architecture}: {len(types.transposed)} transposed-convolution weights read their output units on dim 1")
        return types

    def _analyze(self, state_dict: dict, module_types: ModuleTypes) -> list[LayerReport]:
        reports = WeightAnalyzer(self.config, module_types).analyze(state_dict)
        self.logger.ok(f"Analysed {len(reports)} parameter tensors")

        for check in XraySummarizer.CHECKS_WITH_LIMITS:
            skipped = [report for report in reports if check in report.skipped_checks]
            if skipped:
                named = ", ".join(f"{report.name} ({report.skipped_checks[check]})" for report in skipped)
                self.logger.warning(f"{check} check skipped for {len(skipped)} of {len(reports)} tensors, which count as unchecked rather than clean: {named}")

        return reports

    def _detect(self, reports: list[LayerReport], state_dict: dict) -> list[LayerReport]:
        return IssueDetector(self.config.thresholds).run(reports, state_dict)

    def _summarize(self, reports: list[LayerReport]) -> dict:
        return XraySummarizer().build(reports, self.config.resolved_checkpoint_path)

    def _provenance(self, resolved: ResolvedCheckpoint, module_types: ModuleTypes) -> dict:
        return {
            "created"      : datetime.now().isoformat(timespec="seconds"),
            "checkpoint"   : {"path": str(resolved.path), "key": resolved.key, "epoch": resolved.epoch, "best_epoch": resolved.best_epoch},
            "architecture" : module_types.architecture,
            "config"       : ConfigCli.to_mapping(self.config),
            **RunProvenance.collect(),
        }

    def _plot(self, reports: list[LayerReport], state_dict: dict) -> list[Path]:
        if not self.config.make_plots:
            return []

        return WeightXrayPlots(self.config).render(reports, state_dict)

    def _report(self, reports: list[LayerReport], summary: dict, provenance: dict, plot_paths: list[Path]) -> dict:
        return WeightXrayReport(self.config).write(self.logger, reports, summary, provenance, plot_paths)

    def run(self) -> dict:
        resolved     = self._resolve()
        state_dict   = resolved.state_dict
        module_types = self._module_types(resolved)

        reports    = self._analyze(state_dict, module_types)
        reports    = self._detect(reports, state_dict)
        summary    = self._summarize(reports)
        provenance = self._provenance(resolved, module_types)
        plot_paths = self._plot(reports, state_dict)
        outputs    = self._report(reports, summary, provenance, plot_paths)

        return {"summary": summary, "provenance": provenance, "outputs": outputs}


class WeightXrayBatch:

    def __init__(self, entry_config: WeightXrayEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger

    def _log_configuration(self) -> None:
        self.logger.kv_table(ConfigCli.to_mapping(self.entry_config), title="Weight x-ray configuration")

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.entry_config.runs_dir, self.entry_config.checkpoint_filename, self.logger, action="x-ray")
        return selector.resolve(self.entry_config.run_filter)

    def _run_label(self, run_dir: Path) -> str:
        return str(Path(run_dir).relative_to(Path(self.entry_config.runs_dir)))

    def _xray_run(self, run_dir: Path, position: int, total: int) -> dict:
        self.logger.section(f"Weight x-ray [{position}/{total}] {self._run_label(run_dir)}")

        config = self.entry_config.to_config(run_dir)
        result = WeightXray(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['summary']['verdict']} -> {config.report_directory}")
        return result

    def _summarize(self, run_dirs: list[Path], results: list[dict]) -> None:
        rows = []
        for run_dir, result in zip(run_dirs, results):
            summary = result["summary"]
            rows.append({
                "Run"       : self._run_label(run_dir),
                "Verdict"   : summary["verdict"],
                "Flagged"   : summary["flagged_tensors"],
                "Critical"  : summary["severity_counts"]["critical"],
                "Warning"   : summary["severity_counts"]["warning"],
                "Unchecked" : sum(summary["unchecked"].values()),
                "Output"    : str(result["outputs"]["report"]),
            })

        self.logger.metrics_table(rows, columns=["Run", "Verdict", "Flagged", "Critical", "Warning", "Unchecked", "Output"], title="Weight x-ray summary")

        verdicts = [result["summary"]["verdict"] for result in results]
        counted  = ", ".join(f"{verdicts.count(verdict)} {verdict}" for verdict in XraySummarizer.VERDICTS)
        self.logger.ok(f"X-rayed {len(results)} run(s): {counted}")

    def run(self) -> list[dict]:
        self._log_configuration()

        run_dirs = self._select_runs()
        results  = [self._xray_run(run_dir, position, len(run_dirs)) for position, run_dir in enumerate(run_dirs, start=1)]

        self._summarize(run_dirs, results)
        return results
