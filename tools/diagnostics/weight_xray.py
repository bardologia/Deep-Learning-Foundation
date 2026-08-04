from __future__ import annotations

from pathlib import Path

import torch

from configuration.diagnostics           import WeightXrayConfig, WeightXrayEntryConfig
from tools.diagnostics.weight_xray_plots  import WeightXrayPlots
from tools.diagnostics.weight_xray_report import WeightXrayReport
from tools.runtime.run_selector           import RunSelector

from .weight_xray_analysis import IssueDetector, LayerReport, WeightAnalyzer, XraySummarizer


class StateDictResolver:
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

    def _select_mapping(self, obj) -> dict:
        if self._is_tensor_mapping(obj):
            return obj

        if isinstance(obj, dict):
            for key in self.config.state_dict_keys:
                if key in obj and self._is_tensor_mapping(obj[key]):
                    return obj[key]

            for value in obj.values():
                if self._is_tensor_mapping(value):
                    return value

        raise ValueError(f"Could not locate a parameter tensor mapping in the checkpoint; searched top level and keys {self.config.state_dict_keys}. Found type {type(obj).__name__}")

    def resolve(self) -> dict:
        path = self.config.resolved_checkpoint_path
        obj  = self._load_object(path)
        return self._select_mapping(obj)


class WeightXray:
    def __init__(self, config: WeightXrayConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def _resolve(self) -> dict:
        self.logger.subsection(f"Loading checkpoint: {self.config.resolved_checkpoint_path}")
        state_dict = StateDictResolver(self.config).resolve()
        self.logger.ok(f"Resolved {len(state_dict)} tensors")
        return state_dict

    def _analyze(self, state_dict: dict) -> list[LayerReport]:
        reports = WeightAnalyzer(self.config).analyze(state_dict)
        self.logger.ok(f"Analysed {len(reports)} parameter tensors")
        return reports

    def _detect(self, reports: list[LayerReport], state_dict: dict) -> list[LayerReport]:
        return IssueDetector(self.config.thresholds).run(reports, state_dict)

    def _summarize(self, reports: list[LayerReport]) -> dict:
        return XraySummarizer().build(reports, self.config.resolved_checkpoint_path)

    def _plot(self, reports: list[LayerReport], state_dict: dict) -> list[Path]:
        if not self.config.make_plots:
            return []

        return WeightXrayPlots(self.config).render(reports, state_dict)

    def _report(self, reports: list[LayerReport], summary: dict, plot_paths: list[Path]) -> dict:
        return WeightXrayReport(self.config).write(self.logger, reports, summary, plot_paths)

    def run(self) -> dict:
        self.logger.section("Model weight x-ray")

        state_dict = self._resolve()
        reports    = self._analyze(state_dict)
        reports    = self._detect(reports, state_dict)
        summary    = self._summarize(reports)
        plot_paths = self._plot(reports, state_dict)
        outputs    = self._report(reports, summary, plot_paths)

        return {"summary": summary, "outputs": outputs}


class WeightXrayBatch:
    def __init__(self, entry_config: WeightXrayEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.entry_config.runs_dir, self.entry_config.checkpoint_filename, self.logger, action="x-ray")
        return selector.resolve(self.entry_config.run_filter)

    def _xray_run(self, run_dir: Path) -> dict:
        config = self.entry_config.to_config(run_dir)
        result = WeightXray(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['summary']['verdict']} -> {config.report_directory}")
        return result

    def run(self) -> list[dict]:
        run_dirs = self._select_runs()
        return [self._xray_run(run_dir) for run_dir in run_dirs]
