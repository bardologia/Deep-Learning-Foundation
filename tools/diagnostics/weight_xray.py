"""Orchestration of the model weight x-ray diagnostic.

Resolves a parameter tensor mapping out of a checkpoint, analyses every tensor,
flags anomalies, renders the diagnostic plots and writes the markdown and JSON
reports, either for a single run or for a batch of matched runs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from configuration.diagnostics           import WeightXrayConfig, WeightXrayEntryConfig
from tools.diagnostics.weight_xray_plots  import WeightXrayPlots
from tools.diagnostics.weight_xray_report import WeightXrayReport
from tools.runtime.run_selector           import RunSelector

from .weight_xray_analysis import IssueDetector, LayerReport, WeightAnalyzer, XraySummarizer


class StateDictResolver:
    """Locates the parameter tensor mapping inside a saved checkpoint.

    Attributes:
        config : Weight x-ray configuration holding the checkpoint path and the candidate state-dict keys.
    """

    def __init__(self, config: WeightXrayConfig) -> None:
        """Initialises the resolver.

        Args:
            config : Weight x-ray configuration for one checkpoint.
        """
        self.config = config

    def _load_object(self, path: Path):
        """Loads the checkpoint file onto CPU.

        Args:
            path : Checkpoint file to load.

        Returns:
            The unpickled checkpoint object.

        Raises:
            FileNotFoundError : If no file exists at the path.
        """
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint file at {path}; point --checkpoint-path at a .pt file or a run directory containing {self.config.checkpoint_filename}")

        return torch.load(str(path), map_location="cpu", weights_only=False)

    def _is_tensor_mapping(self, candidate) -> bool:
        """Returns whether a candidate is a non-empty dict whose values are mostly tensors."""
        if not isinstance(candidate, dict) or not candidate:
            return False

        tensor_values = sum(1 for value in candidate.values() if torch.is_tensor(value))
        return tensor_values >= max(1, int(0.5 * len(candidate)))

    def _select_mapping(self, obj) -> dict:
        """Returns the tensor mapping held by a checkpoint object.

        Args:
            obj : Unpickled checkpoint object.

        Returns:
            The parameter tensor mapping, taken from the top level, from one of
            the configured state-dict keys, or from any nested value.

        Raises:
            ValueError : If no tensor mapping can be located.
        """
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
        """Returns the parameter tensor mapping of the configured checkpoint."""
        path = self.config.resolved_checkpoint_path
        obj  = self._load_object(path)
        return self._select_mapping(obj)


class WeightXray:
    """Runs the full weight x-ray for one checkpoint.

    Attributes:
        config : Weight x-ray configuration for the checkpoint.
        logger : Logger used for progress reporting.
    """

    def __init__(self, config: WeightXrayConfig, logger) -> None:
        """Initialises the x-ray.

        Args:
            config : Weight x-ray configuration for one checkpoint.
            logger : Logger used for progress reporting.
        """
        self.config = config
        self.logger = logger

    def _resolve(self) -> dict:
        """Loads the checkpoint and returns its parameter tensor mapping."""
        self.logger.subsection(f"Loading checkpoint: {self.config.resolved_checkpoint_path}")
        state_dict = StateDictResolver(self.config).resolve()
        self.logger.ok(f"Resolved {len(state_dict)} tensors")
        return state_dict

    def _analyze(self, state_dict: dict) -> list[LayerReport]:
        """Computes the per-tensor statistics of every analysable parameter tensor.

        Every check a size limit skipped is reported with the tensors it left
        unchecked, so a clean verdict can be read against what was not tested.
        """
        reports = WeightAnalyzer(self.config).analyze(state_dict)
        self.logger.ok(f"Analysed {len(reports)} parameter tensors")

        for check in XraySummarizer.CHECKS_WITH_LIMITS:
            skipped = [report for report in reports if check in report.skipped_checks]
            if skipped:
                named = ", ".join(f"{report.name} ({report.skipped_checks[check]})" for report in skipped)
                self.logger.warning(f"{check} check skipped for {len(skipped)} of {len(reports)} tensors, which count as unchecked rather than clean: {named}")

        return reports

    def _detect(self, reports: list[LayerReport], state_dict: dict) -> list[LayerReport]:
        """Attaches threshold-based issues to every layer report."""
        return IssueDetector(self.config.thresholds).run(reports, state_dict)

    def _summarize(self, reports: list[LayerReport]) -> dict:
        """Returns the checkpoint-level summary of the layer reports."""
        return XraySummarizer().build(reports, self.config.resolved_checkpoint_path)

    def _plot(self, reports: list[LayerReport], state_dict: dict) -> list[Path]:
        """Renders the diagnostic plots, or returns an empty list when plotting is disabled."""
        if not self.config.make_plots:
            return []

        return WeightXrayPlots(self.config).render(reports, state_dict)

    def _report(self, reports: list[LayerReport], summary: dict, plot_paths: list[Path]) -> dict:
        """Writes the console, markdown and JSON reports and returns the output paths."""
        return WeightXrayReport(self.config).write(self.logger, reports, summary, plot_paths)

    def run(self) -> dict:
        """Runs resolve, analysis, detection, plotting and reporting in sequence.

        Returns:
            Mapping with the checkpoint summary under "summary" and the written
            report paths under "outputs".
        """
        self.logger.section("Model weight x-ray")

        state_dict = self._resolve()
        reports    = self._analyze(state_dict)
        reports    = self._detect(reports, state_dict)
        summary    = self._summarize(reports)
        plot_paths = self._plot(reports, state_dict)
        outputs    = self._report(reports, summary, plot_paths)

        return {"summary": summary, "outputs": outputs}


class WeightXrayBatch:
    """Runs the weight x-ray over every run matched by an entry-point filter.

    Attributes:
        entry_config : Entry configuration holding the run root and filter.
        logger       : Logger used for progress reporting.
    """

    def __init__(self, entry_config: WeightXrayEntryConfig, logger) -> None:
        """Initialises the batch x-ray.

        Args:
            entry_config : Entry configuration holding the run root and filter.
            logger       : Logger used for progress reporting.
        """
        self.entry_config = entry_config
        self.logger       = logger

    def _select_runs(self) -> list[Path]:
        """Returns the run directories that carry a checkpoint and match the filter."""
        selector = RunSelector(self.entry_config.runs_dir, self.entry_config.checkpoint_filename, self.logger, action="x-ray")
        return selector.resolve(self.entry_config.run_filter)

    def _xray_run(self, run_dir: Path) -> dict:
        """X-rays one run's checkpoint and returns its summary and output paths."""
        config = self.entry_config.to_config(run_dir)
        result = WeightXray(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['summary']['verdict']} -> {config.report_directory}")
        return result

    def run(self) -> list[dict]:
        """X-rays every selected run and returns one result mapping per run."""
        run_dirs = self._select_runs()
        return [self._xray_run(run_dir) for run_dir in run_dirs]
