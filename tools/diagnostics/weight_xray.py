"""Orchestration of the model weight x-ray diagnostic.

Resolves a parameter tensor mapping out of a checkpoint, analyses every tensor,
flags anomalies, renders the diagnostic plots and writes the markdown and JSON
reports, either for a single run or for a batch of matched runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path
from typing      import Optional

import torch

from configuration.diagnostics           import WeightXrayConfig, WeightXrayEntryConfig
from tools.diagnostics.weight_xray_plots  import WeightXrayPlots
from tools.diagnostics.weight_xray_report import WeightXrayReport
from tools.runtime.config_cli             import ConfigCli
from tools.runtime.git_state              import RunProvenance
from tools.runtime.run_selector           import RunSelector

from .weight_xray_analysis import IssueDetector, LayerReport, WeightAnalyzer, XraySummarizer


@dataclass
class ResolvedCheckpoint:
    """Parameter tensors located inside a checkpoint, with where they came from.

    Attributes:
        path       : Checkpoint file the tensors were read from.
        key        : Checkpoint entry holding the tensors, or "(top level)" when the file is the mapping itself.
        state_dict : Mapping of parameter name to tensor.
        epoch      : Zero-based epoch stored beside the tensors, or None when the checkpoint records none.
        best_epoch : Zero-based epoch of the best validation loss stored beside the tensors, or None.
    """
    path       : Path
    key        : str
    state_dict : dict
    epoch      : Optional[int]
    best_epoch : Optional[int]


class StateDictResolver:
    """Locates the parameter tensor mapping inside a saved checkpoint.

    Attributes:
        config    : Weight x-ray configuration holding the checkpoint path and the candidate state-dict keys.
        TOP_LEVEL : Entry name reported when the checkpoint file is the tensor mapping itself.
    """

    TOP_LEVEL = "(top level)"

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

    def _select_mapping(self, obj, path: Path) -> tuple[str, dict]:
        """Returns the checkpoint entry name and the tensor mapping it holds.

        Args:
            obj  : Unpickled checkpoint object.
            path : Checkpoint file, named in the error.

        Returns:
            Pair of the entry key and the parameter tensor mapping; the key is
            TOP_LEVEL when the checkpoint is the mapping itself.

        Raises:
            ValueError : If the checkpoint is no tensor mapping and none of the configured state-dict keys holds one. The error names the entries that do hold tensors, so the caller picks one explicitly rather than the x-ray silently analysing the first mapping it meets, which for a trainer state would be the raw weights rather than the EMA weights inference uses.
        """
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
        """Returns the integer epoch entry stored beside the tensors, or None when the checkpoint has none."""
        if isinstance(obj, dict) and name in obj:
            return int(obj[name])
        return None

    def resolve(self) -> ResolvedCheckpoint:
        """Returns the parameter tensor mapping of the configured checkpoint with its entry and epochs."""
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

    @staticmethod
    def _epoch_text(resolved: ResolvedCheckpoint) -> str:
        """Returns the stored epoch indices as a phrase, or says the checkpoint stores none."""
        if resolved.epoch is None and resolved.best_epoch is None:
            return "no epoch stored"
        return f"epoch index {resolved.epoch}, best epoch index {resolved.best_epoch}"

    def _resolve(self) -> ResolvedCheckpoint:
        """Loads the checkpoint and returns its parameter tensor mapping, logging which entry it came from."""
        self.logger.subsection(f"Loading checkpoint: {self.config.resolved_checkpoint_path}")
        resolved = StateDictResolver(self.config).resolve()
        self.logger.ok(f"Resolved {len(resolved.state_dict)} tensors from checkpoint entry '{resolved.key}' ({self._epoch_text(resolved)})")
        return resolved

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

    def _provenance(self, resolved: ResolvedCheckpoint) -> dict:
        """Returns what produced this report, so a stored verdict can be tied to its thresholds, checkpoint epoch and code.

        Args:
            resolved : Checkpoint the tensors were read from.

        Returns:
            Mapping with the creation time, the checkpoint path, entry and
            stored epochs, every effective configuration leaf and the commit,
            host, interpreter and library versions of this process.
        """
        return {
            "created"    : datetime.now().isoformat(timespec="seconds"),
            "checkpoint" : {"path": str(resolved.path), "key": resolved.key, "epoch": resolved.epoch, "best_epoch": resolved.best_epoch},
            "config"     : ConfigCli.to_mapping(self.config),
            **RunProvenance.collect(),
        }

    def _plot(self, reports: list[LayerReport], state_dict: dict) -> list[Path]:
        """Renders the diagnostic plots, or returns an empty list when plotting is disabled."""
        if not self.config.make_plots:
            return []

        return WeightXrayPlots(self.config).render(reports, state_dict)

    def _report(self, reports: list[LayerReport], summary: dict, provenance: dict, plot_paths: list[Path]) -> dict:
        """Writes the console, markdown and JSON reports and returns the output paths."""
        return WeightXrayReport(self.config).write(self.logger, reports, summary, provenance, plot_paths)

    def run(self) -> dict:
        """Runs resolve, analysis, detection, plotting and reporting in sequence.

        Returns:
            Mapping with the checkpoint summary under "summary", the report
            provenance under "provenance" and the written report paths under
            "outputs".
        """
        resolved   = self._resolve()
        state_dict = resolved.state_dict

        reports    = self._analyze(state_dict)
        reports    = self._detect(reports, state_dict)
        summary    = self._summarize(reports)
        provenance = self._provenance(resolved)
        plot_paths = self._plot(reports, state_dict)
        outputs    = self._report(reports, summary, provenance, plot_paths)

        return {"summary": summary, "provenance": provenance, "outputs": outputs}


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

    def _log_configuration(self) -> None:
        """Logs every effective entry setting once, so the thresholds behind each verdict survive in the log."""
        self.logger.kv_table(ConfigCli.to_mapping(self.entry_config), title="Weight x-ray configuration")

    def _select_runs(self) -> list[Path]:
        """Returns the run directories that carry a checkpoint and match the filter."""
        selector = RunSelector(self.entry_config.runs_dir, self.entry_config.checkpoint_filename, self.logger, action="x-ray")
        return selector.resolve(self.entry_config.run_filter)

    def _run_label(self, run_dir: Path) -> str:
        """Returns the run's path below the runs root, which tells nested seed runs apart."""
        return str(Path(run_dir).relative_to(Path(self.entry_config.runs_dir)))

    def _xray_run(self, run_dir: Path, position: int, total: int) -> dict:
        """X-rays one run's checkpoint under a numbered section and returns its summary and output paths.

        Args:
            run_dir  : Run directory holding the checkpoint.
            position : One-based position of the run in the batch.
            total    : Number of runs in the batch.

        Returns:
            The WeightXray result mapping of the run.
        """
        self.logger.section(f"Weight x-ray [{position}/{total}] {self._run_label(run_dir)}")

        config = self.entry_config.to_config(run_dir)
        result = WeightXray(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['summary']['verdict']} -> {config.report_directory}")
        return result

    def _summarize(self, run_dirs: list[Path], results: list[dict]) -> None:
        """Closes the batch with one row per run and the verdict counts, so no verdict has to be grepped out of the log.

        Args:
            run_dirs : Run directories in batch order.
            results  : WeightXray result mappings in the same order.
        """
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
        """Logs the configuration, x-rays every selected run and closes with the batch summary.

        Returns:
            One result mapping per run, in selection order.
        """
        self._log_configuration()

        run_dirs = self._select_runs()
        results  = [self._xray_run(run_dir, position, len(run_dirs)) for position, run_dir in enumerate(run_dirs, start=1)]

        self._summarize(run_dirs, results)
        return results
