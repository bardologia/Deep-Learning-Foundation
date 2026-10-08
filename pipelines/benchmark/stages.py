"""Stages of the architecture benchmark: batch probing, sizing, training and comparison.

Each stage owns its own artefact paths and resume rules; the seed-expansion mixin
turns the selected models and loss components into the per-seed units the queued
training and inference stages consume, and matches them to the run directories
their manifests identify when a run is resumed.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib     import Path

from configuration.benchmark                       import BenchmarkConfig
from configuration.dataset.gaussian             import GaussianConfig
from pipelines.backbone.training.loss_terms        import LossComponentCatalog
from pipelines.benchmark.results                   import BenchmarkSeedCollector
from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.benchmark.sizing                    import SizeMatcher, SizeMatchResult
from pipelines.shared.model.model_builder          import ModelBuilder
from pipelines.shared.orchestration.unit_manifest  import UnidentifiedRuns, UnitLedger, UnitManifest
from pipelines.shared.training.run_naming          import JepaNamingSpec, RunNaming
from pipelines.shared.training.seed_sweep          import SeedSet
from tools.orchestration                           import ExperimentStage, GpuJob, QueuedInferenceStage, QueuedTrainingStage
from tools.data.io                                 import FileIO
from tools.monitoring.logger                       import Logger
from tools.runtime.run_tag                         import RunTag


class BenchmarkUnitManifest:
    """Identity of one benchmark unit, as the training worker records it in the unit's run directory.

    Attributes:
        KIND : Manifest kind every benchmark unit is written with.
    """

    KIND = "benchmark"

    @staticmethod
    def identity(manifest: dict) -> tuple:
        """Returns the (model key, loss component, seed) triple a manifest identifies."""
        return (manifest["model"], manifest["loss_component"], manifest["seed"])

    @classmethod
    def write(cls, run_directory: Path, run_tag: str, unit: str, model: str, loss_component: str | None, seed: int | None) -> Path:
        """Writes the manifest of one unit before its training starts.

        Args:
            run_directory  : Run directory of the unit.
            run_tag        : Tag of the benchmark run the unit belongs to.
            unit           : Run name of the unit, relative to the training directory.
            model          : Model key of the unit.
            loss_component : Loss component the unit isolates, or None.
            seed           : Seed of the unit, or None when the sweep has no seed dimension.

        Returns:
            Path of the written manifest.
        """
        payload = {
            "run_tag"        : run_tag,
            "unit"           : unit,
            "model"          : model,
            "loss_component" : loss_component,
            "seed"           : seed,
        }

        return UnitManifest.write(run_directory, cls.KIND, payload)


class SeedExpandedStage:
    """Mixin expanding models and loss components into named per-seed units.

    Unit names double as run directory names for fresh units, and are mapped
    back to their model, loss component and seed when the worker command line
    is built. On a resumed run every unit is matched to the run directory whose
    manifest records its identity, whatever that directory is called.

    Attributes:
        unit_dirs    : Existing directory per unit name, relative to the training directory, for the units a resume found through their manifests.
        unidentified : Run directories under the training directory that carry no benchmark manifest and are left untouched.
    """

    _N_GAUSSIANS : dict = {}

    @classmethod
    def components(cls, config: BenchmarkConfig) -> list[str | None]:
        """Returns the loss components swept per model, or a single None for non-backbone types.

        Args:
            config : Benchmark configuration carrying `sweep_loss_components`.

        Returns:
            The validated component names, or `[None]` when the training type has
            no loss-component axis.

        Raises:
            SystemExit : If the component list is empty or names an unregistered loss component.
        """
        if config.training_type != "backbone":
            return [None]

        components = list(config.sweep_loss_components)
        if not components:
            raise SystemExit("sweep_loss_components is empty; select at least one loss component to sweep")

        valid   = set(LossComponentCatalog.names())
        unknown = [component for component in components if component not in valid]
        if unknown:
            raise SystemExit(f"unknown loss component(s) {unknown}; valid: {', '.join(sorted(valid))}")

        return components

    @classmethod
    def n_gaussians(cls, config: BenchmarkConfig) -> int:
        """Returns the dataset's default Gaussian slot count, cached per dataset path pair."""
        key = (str(config.paths.dataset_path), str(config.paths.parameters_path))

        if key not in cls._N_GAUSSIANS:
            cls._N_GAUSSIANS[key] = GaussianConfig.from_dataset(config.paths.dataset_path, config.paths.parameters_path).n_default_gaussians

        return cls._N_GAUSSIANS[key]

    @classmethod
    def unit_base(cls, config: BenchmarkConfig, model: str, component: str | None) -> str:
        """Returns the seed-free run name of one model and loss component.

        Args:
            config    : Benchmark configuration determining the naming scheme.
            model     : Model key.
            component : Loss component swept for this unit, or None.

        Returns:
            The run name, following the run-naming convention of the training type.
        """
        if config.training_type == "backbone":
            return RunNaming.benchmark_unit(model, component, config.loss, cls.n_gaussians(config), config.augmentation)
        if config.training_type == "jepa":
            name, head = ModelBuilder.split_key(model)
            return RunNaming.jepa_tag(name, head, JepaNamingSpec.resolve(config.jepa), cls.n_gaussians(config), config.augmentation)

        return model if component is None else f"{model}__{component}"

    @classmethod
    def planned_units(cls, config: BenchmarkConfig, models: list[str]) -> list[tuple[str, tuple]]:
        """Returns every unit a benchmark configuration plans, named and identified.

        The unit name is the directory a fresh unit is created under, and the
        identity is what its manifest records, so anything that has to line
        planned units up with directories on disk reads both from here.

        Args:
            config : Benchmark configuration carrying the loss components and seeds.
            models : Selected model keys.

        Returns:
            Pairs of unit name and its (model key, loss component, seed) identity.
        """
        components = cls.components(config)
        pairs      = [(model, component) for model in models for component in components]
        by_base    = {cls.unit_base(config, model, component): (model, component) for model, component in pairs}

        units = SeedSet.units(list(by_base.keys()), config.seeds)

        return [(name, (*by_base[base], seed)) for base, seed, name in units]

    def _expand(self, config: BenchmarkConfig, models: list[str]) -> list[str]:
        """Expands models and loss components into per-seed unit names, caching each unit's identity.

        Args:
            config : Benchmark configuration carrying the seeds.
            models : Selected model keys.

        Returns:
            The unit names, one per model, loss component and seed.
        """
        planned    = self.planned_units(config, models)
        self._unit = dict(planned)

        return [name for name, _ in planned]

    def _identity(self, name: str) -> tuple:
        """Returns the (model key, loss component, seed) identity of a planned unit."""
        return self._unit[name]

    def _reconcile(self, training_dir: Path) -> None:
        """Maps every planned unit onto the run directory its manifest identifies, when one exists.

        Args:
            training_dir : Directory holding the unit run directories.
        """
        ledger = UnitLedger(training_dir, BenchmarkUnitManifest.KIND, BenchmarkUnitManifest.identity).build()

        self.unit_dirs    = {}
        self.unidentified = list(ledger.unidentified)

        for name in self._unit:
            directory = ledger.directory(self._identity(name))
            if directory is not None:
                self.unit_dirs[name] = directory.relative_to(training_dir).as_posix()

    def _log_unidentified(self) -> None:
        """Warns, from a queued stage, about the run directories under its stage directory that no planned unit's manifest claims."""
        UnidentifiedRuns(self.stage_dir, self.unidentified, BenchmarkUnitManifest.KIND, {item: self.unit_dir(item) for item in self.items}).log(self.logger, type(self).__name__)

    def _worker_argv(self, action: str, name: str) -> list[str]:
        """Returns the worker arguments running `action` for the unit named `name`."""
        model, component, seed = self._identity(name)

        loss_args = ["--loss-component", component] if component is not None else []

        return ["--worker", action, "--model", model, *loss_args, *SeedSet.cli_args(seed), "--run-name", self.unit_name(name), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)]

    def _seed_command(self, action: str, name: str) -> list[str]:
        """Returns the full subprocess command running `action` for one unit."""
        return [sys.executable, str(self.entry_script), *self._worker_argv(action, name)]

    def _job(self, item: str) -> GpuJob:
        """Returns the GPU job running this stage's worker action for one unit."""
        return GpuJob(
            name     = item,
            command  = self._seed_command(self.worker_action, item),
            log_path = self.unit_dir(item) / self.worker_logname,
        )

    def _scored_splits(self) -> list[str]:
        """Returns the split the benchmark scores, empty when the family runs no inference."""
        return [self.config.inference.split] if self.config.runs_inference() else []


class MaxBatchStage(ExperimentStage):
    """Probes the maximum trainable batch size of every model on the GPU pool.

    Attributes:
        models       : Model keys to probe.
        stage_dir    : Directory holding the per-model probe outputs.
        records_path : JSON file collecting every probe record.
        report_path  : Markdown report of the probe results.
    """

    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        """Resolves the probe stage directories and artefact paths.

        Args:
            config       : Benchmark configuration.
            entry_script : Entry script the probe workers re-invoke.
            run_tag      : Tag identifying this benchmark run.
            models       : Model keys to probe.
            logger       : Logger for the stage.
        """
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.models       = models
        self.stage_dir    = self.run_dir / "max_batch"
        self.records_path = self.run_dir / "pipeline" / "max_batch.json"
        self.report_path  = self.run_dir / "pipeline" / "max_batch_report.md"

    def _result_path(self, model_name: str) -> Path:
        """Returns the JSON path of one model's probe result."""
        return self.stage_dir / model_name / "max_batch_result.json"

    def _cache_verdict(self, model_name: str) -> str | None:
        """Returns why a model's previous probe result cannot be reused, or None when it can.

        Args:
            model_name : Model key to check.

        Returns:
            None when resuming and the result file holds a PASS measured under
            the current probe settings; otherwise the reason it is re-probed:
            no previous result, resume off, a status other than PASS, a result
            predating the current settings schema, or a changed setting.
        """
        path = self._result_path(model_name)

        if not path.exists():
            return "no previous result"
        if not self.config.resume:
            return "resume is off"

        record = FileIO.load_json(path)

        if record["status"] != "PASS":
            return f"previous status was {record['status']}"

        settings = self.config.max_batch_settings()
        stored   = record["settings"]

        if set(stored) != set(settings):
            return "the result predates the current probe settings schema"

        changed = {key: (stored[key], value) for key, value in settings.items() if stored[key] != value}
        if changed:
            return f"measured under {changed} (stored, current)"

        return None

    def _has_result(self, model_name: str) -> bool:
        """Returns whether a cached passing result measured under the current settings exists for a model."""
        return self._cache_verdict(model_name) is None

    def _discard_result(self, model_name: str, reason: str) -> None:
        """Deletes a model's previous probe result before it is re-probed, saying why, so a worker that dies before writing leaves no stale file behind.

        Args:
            model_name : Model key about to be re-probed.
            reason     : Verdict explaining why the previous result is not reused.
        """
        path = self._result_path(model_name)
        if not path.exists():
            return

        self.logger.warning(f"{model_name}: previous probe result discarded ({reason}), re-probing")
        path.unlink()

    def _worker_log(self, model_name: str) -> Path:
        """Returns the log file of one model's probe worker."""
        return self.stage_dir / model_name / "worker.log"

    def _job(self, model_name: str) -> GpuJob:
        """Returns the GPU job probing the maximum batch size of one model."""
        return GpuJob(
            name     = model_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "maxbatch", "--model", model_name, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self._worker_log(model_name),
        )

    def _fail_record(self, model_name: str, error: str) -> dict:
        """Returns a FAIL probe record for a model whose probe left no usable result.

        Args:
            model_name : Model key the record stands for.
            error      : Text explaining the failure.
        """
        return {
            "model"                    : model_name,
            "status"                   : "FAIL",
            "batch_size"               : None,
            "peak_gb"                  : None,
            "budget_gb"                : self.config.max_batch.vram_budget_gb,
            "ceiling"                  : self.config.max_batch.max_batch,
            "context_gb"               : None,
            "trials"                   : [],
            "lr_multiplier"            : None,
            "lr_min_loss_multiplier"   : None,
            "lr_divergence_multiplier" : None,
            "lr_rungs_run"             : None,
            "error"                    : error,
            "settings"                 : self.config.max_batch_settings(),
        }

    def _load_result(self, model_name: str, job: dict | None) -> dict:
        """Loads one model's probe record, or synthesises a FAIL record when the probe left none.

        Args:
            model_name : Model key whose result is read.
            job        : GPU queue result of the probe worker this launch ran for the model, or None when the cached result was reused.

        Returns:
            The record on disk, turned into FAIL when it was measured under
            other probe settings than the current ones, or a FAIL record naming
            the worker exit and its log when the worker died before writing one.
        """
        path = self._result_path(model_name)

        if not path.exists():
            if job is None:
                return self._fail_record(model_name, f"missing result file: {path}")

            return self._fail_record(model_name, f"worker exited {job['returncode']} before writing a result, see {job['log_file']}")

        record   = FileIO.load_json(path)
        settings = self.config.max_batch_settings()

        if record["settings"] != settings:
            self.logger.error(f"{model_name}: probe result at {path} was measured under {record['settings']}, the current settings are {settings}; the model counts as not probed")
            return {**record, "status": "FAIL", "error": f"stale probe result measured under {record['settings']}; current settings {settings}"}

        return record

    def _write_report(self, records: dict) -> None:
        """Writes the markdown table of batch size, peak memory and status per model.

        Args:
            records : Probe record per model key.
        """
        budget  = self.config.max_batch.vram_budget_gb
        ceiling = self.config.max_batch.max_batch

        lines = [
            "# Maximum Batch Size Report",
            f"\n_Generated {RunTag.timestamp()}  —  run tag `{self.run_tag}`_\n",
            f"Largest power-of-two batch size whose peak training memory stays under **{budget:g} GB** and below **{ceiling}**.",
            f"Measured at patch size {tuple(self.config.training.patch_size)} with {self.config.size_match.in_channels} input channels.\n",
            "## Results\n",
            "| Model | Max batch | Peak (GB) | LR multiplier | Status | Error |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for model_name in self.models:
            record     = records[model_name]
            batch_size = record["batch_size"] if record["batch_size"] is not None else "—"
            peak_gb    = f"{record['peak_gb']:.2f}" if record.get("peak_gb") is not None else "—"
            multiplier = f"x{record['lr_multiplier']:.4g}" if record["lr_multiplier"] is not None else "—"
            error      = (record.get("error") or "—").strip().splitlines()[-1]
            lines.append(f"| `{model_name}` | {batch_size} | {peak_gb} | {multiplier} | {record['status']} | {error} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")

    def _log_summary(self, records: dict) -> None:
        """Logs how many models were measured, the batch size and peak of each, and why each failure failed.

        A model whose LR range test found no usable response is warned about,
        since it trains at the configured learning rate, and every failed
        model is listed with the last line of its error and its worker log.

        Args:
            records : Probe record per model key.
        """
        passed = [m for m in records if records[m]["status"] == "PASS"]
        failed = [m for m in records if records[m]["status"] != "PASS"]

        self.logger.subsection("Max batch summary")
        self.logger.kv_table({
            "Total"  : len(records),
            "Passed" : len(passed),
            "Failed" : len(failed),
        }, title=f"{len(passed)}/{len(records)} measured")

        for model_name in passed:
            multiplier = records[model_name]["lr_multiplier"]
            lr_note    = f"  lr x{multiplier:.4g}" if multiplier is not None else ""
            self.logger.info(f"{model_name:<18} batch {records[model_name]['batch_size']:>4}  peak {records[model_name]['peak_gb']:.2f} GB{lr_note}")

            if self.config.find_lr.enabled and multiplier is None:
                self.logger.warning(f"{model_name}: LR range test found no usable response; training keeps lr_multiplier={self.config.training.lr_multiplier}")

        for model_name in failed:
            reason = records[model_name]["error"].strip().splitlines()[-1]
            self.logger.error(f"FAILED  {model_name}: {reason}  (see {self._worker_log(model_name)})")

    def run(self) -> dict:
        """Probes every model that has no reusable cached result and reports the outcome.

        Returns:
            The probe record per model key, cached and freshly measured alike.
        """
        self.logger.section("Maximum batch size")
        self.logger.kv_table({
            "Models"        : len(self.models),
            "VRAM budget"   : f"{self.config.max_batch.vram_budget_gb:g} GB",
            "Ceiling"       : self.config.max_batch.max_batch,
            "Measure steps" : self.config.max_batch.measure_steps,
            "LR range test" : "enabled" if self.config.find_lr.enabled else "disabled",
            "GPUs"          : self.config.gpus,
        }, title="Configuration")

        if self.config.training.batch_size > self.config.max_batch.max_batch:
            self.logger.warning(f"training.batch_size={self.config.training.batch_size} exceeds the probe ceiling max_batch.max_batch={self.config.max_batch.max_batch}; the probed batch replaces the configured one, so training would be capped at the ceiling — raise max_batch.max_batch to probe up to the intended batch")

        verdicts = {m: self._cache_verdict(m) for m in self.models}
        cached   = [m for m in self.models if verdicts[m] is None]
        pending  = [m for m in self.models if verdicts[m] is not None]

        for model_name in cached:
            self.logger.info(f"{model_name}: cached result reused")

        for model_name in pending:
            self._discard_result(model_name, verdicts[model_name])

        jobs = {}
        if pending:
            jobs = {result["name"]: result for result in self._run_queue([self._job(m) for m in pending])}

        records = {m: self._load_result(m, jobs[m] if m in jobs else None) for m in self.models}

        self._write_results(records, self.records_path)
        self._write_report(records)
        self._log_summary(records)

        return records


class SizeMatchStage(ExperimentStage):
    """Matches every model's capacity to the reference model, in process.

    Attributes:
        models       : Model keys to match.
        records_path : JSON file collecting the per-model match records.
        report_path  : Markdown report of the matched widths.
    """

    def __init__(self, config: BenchmarkConfig, run_tag: str, models: list[str], logger: Logger) -> None:
        """Resolves the artefact paths of the capacity-matching stage.

        Args:
            config  : Benchmark configuration.
            run_tag : Tag identifying this benchmark run.
            models  : Model keys to match.
            logger  : Logger for the stage.
        """
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.models       = models
        self.records_path = self.run_dir / "pipeline" / "size_match.json"
        self.report_path  = self.run_dir / "pipeline" / "size_match_report.md"

    @staticmethod
    def resolve_reference(reference: str, models: list[str], heads: list[str]) -> str:
        """Expands a bare reference backbone name into a full model key when unambiguous.

        Args:
            reference : Configured reference model, with or without a head suffix.
            models    : Swept model keys.
            heads     : Benchmarked output heads.

        Returns:
            The reference as a swept model key.

        Raises:
            SystemExit : If the reference cannot be resolved to a swept key.
        """
        if reference in models:
            return reference

        if "-" not in reference and len(heads) == 1:
            candidate = ModelBuilder.model_key(reference, heads[0])
            if candidate in models:
                return candidate

        SizeMatchStage.validate_reference(reference, models)
        return reference

    @staticmethod
    def validate_reference(reference: str, models: list[str]) -> None:
        """Checks that the reference model is one of the swept keys.

        Args:
            reference : Reference model key.
            models    : Swept model keys.

        Raises:
            SystemExit : If the reference is not among them, which would bisect every arm against a model the benchmark never trains.
        """
        if reference not in models:
            raise SystemExit(f"size_match.reference_model '{reference}' is not one of the swept model keys {models}; give the full key including the head (e.g. 'unet-set_pred') so the reference is capacity-matched against its own architecture and not bisected against a different head")

    def _load_cached(self) -> dict | None:
        """Loads the cached match records when they cover every model under the current settings.

        Returns:
            The cached records, or None when not resuming, a model is missing or
            the stored settings differ.
        """
        if not self.config.resume or not self.records_path.exists():
            return None

        records = FileIO.load_json(self.records_path)

        if not all(m in records for m in self.models):
            return None

        settings = self.config.size_match_settings()
        stale    = [m for m in self.models if records[m]["settings"] != settings]

        if stale:
            self.logger.warning(f"Cached size match of {stale} was produced under {records[stale[0]]['settings']}, current settings are {settings}; re-matching every model")
            return None

        return records

    def _reference_record(self, reference: str, target: int) -> dict:
        """Returns the trivial match record of the reference model, at scale 1.0 and zero deviation."""
        return asdict(SizeMatchResult(
            model         = reference,
            scale         = 1.0,
            overrides     = {},
            parameters    = target,
            target        = target,
            deviation_pct = 0.0,
            iterations    = 0,
            settings      = self.config.size_match_settings(),
        ))

    def _write_report(self, records: dict, target: int, out_channels: int) -> None:
        """Writes the markdown table of scale, widths, parameter count and flags per model.

        Args:
            records      : Match record per model key.
            target       : Reference parameter count.
            out_channels : Output channel count the models were counted at.
        """
        lines = [
            "# Capacity Matching Report",
            f"\n_Generated {RunTag.timestamp()}  —  run tag `{self.run_tag}`_\n",
            f"Reference model `{self.config.size_match.reference_model}` at **{target:,}** parameters.",
            f"Counting performed with {self.config.size_match.in_channels} input channels, {out_channels} output channels, patch size {tuple(self.config.training.patch_size)}.\n",
            "## Matched Widths\n",
            "| Model | Scale | Scaled attributes | Parameters | Δ vs reference | Iterations | Flags |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        for model_name in sorted(records.keys()):
            record    = records[model_name]
            overrides = ", ".join(f"`{k}={v}`" for k, v in record["overrides"].items()) or "_(defaults)_"
            flags     = "; ".join(record["flags"]) or "—"
            lines.append(f"| `{model_name}` | {record['scale']:.4f} | {overrides} | {record['parameters']:,} | {record['deviation_pct']:+.3f} % | {record['iterations']} | {flags} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")

    def run(self) -> dict:
        """Matches every non-reference model onto the reference parameter count.

        Returns:
            The match record per model key, reused from cache when valid.

        Raises:
            SystemExit : If the reference is not swept, or a model cannot reach the target within the configured scale bounds.
        """
        self.validate_reference(self.config.size_match.reference_model, self.models)

        matcher = SizeMatcher(config=self.config, logger=self.logger)

        self.logger.section("Capacity matching")
        self.logger.kv_table({
            "Reference model" : self.config.size_match.reference_model,
            "Tolerance"       : f"{100.0 * self.config.size_match.tolerance:.2f} %",
            "Max iterations"  : self.config.size_match.max_iterations,
            "In channels"     : self.config.size_match.in_channels,
            "Out channels"    : matcher.out_channels,
        }, title="Configuration")

        cached = self._load_cached()
        if cached is not None:
            self.logger.info(f"Cached size match reused from: {self.records_path}")
            return cached

        reference = self.config.size_match.reference_model
        target    = matcher.reference_count()

        self.logger.info(f"Reference '{reference}': {target:,} parameters")

        records = {reference: self._reference_record(reference, target)}

        for model_name in self.models:
            if model_name == reference:
                continue

            result              = matcher.match(model_name, target)
            records[model_name] = asdict(result)

            self.logger.info(f"{model_name:<18} {result.parameters:>14,}  Δ {result.deviation_pct:+7.3f} %  (scale {result.scale:.4f}, {result.iterations} iterations)")

            for flag in result.flags:
                self.logger.warning(f"{model_name}: {flag}")

        self._write_results(records, self.records_path)
        self._write_report(records, target, matcher.out_channels)

        return records


class TrainingStage(SeedExpandedStage, QueuedTrainingStage):
    """Trains every seed-expanded benchmark unit through the GPU queue."""

    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        """Expands the models into per-seed units, reconciles them against the training directory and hands them to the queued stage.

        Args:
            config       : Benchmark configuration.
            entry_script : Entry script the training workers re-invoke.
            run_tag      : Tag identifying this benchmark run.
            models       : Model keys to train.
            logger       : Logger for the stage.
        """
        items = self._expand(config, models)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.unidentified = []

        if config.resume:
            self._reconcile(self.stage_dir)
            self._log_unidentified()


class InferenceStage(SeedExpandedStage, QueuedInferenceStage):
    """Runs inference over every seed-expanded benchmark unit through the GPU queue."""

    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        """Expands the models into per-seed units, reconciles them against the training directory and hands them to the queued stage.

        Args:
            config       : Benchmark configuration.
            entry_script : Entry script the inference workers re-invoke.
            run_tag      : Tag identifying this benchmark run.
            models       : Model keys to infer with.
            logger       : Logger for the stage.
        """
        items = self._expand(config, models)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.unidentified = []

        if config.resume:
            self._reconcile(self.stage_dir)
            self._log_unidentified()


class ComparisonStage(ExperimentStage):
    """Collects the finished runs and writes the cross-model comparison reports.

    Attributes:
        reference_model : Model key every other model is compared against.
        embed_images    : Whether figures are embedded into the reports.
        inference_split : Dataset split whose completed inference stamps supply each run's metrics.
    """

    def __init__(self, config, run_tag: str, logger: Logger, reference_model: str, embed_images: bool, inference_split: str, run_dir: Path | None = None) -> None:
        """Stores the comparison reference, figure embedding preference and the scored split.

        Args:
            config          : Benchmark configuration, or the standalone comparison entry configuration, supplying the paths.
            run_tag         : Tag identifying this benchmark run.
            logger          : Logger for the stage.
            reference_model : Model key used as comparison baseline.
            embed_images    : Whether figures are embedded into the reports.
            inference_split : Dataset split whose completed inference stamps are collected.
            run_dir         : Explicit run directory, or None to derive it from the configured log base and the run tag.
        """
        super().__init__(config=config, run_tag=run_tag, logger=logger, run_dir=run_dir)
        self.reference_model = reference_model
        self.embed_images    = embed_images
        self.inference_split = inference_split

    def run(self) -> Path:
        """Aggregates the run records over seeds and writes the reports.

        Returns:
            The timestamped directory the reports were written into.
        """
        self.logger.section("Comparison reports")

        collector = BenchmarkSeedCollector(run_dir=self.run_dir, logger=self.logger, inference_split=self.inference_split)
        records   = collector.collect()

        out_dir = self.run_dir / "comparison" / RunTag.now()

        report = ComparisonReport(
            records         = records,
            out_dir         = out_dir,
            reference_model = self.reference_model,
            embed_images    = self.embed_images,
            logger          = self.logger,
            seed_dispersion = collector.seed_dispersion,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
