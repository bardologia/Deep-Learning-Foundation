from __future__ import annotations

import sys
from pathlib import Path

from configuration.patch_sweep            import PatchSweepConfig
from pipelines.patch_sweep.planner        import PatchSweepPlanner
from pipelines.patch_sweep.report         import PatchSweepReport, SweepCollector
from pipelines.shared.training.seed_sweep import SeedSet
from tools.orchestration.gpu_queue        import GpuJob
from tools.orchestration.stages           import ExperimentStage, QueuedTrainingStage
from tools.monitoring.logger              import Logger
from tools.runtime.run_tag                import RunTag


class SweepTrainingStage(QueuedTrainingStage):
    def __init__(self, config: PatchSweepConfig, entry_script: Path, run_tag: str, planner: PatchSweepPlanner, logger: Logger) -> None:
        units      = SeedSet.units([unit.name for unit in planner.units()], config.seeds)
        self._unit = {run_name: (unit_name, seed) for unit_name, seed, run_name in units}
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=[run_name for _, _, run_name in units], logger=logger)
        self.planner = planner

    def _config_kv(self) -> dict:
        azimuth_sizes, range_sizes = self.planner.patch_sizes()

        return {
            "Model"         : f"{self.config.backbone_name}-{self.config.backbone_head}",
            "Datasets"      : [dataset.name for dataset in self.planner.datasets],
            "Azimuth sizes" : azimuth_sizes,
            "Range sizes"   : range_sizes,
            "Seeds"         : self.config.seeds or "—",
            "Units"         : len(self.items),
            "Epochs"        : self.config.training.epochs,
            "GPUs"          : self.config.gpus,
            "Stage dir"     : str(self.stage_dir),
        }

    def _job(self, item: str) -> GpuJob:
        unit_name, seed = self._unit[item]

        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), "--worker", self.worker_action, "--unit", unit_name, *SeedSet.cli_args(seed), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / item / self.worker_logname,
        )


class SweepReportStage(ExperimentStage):
    def __init__(self, config: PatchSweepConfig, run_tag: str, planner: PatchSweepPlanner, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.planner = planner

    def run(self) -> Path:
        self.logger.section("Patch sweep report")

        collector = SweepCollector(run_dir=self.run_dir, planner=self.planner, logger=self.logger)
        records   = collector.collect()

        out_dir = self.run_dir / "report" / RunTag.now()
        report  = PatchSweepReport(records=records, planner=self.planner, out_dir=out_dir, logger=self.logger)

        written = report.write_all()

        self.logger.subsection("Report written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
