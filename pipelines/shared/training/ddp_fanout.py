
from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing  import Callable

from copy import deepcopy

from pipelines.shared.training.seed_sweep      import SeedSet, SeedSweepRunner
from pipelines.shared.training.training_runner import RunDirectorySettledMarker
from tools.monitoring.logger                   import Logger
from tools.orchestration.gpu_queue             import GpuJob, GpuJobResult, GpuPoolFile, GpuProgressFile
from tools.runtime.completion                  import CompletionMarker
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.cuda_mask                   import InheritedCudaMask
from tools.training.distributed                import DistributedContext


class DdpFanout:

    OWN_FIELDS = ("ddp", "gpu", "gpus", "gpus_file", "poll_interval_s", "seed", "seeds", "run_name", "logdir", "resume_from")

    MIRROR_PATTERN = re.compile(r".*rank(\d+)$")

    POLL_S  = 2.0
    GRACE_S = 30.0

    def __init__(self, config, cli_overrides: dict, entry_script: Path, run_name: str, base_name: str, seed: int, mirrors: dict | None = None) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_name     = run_name
        self.base_name    = base_name
        self.mirrors      = dict(mirrors or {})
        self.seed         = int(seed)
        self.gpus         = [int(gpu) for gpu in config.gpus]

        self.run_dir = Path(config.logdir) / run_name
        self.ddp_dir = self.run_dir / "ddp"
        self.log_dir = self.ddp_dir / "logs"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.OWN_FIELDS}

        self.logger      = None
        self.progress    = None
        self.rank_starts = {}

    @classmethod
    def _mirrors(cls, target: Path) -> dict:
        mirrors = {}

        if not (target / "ddp").is_dir():
            return mirrors

        for path in sorted((target / "ddp").iterdir()):
            match = cls.MIRROR_PATTERN.fullmatch(path.name)
            if path.is_dir() and match:
                mirrors[int(match.group(1))] = path

        return mirrors

    @classmethod
    def for_runner(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None) -> "DdpFanout":
        if config.resume_from:
            target = Path(config.resume_from).expanduser()

            mirrors = cls._mirrors(target)

            config.logdir = target.parent
            config.resume = True
            run_name      = target.name
            base_name     = target.name
        else:
            base_name      = SeedSweepRunner.base_run_name(config, base_label)
            probe          = deepcopy(config)
            probe.run_name = base_name
            run_name       = runner_factory(probe)._resolve_run_name()
            mirrors        = {}

        seed = SeedSet.resolve(config.seeds, config.seed)[0]

        return cls(config, cli_overrides, entry_script, run_name=run_name, base_name=base_name, seed=seed, mirrors=mirrors)

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def _rank_overrides(self, rank: int) -> dict:
        overrides = {**self.forward_overrides, "seed": self.seed, "seeds": (self.seed,)}

        if self.config.resume_from:
            overrides["resume_from"] = str(self.run_dir if rank == 0 else self.mirrors[rank])
        elif rank == 0:
            overrides.update({"logdir": str(self.config.logdir), "run_name": self.base_name})
        else:
            overrides.update({"logdir": str(self.ddp_dir), "run_name": f"rank{rank}"})

        return overrides

    def _launch(self, rank: int, gpu: int, port: int, handles: list):
        command     = [sys.executable, str(self.entry_script), "--gpu", str(gpu)] + ConfigCli.to_argv(self._rank_overrides(rank))
        environment = {**InheritedCudaMask.restored(), **DistributedContext.environment(rank, len(self.gpus), port)}

        self.rank_starts[rank] = {"gpu": gpu, "started": time.monotonic()}

        if rank == 0:
            return subprocess.Popen(command, env=environment)

        handle = open(self.log_dir / f"rank{rank}.log", "a", encoding="utf-8")
        handles.append(handle)

        return subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)

    def _await_run_directory(self, rank_zero) -> bool:
        sentinel = RunDirectorySettledMarker.path_for(self.run_dir)
        started  = time.time()

        while True:
            if sentinel.is_file() and sentinel.stat().st_mtime >= started - 1.0:
                return True

            code = rank_zero.poll()
            if code == 0:
                return False
            if code is not None:
                raise SystemExit(f"DDP rank 0 failed with exit code {code} before opening the run directory; see the console output")

            time.sleep(self.POLL_S)

    def _open_logger(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir=str(self.log_dir), name="ddp_fanout")

    def _open_progress(self) -> None:
        pool_file = GpuPoolFile.resolve(self.config.gpus_file, self.run_dir)
        jobs      = [GpuJob(name=f"rank{rank}", command=[], log_path=self.log_dir / f"rank{rank}.log") for rank in range(1, len(self.gpus))]

        self.progress = GpuProgressFile(GpuProgressFile.resolve(pool_file), jobs, self.logger)
        self.progress.write(self.progress.snapshot(running=[], queued=len(jobs), workers=len(jobs)), force=True)

    def _clear_rank_completion(self) -> None:
        for rank, mirror in self._mirrors(self.run_dir).items():
            if rank < len(self.gpus) and CompletionMarker.is_complete(mirror):
                CompletionMarker.clear(mirror)
                self.logger.warning(f"rank {rank}: cleared the completion marker of {mirror} so the mirror rejoins the group instead of skipping training")

    def _await_group(self, processes: list) -> None:
        try:
            while any(process.poll() is None for process in processes):
                for rank, process in enumerate(processes):
                    code = process.poll()
                    if code not in (None, 0):
                        self.logger.error(f"rank {rank} exited with code {code}; terminating the remaining ranks")
                        self._terminate(processes)
                        raise SystemExit(f"DDP rank {rank} failed with exit code {code}; see {self.log_dir / f'rank{rank}.log' if rank else 'the console output'}")

                self._publish_progress(processes)
                time.sleep(self.POLL_S)
        except KeyboardInterrupt:
            self.logger.warning("interrupted; terminating every rank")
            self._terminate(processes)
            raise

        failed = [(rank, process.returncode) for rank, process in enumerate(processes) if process.returncode != 0]
        if failed:
            descriptions = ", ".join(f"rank {rank} (exit {code})" for rank, code in failed)
            raise SystemExit(f"DDP ranks failed: {descriptions}; see {self.log_dir}")

    def _publish_progress(self, processes: list, force: bool = False) -> None:
        if self.progress is None:
            return

        running = []
        for rank, process in enumerate(processes):
            if rank == 0:
                continue

            log   = str(self.log_dir / f"rank{rank}.log")
            meta  = self.rank_starts[rank]
            code  = process.poll()

            if code is None:
                running.append({"name": f"rank{rank}", "gpu": meta["gpu"], "log": log, "elapsed_s": round(time.monotonic() - meta["started"], 1)})
            elif self.progress.units[log]["status"] not in GpuProgressFile.TERMINAL:
                status = "DONE" if code == 0 else "FAILED"
                self.progress.record(GpuJobResult(name=f"rank{rank}", gpu=meta["gpu"], status=status, returncode=code, duration_s=time.monotonic() - meta["started"], log_file=log))

        queued = len(self.gpus) - len(processes)
        self.progress.write(self.progress.snapshot(running, queued, workers=len(self.gpus) - 1), force=force)

    def _terminate(self, processes: list) -> None:
        alive = [process for process in processes if process.poll() is None]
        for process in alive:
            process.terminate()

        deadline = time.monotonic() + self.GRACE_S
        for process in alive:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def run(self) -> None:
        port = self._free_port()

        handles   = []
        processes = []
        error     = None

        try:
            processes.append(self._launch(0, self.gpus[0], port, handles))

            if not self._await_run_directory(processes[0]):
                return

            self._open_logger()
            self.logger.section("DDP fan-out")
            self.logger.kv_table({
                "Run dir"      : str(self.run_dir),
                "Ranks"        : len(self.gpus),
                "GPUs"         : self.gpus,
                "Global batch" : f"{self.config.training.batch_size} ({self.config.training.batch_size // len(self.gpus)} per rank)",
                "Seed"         : self.seed,
                "Rendezvous"   : f"tcp://127.0.0.1:{port}",
                "Resume from"  : self.config.resume_from or "—",
                "Forwarded overrides (fan-out options excluded)" : self.forward_overrides or "—",
                "Rank logs"    : f"{self.log_dir} (rank 0 streams to this console)",
            }, title="Configuration")

            self._open_progress()
            self._clear_rank_completion()

            for rank, gpu in enumerate(self.gpus):
                if rank > 0:
                    processes.append(self._launch(rank, gpu, port, handles))
                    self.logger.info(f"rank {rank} launched on GPU {gpu}, streaming into {self.log_dir / f'rank{rank}.log'}; the console stays quiet until every rank reaches the rendezvous")

            self._await_group(processes)
            self.logger.ok(f"all {len(self.gpus)} ranks finished; run artifacts are in {self.run_dir}")
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._terminate(processes)
            self._publish_progress(processes, force=True)
            for handle in handles:
                handle.close()
            if self.logger is not None:
                self.logger.close(error)


class DdpDispatch:

    @classmethod
    def single_seed(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None) -> bool:
        if not config.ddp or DistributedContext.current().active:
            return False

        DdpFanout.for_runner(config, cli_overrides, entry_script, runner_factory, base_label=base_label).run()

        return True
