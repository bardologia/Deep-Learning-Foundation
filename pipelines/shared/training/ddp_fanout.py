"""Data-parallel fan-out of one training run across the local GPUs.

Launches one synchronised training rank per device in two stages: rank zero
starts alone and owns the canonical run directory and the console, and the
other ranks join only once rank zero's resume gate has settled that directory,
writing their mirrored artifacts into an isolated ``ddp/rank<N>`` subtree so no
file is ever written by two processes. The ranks rendezvous through a
``torch.distributed`` process group whose coordinates are stamped into each
child's environment, average their gradients every optimizer window, and reduce
their validation losses, so the processes behave as one run with the configured
global batch split across the devices.

The dispatch class next to the fan-out holds the decision every training entry
shares: whether this process fans a run out, trains it in place as one of the
ranks, or must refuse because a seed sweep or a trial sweep already claims one
GPU per unit.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing  import Callable

from pipelines.shared.training.seed_sweep      import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from pipelines.shared.training.training_runner import RunDirectorySettledMarker
from tools.monitoring.logger                   import Logger
from tools.orchestration.gpu_queue             import GpuJob, GpuJobResult, GpuPoolFile, GpuProgressFile
from tools.runtime.completion                  import CompletionMarker
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.cuda_mask                   import InheritedCudaMask
from tools.training.distributed                import DistributedContext


class DdpFanout:
    """Runs one training configuration as a staged group of data-parallel rank processes.

    Rank zero is launched first and alone, so its resume gate can settle the
    canonical run directory; the remaining ranks are launched once that
    directory is confirmed settled.

    Attributes:
        OWN_FIELDS        : Top-level config fields owned by the fan-out and therefore never forwarded to the rank processes.
        POLL_S            : Seconds between liveness polls of the rank processes.
        GRACE_S           : Seconds a terminated rank gets before it is killed.
        config            : Entry configuration of the run.
        entry_script      : Training entry script each rank re-invokes.
        run_name          : Resolved run directory name shared by every rank's naming.
        base_name         : Run name handed to rank zero, which its own naming scheme composes back into ``run_name``.
        naming_prefix     : Descriptive fragment the entry's naming scheme prepends to a run name, empty for entries that name runs verbatim.
        seed              : Seed pinned identically on every rank.
        gpus              : CUDA device indices, one rank launched per device.
        run_dir           : Canonical run directory owned by rank zero.
        ddp_dir           : Subtree of the run directory holding the other ranks' output.
        log_dir           : Directory holding the fan-out log and the per-rank launch logs.
        forward_overrides : CLI overrides forwarded to every rank.
        logger            : Logger of the fan-out, opened only once rank zero has settled the run directory (a fresh run deletes a stale unfinished directory, so nothing may be created inside it before that).
        progress          : Per-rank progress ledger written beside the GPU pool path so the console's sub-run selector can stream the rank logs, opened with the logger.
        rank_starts       : Device index and monotonic start time per launched rank, keyed by rank, feeding the progress snapshots.
    """

    OWN_FIELDS = ("ddp", "gpu", "gpus", "gpus_file", "poll_interval_s", "seed", "seeds", "run_name", "logdir", "resume_from")

    POLL_S  = 2.0
    GRACE_S = 30.0

    def __init__(self, config, cli_overrides: dict, entry_script: Path, run_name: str, base_name: str, naming_prefix: str, seed: int) -> None:
        """Prepares the rank plan, directories and forwarded overrides.

        Args:
            config        : Entry configuration of the run.
            cli_overrides : All CLI overrides; fan-out-owned fields are filtered out before forwarding.
            entry_script  : Training entry script each rank re-invokes.
            run_name      : Resolved run directory name.
            base_name     : Run name handed to rank zero, composed back into ``run_name`` by the entry's own naming scheme.
            naming_prefix : Fragment that naming scheme prepends to a run name.
            seed          : Seed pinned on every rank.
        """
        self.config        = config
        self.entry_script  = Path(entry_script)
        self.run_name      = run_name
        self.base_name     = base_name
        self.naming_prefix = naming_prefix
        self.seed          = int(seed)
        self.gpus          = [int(gpu) for gpu in config.gpus]

        self.run_dir = Path(config.logdir) / run_name
        self.ddp_dir = self.run_dir / "ddp"
        self.log_dir = self.ddp_dir / "logs"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.OWN_FIELDS}

        self.logger      = None
        self.progress    = None
        self.rank_starts = {}

    @classmethod
    def _validate(cls, config) -> None:
        """Rejects configurations the synchronous rank group cannot run.

        Args:
            config : Entry configuration of the run.

        Raises:
            ValueError : If fewer than two distinct GPUs are given, the batch size does not divide across them, or a pre-training probe is enabled (probes explore rank-locally, so their outcomes could diverge and desynchronise the group's collectives).
        """
        gpus = [int(gpu) for gpu in config.gpus]

        if len(gpus) < 2 or len(set(gpus)) != len(gpus):
            raise ValueError(f"ddp=True needs at least two distinct devices in gpus, got {config.gpus}; run without ddp for a single-GPU training")

        if config.training.batch_size % len(gpus) != 0:
            raise ValueError(f"training.batch_size {config.training.batch_size} does not divide across {len(gpus)} DDP ranks; pick a multiple of the rank count so the global batch is preserved exactly")

        probes = {"find_batch_size": config.pretrain.find_batch_size, "tune_loader": config.pretrain.tune_loader, "find_lr": config.pretrain.find_lr}
        active = [name for name, enabled in probes.items() if enabled]
        if active:
            raise ValueError(f"ddp=True cannot run the pre-training probes ({', '.join(active)}): probe outcomes are rank-local and could diverge, desynchronising the rank group; disable the probes or run them once without ddp first")

    @classmethod
    def _validate_resume_group(cls, target: Path, prefix: str, n_ranks: int) -> None:
        """Refuses resuming a run whose rank mirrors do not match the requested rank group.

        Every rank of a resumed group restores its own trainer state, so a rank
        without a mirror would train from scratch and desynchronise the gradient
        averaging, while a leftover mirror would silently drop a trained rank.

        Args:
            target  : Run directory being resumed.
            prefix  : Naming-scheme fragment the mirror directory names are prefixed with.
            n_ranks : Number of ranks the current ``gpus`` setting builds.

        Raises:
            SystemExit : If the run's ``ddp`` mirrors do not cover exactly the ranks 1 to n_ranks - 1.
        """
        expected = {f"{prefix}rank{rank}" for rank in range(1, n_ranks)}
        found    = {path.name for path in (target / "ddp").glob(f"{prefix}rank*") if path.is_dir()}

        if found != expected:
            raise SystemExit(f"resume_from run '{target.name}' was trained by a different rank group: gpus asks for {n_ranks} ranks, expecting the mirror directories {sorted(expected)} under {target / 'ddp'}, but the run holds {sorted(found) or 'none'}. A rank without a mirror would train from scratch and desynchronise the gradient averaging; resume with the run's original device count or start a fresh run")

    @classmethod
    def for_runner(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None) -> "DdpFanout":
        """Builds the fan-out for an entry, resuming or naming a fresh run.

        Args:
            config         : Entry configuration of the run.
            cli_overrides  : All CLI overrides seen by the entry.
            entry_script   : Training entry script each rank re-invokes.
            runner_factory : Callable turning a config into a single-run runner, used to compose the fresh run name.
            base_label     : Optional label prefixed to a timestamped run name.

        Returns:
            The configured fan-out.

        Raises:
            FileNotFoundError : If ``resume_from`` is not an existing directory.
            SystemExit        : If a resumed run directory does not match the naming scheme composed from the current settings, or its rank mirrors do not match the requested rank group.
            ValueError        : If the configuration cannot run as a rank group.
        """
        cls._validate(config)

        prefix = SeedFanoutScheduler.naming_prefix(config, runner_factory)

        if config.resume_from:
            target = Path(config.resume_from).expanduser()
            if not target.is_dir():
                raise FileNotFoundError(f"resume_from points at '{target}', which is not an existing run directory; paste the full path of the run to resume")

            if not target.name.startswith(prefix) or len(target.name) == len(prefix):
                raise SystemExit(f"resume_from run directory '{target.name}' does not match '{prefix}<base>' composed from the current settings; align the entry settings with the original run before resuming it data-parallel")

            cls._validate_resume_group(target, prefix, len(config.gpus))

            config.logdir          = target.parent
            config.resume          = True
            config.training.resume = True
            run_name               = target.name
            base_name              = target.name[len(prefix):]
        else:
            base_name = SeedSweepRunner.base_run_name(config, base_label)
            run_name  = f"{prefix}{base_name}"

        seed = SeedSet.resolve(config.seeds, config.seed)[0]

        return cls(config, cli_overrides, entry_script, run_name=run_name, base_name=base_name, naming_prefix=prefix, seed=seed)

    @staticmethod
    def _free_port() -> int:
        """Returns a currently free TCP port on the loopback interface."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def _mirror_name(self, rank: int) -> str:
        """Returns the directory name a secondary rank's own naming scheme produces.

        Args:
            rank : Zero-based rank of the process, greater than zero.
        """
        return f"{self.naming_prefix}rank{rank}"

    def _rank_overrides(self, rank: int) -> dict:
        """Returns the CLI overrides pinning one rank's identity and output tree.

        Every rank is handed the bare run name its own naming scheme composes,
        so the directory it opens matches the one the fan-out predicted. Rank
        zero keeps the canonical run directory; every other rank is redirected
        into the ``ddp/rank<N>`` subtree so its mirrored artifacts never collide
        with the canonical ones.

        Args:
            rank : Zero-based rank of the process.
        """
        overrides = {**self.forward_overrides, "seed": self.seed, "seeds": (self.seed,)}

        if self.config.resume_from:
            overrides.update({"resume": True, "training.resume": True})

        if rank == 0:
            overrides.update({"logdir": str(self.config.logdir), "run_name": self.base_name})
        else:
            overrides.update({"logdir": str(self.ddp_dir), "run_name": f"rank{rank}"})

        return overrides

    def _launch(self, rank: int, gpu: int, port: int, handles: list):
        """Starts one rank process pinned to its device.

        Rank zero inherits the console; the other ranks stream into per-rank
        launch logs. Each rank's environment restores the CUDA mask the parent
        inherited before pinning itself, so the rank resolves its --gpu against
        the original device set and stashes its own snapshot.

        Args:
            rank    : Zero-based rank of the process.
            gpu     : CUDA device index the rank is pinned to.
            port    : Rendezvous port shared by the group.
            handles : List collecting the opened log handles for later closing.

        Returns:
            The started subprocess.
        """
        command     = [sys.executable, str(self.entry_script), "--gpu", str(gpu)] + ConfigCli.to_argv(self._rank_overrides(rank))
        environment = {**InheritedCudaMask.restored(), **DistributedContext.environment(rank, len(self.gpus), port)}

        self.rank_starts[rank] = {"gpu": gpu, "started": time.monotonic()}

        if rank == 0:
            return subprocess.Popen(command, env=environment)

        handle = open(self.log_dir / f"rank{rank}.log", "w", encoding="utf-8")
        handles.append(handle)

        return subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)

    def _await_run_directory(self, rank_zero) -> bool:
        """Waits for rank zero to settle the run directory before anything joins it.

        A fresh run's resume gate deletes a stale unfinished run directory, so
        the other ranks and the fan-out's own files must stay out of it until
        rank zero has re-created it. The freshly written
        ``RunDirectorySettledMarker`` marks that moment.

        Args:
            rank_zero : The already-started rank-zero subprocess.

        Returns:
            True once the run directory is settled; False when rank zero exited
            cleanly before opening it (the run was already complete and skipped).

        Raises:
            SystemExit : If rank zero exited with a non-zero code first.
        """
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
        """Creates the fan-out log directory and its logger inside the settled run."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir=str(self.log_dir), name="ddp_fanout")

    def _open_progress(self) -> None:
        """Publishes a progress ledger with one unit per secondary rank.

        The ledger is the same per-unit registry every fan-out writes beside
        its GPU pool path, so the web console's sub-run selector lists the
        ``ddp/logs/rank<N>.log`` files and streams them live. Rank zero carries
        no unit: its output is the pipeline console itself.
        """
        pool_file = GpuPoolFile.resolve(self.config.gpus_file, self.run_dir)
        jobs      = [GpuJob(name=f"rank{rank}", command=[], log_path=self.log_dir / f"rank{rank}.log") for rank in range(1, len(self.gpus))]

        self.progress = GpuProgressFile(GpuProgressFile.resolve(pool_file), jobs, self.logger)
        self.progress.write(self.progress.snapshot(running=[], queued=len(jobs), workers=len(jobs)), force=True)

    def _clear_rank_completion(self) -> None:
        """Clears the completion markers of the secondary ranks' mirror directories.

        A mirror that still carries a marker would be reused by its rank's
        resume gate, so that rank would exit cleanly without ever joining the
        process group and leave the training ranks blocked in the rendezvous
        until it times out. Rank zero's gate has already decided that this run
        trains, so its mirrors must train with it.
        """
        for rank in range(1, len(self.gpus)):
            mirror = self.ddp_dir / self._mirror_name(rank)

            if CompletionMarker.is_complete(mirror):
                CompletionMarker.clear(mirror)
                self.logger.warning(f"rank {rank}: cleared the completion marker of {mirror} so the mirror rejoins the group instead of skipping training")

    def _await_group(self, processes: list) -> None:
        """Waits for every rank, tearing the group down on the first failure.

        Args:
            processes : Rank subprocesses in rank order.

        Raises:
            SystemExit : If any rank exited with a non-zero code.
        """
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
        """Folds exited secondary ranks into the ledger and writes a snapshot.

        Args:
            processes : Rank subprocesses in rank order, rank zero first.
            force     : Writes the snapshot regardless of the throttle when True.
        """
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
        """Terminates every still-running rank, escalating to kill after the grace period.

        Args:
            processes : Rank subprocesses in rank order.
        """
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
        """Launches rank zero, then the remaining ranks, and waits for the group.

        Rank zero starts alone so its resume gate can settle the run directory;
        once the directory is confirmed fresh the mirrors' completion markers
        are cleared and the other ranks join it. A rank zero that exits cleanly
        before opening the directory skipped an already-complete run, and no
        further rank is launched.

        Raises:
            SystemExit : If any rank exited with a non-zero code.
        """
        port = self._free_port()

        handles   = []
        processes = []

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
        finally:
            self._terminate(processes)
            self._publish_progress(processes, force=True)
            for handle in handles:
                handle.close()
            if self.logger is not None:
                self.logger.close()


class DdpDispatch:
    """Decides, for one training entry, whether a run fans out as a rank group.

    Every training entry offers the same ``ddp`` switch, and the same
    combinations are impossible: several seeds, or a trial sweep, already claim
    one GPU per unit, so they cannot also split one run across the pool. This
    class holds that shared decision and the loud rejections around it.
    """

    @staticmethod
    def reject_seed_sweep(config, seeds: list[int]) -> None:
        """Refuses a multi-seed sweep combined with data-parallel training.

        Args:
            config : Entry configuration of the run.
            seeds  : Seeds resolved from the configuration.

        Raises:
            ValueError : If ``ddp`` is set for more than one seed.
        """
        if config.ddp:
            raise ValueError(f"ddp=True trains exactly one seed data-parallel, but seeds={seeds}; pin a single seed, or drop ddp to fan the seeds out one GPU each")

    @staticmethod
    def reject_trials(config, trial_unit: bool) -> None:
        """Refuses a trial sweep, or one of its units, combined with data-parallel training.

        Args:
            config     : Entry configuration of the run.
            trial_unit : Whether the entry was re-invoked as a single sweep unit.

        Raises:
            ValueError : If ``ddp`` is set for a trial sweep or a trial unit.
        """
        if not config.ddp:
            return

        if trial_unit:
            raise ValueError("ddp=True is not available for a --trial unit: a trial sweep fans its units out one GPU each, so a unit owns a single device; drop --trial to train one data-parallel run, or drop ddp")

        raise ValueError(f"ddp=True trains one run across the gpus devices, but trials_enabled=True schedules the '{config.trials_mode}' sweep, which fans its units out one GPU each; disable the trials to train a single data-parallel run, or drop ddp to run the sweep. Data-parallel training is local-only for now, so a sweep unit never carries it to the cluster either")

    @classmethod
    def single_seed(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None) -> bool:
        """Fans the single resolved seed out as a rank group when ``ddp`` asks for it.

        A process that is itself a rank of a running group never fans out
        again; it trains in place.

        Args:
            config         : Entry configuration of the run.
            cli_overrides  : All CLI overrides seen by the entry.
            entry_script   : Training entry script each rank re-invokes.
            runner_factory : Callable turning a config into a single-run runner.
            base_label     : Optional label prefixed to a timestamped run name.

        Returns:
            True when the rank group ran, False when the caller must train the
            run in process itself.
        """
        if not config.ddp or DistributedContext.current().active:
            return False

        DdpFanout.for_runner(config, cli_overrides, entry_script, runner_factory, base_label=base_label).run()

        return True
