"""Composition root of the ML Template control console HTTP server.

Instantiates every console component, wires them into the request router, and
runs the threaded HTTP server together with the background monitors.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib     import Path

from config_registry                   import ConfigRegistry
from cube_explorer                     import CubeExplorer, SliceCollector
from dataset_browser                   import DatasetBrowser
from energy_meter                      import JobEnergyMeter
from equation_library                  import EquationLibrary
from ab_autopsy                        import AbAutopsy
from fit_lab                           import FitLab
from model_probe                       import ModelProbe
from model_survey                      import ModelSurvey
from triage_board                      import TriageBoard
from flow_library                      import FlowLibrary
from gpu_schedule                      import GpuSchedule
from gpu_watchdog                      import GpuWatchdog
from job_describer                     import JobDescriber
from launch_layout                     import LaunchLayout
from backbone_model_library            import BackboneModelLibrary
from image_autoencoder_model_library   import ImageAutoencoderModelLibrary
from pipeline_library                  import PipelineLibrary
from repomap_library                   import RepoMapLibrary
from profile_autoencoder_model_library import ProfileAutoencoderModelLibrary
from jepa_model_library                import JepaModelLibrary
from notifier                          import ExperimentProgressWatcher, JobNotifier
from process_manager                   import ProcessManager, ProcessNuke, ServerDetacher
from project_paths                     import ProjectPaths
from request_router                    import RequestRouter
from resource_watchdog                 import ResourceWatchdog
from contention_monitor                import ContentionMonitor
from results_browser                   import ResultsBrowser
from run_config_reader                 import RunConfigReader
from run_launcher                      import RunLauncher
from run_leaderboard                   import RunLeaderboard
from saved_run_store                   import SavedRunStore
from sequence_decisions                import SequenceDecisions
from sequence_runner                   import SequenceRunner
from sequence_store                    import SequenceStore
from script_catalog                    import ScriptCatalog
from script_config_resolver            import ScriptConfigResolver
from system_monitor                    import SystemMonitor
from tensorboard_manager               import TensorboardManager
from training_curves                   import TrainingCurves
from web_logger                        import WebLogger

from routers.analysis_routers   import AutopsyRouter, FitLabRouter, ProbeRouter, SurveyRouter, TriageRouter
from routers.cube_routers       import CubeRouter, SliceRouter
from routers.launch_routers     import CatalogRouter, JobRouter, RunConfigRouter, SavedRunRouter
from routers.sequence_router    import SequenceRouter
from routers.library_routers    import BackboneRouter, ContentLibraryRouter, ModelLibraryRouter
from routers.results_routers    import CurvesRouter, DatasetRouter, LeaderboardRouter, ResultsRouter
from routers.static_router      import StaticRouter
from routers.system_router      import SystemRouter
from routers.tensorboard_router import TensorboardRouter


class _Server(ThreadingHTTPServer):
    """Threading HTTP server carrying the request router on the server object."""


    request_queue_size = 64
    daemon_threads     = True


class _Handler(BaseHTTPRequestHandler):
    """HTTP/1.1 request handler that delegates every method to the router."""


    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        """Routes a GET request through the server's request router."""

        self.server.router.route(self)

    def do_POST(self) -> None:
        """Routes a POST request through the server's request router."""

        self.server.router.route(self)

    def log_message(self, fmt: str, *args) -> None:
        """Suppresses the default per-request access logging."""

        return


class WebUIServer:
    """Builds and runs the control console: components, routes and background loops.

    Attributes:
        host   : Interface the HTTP server binds to.
        port   : TCP port the HTTP server binds to.
        logger : Console logger shared by every component.
        paths  : Project path resolver rooted at the repository.
        router : Request router holding the ordered route handlers.
    """


    def __init__(self, host: str = "127.0.0.1", port: int = 8772, root: Path | None = None) -> None:
        """Constructs every console component and assembles the request router.

        Args:
            host : Interface to bind the HTTP server to.
            port : TCP port to bind the HTTP server to.
            root : Repository root override; discovered from this file when None.
        """

        self.host   = host
        self.port   = port
        self.logger = WebLogger()
        self.paths  = ProjectPaths(root)

        self.resolver          = ScriptConfigResolver(self.paths)
        self.catalog           = ScriptCatalog(self.paths, self.resolver)
        self.layout            = LaunchLayout()
        self.configs           = ConfigRegistry(self.paths)
        self.equations         = EquationLibrary()
        self.flows             = FlowLibrary()
        self.models            = BackboneModelLibrary()
        self.profile_ae_models = ProfileAutoencoderModelLibrary()
        self.image_ae_models   = ImageAutoencoderModelLibrary()
        self.jepa_models       = JepaModelLibrary()
        self.pipelines         = PipelineLibrary()
        self.repomap           = RepoMapLibrary()
        self.notifier          = JobNotifier(self.paths, self.logger)
        self.describer         = JobDescriber(self.paths, self.resolver)
        self.processes         = ProcessManager(self.paths, self.logger, self.notifier, self.describer)
        self.progress_watch    = ExperimentProgressWatcher(self.processes, self.notifier, self.logger)
        self.saved_runs        = SavedRunStore(self.paths, self.logger)
        self.nuke              = ProcessNuke(self.logger)
        self.detacher          = ServerDetacher(self.paths, self.logger)
        self.system            = SystemMonitor(self.paths, self.logger)
        self.contention        = ContentionMonitor(self.paths, self.logger, self.notifier, self.nuke)
        self.watchdog          = ResourceWatchdog(self.processes, self.contention, self.logger, self.notifier, self.paths)
        self.gpu_guard         = GpuWatchdog(self.system, self.paths, self.logger, self.processes, self.notifier)
        self.gpu_schedule      = GpuSchedule(self.paths, self.logger, self.processes, self.system, self.gpu_guard)
        self.energy            = JobEnergyMeter(self.system, self.processes, self.logger)
        self.tensorboard       = TensorboardManager(self.paths, self.logger)
        self.results           = ResultsBrowser(self.logger)
        self.cubes             = CubeExplorer(self.logger)
        self.slices            = SliceCollector(self.cubes, self.logger)
        self.datasets          = DatasetBrowser(self.logger)
        self.leaderboard       = RunLeaderboard(self.logger)
        self.curves            = TrainingCurves(self.logger)
        self.fitlab            = FitLab(self.logger)
        self.probe             = ModelProbe(self.logger)
        self.survey            = ModelSurvey(self.logger)
        self.triage            = TriageBoard(self.paths, self.cubes.roots, self.logger)
        self.autopsy           = AbAutopsy(self.logger)
        self.launcher          = RunLauncher(self.paths, self.logger, self.resolver, self.processes, self.tensorboard, self.notifier)
        self.run_configs       = RunConfigReader(self.logger)
        self.sequence_decide   = SequenceDecisions(self.resolver, self.launcher, self.logger)
        self.sequence_store    = SequenceStore(self.paths, self.logger, self.sequence_decide)
        self.sequence_runner   = SequenceRunner(self.sequence_store, self.processes, self.launcher, self.resolver, self.notifier, self.logger)

        self.router = RequestRouter(self.logger, [
            StaticRouter(self.paths, self.results),
            ResultsRouter(self.results),
            DatasetRouter(self.datasets),
            LeaderboardRouter(self.leaderboard),
            CurvesRouter(self.curves),
            CubeRouter(self.cubes),
            SliceRouter(self.slices),
            FitLabRouter(self.fitlab),
            ProbeRouter(self.probe),
            SurveyRouter(self.survey),
            TriageRouter(self.triage),
            AutopsyRouter(self.autopsy),
            ContentLibraryRouter("/api/equations",    self.equations,    "groups"),
            ContentLibraryRouter("/api/flows",        self.flows,        "flows"),
            ContentLibraryRouter("/api/pipelines",    self.pipelines,    "pipelines"),
            ContentLibraryRouter("/api/repomap",      self.repomap,      "folders"),
            ContentLibraryRouter("/api/configs",      self.configs,      "groups"),
            BackboneRouter("/api/backbones", self.models),
            ModelLibraryRouter("/api/profile-autoencoders", self.profile_ae_models),
            ModelLibraryRouter("/api/image-autoencoders",   self.image_ae_models),
            ModelLibraryRouter("/api/jepa-variants",        self.jepa_models),
            CatalogRouter(self.paths, self.catalog, self.resolver, self.layout, self.models, self.launcher),
            JobRouter(self.paths, self.processes, self.launcher),
            SavedRunRouter(self.paths, self.saved_runs, self.launcher),
            SequenceRouter(self.sequence_store, self.sequence_runner),
            RunConfigRouter(self.run_configs),
            SystemRouter(self.system, self.watchdog, self.contention, self.gpu_guard, self.gpu_schedule, self.detacher, self.notifier, self.nuke, self.processes, self.logger),
            TensorboardRouter(self.tensorboard, self.launcher),
        ])

    def _report_ready(self) -> None:
        """Prints the startup banner with the URL, profile, roots, script count and interpreters."""

        scripts      = self.catalog.list_scripts()
        interpreters = self.paths.discover_interpreters()
        preferred    = self.paths.preferred_interpreter(interpreters)
        profile      = self.paths.profile

        self.logger.banner("ML Template Control Console", [
            f"URL          http://{self.host}:{self.port}",
            f"repo root    {self.paths.repo_root}",
            f"profile      {profile.path if profile.exists else 'none (run ./install.sh)'}",
            f"runs root    {self.paths.runs_dir}",
            f"scripts      {len(scripts)} entry points",
            f"interpreter  {preferred}",
            f"envs found   {len(interpreters)}",
        ])

    def serve(self) -> None:
        """Serves the console until interrupted, then stops the background workers.

        Starts the contention, resource, GPU, energy and progress monitors, the
        finalize chains of launched runs and the sequence runner, runs the HTTP
        server on a worker thread, and blocks in the detacher wait loop until a
        keyboard interrupt or detach request arrives.
        """

        server        = _Server((self.host, self.port), _Handler)
        server.router = self.router

        self._report_ready()
        self.contention.start()
        self.watchdog.start()
        self.gpu_guard.start()
        self.gpu_schedule.start()
        self.energy.start()
        self.progress_watch.start()
        self.launcher.start()
        self.sequence_runner.start()

        worker = threading.Thread(target=server.serve_forever, name="HttpServer", daemon=True)
        worker.start()

        try:
            self.detacher.wait_loop()
        except KeyboardInterrupt:
            self.logger.warning("shutting down")
        finally:
            self.tensorboard.stop_all()
            server.shutdown()
            server.server_close()
