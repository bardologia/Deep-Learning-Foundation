from __future__ import annotations

import json
import queue

from backbone_model_library import BackboneModelLibrary
from launch_layout          import LaunchLayout, LayoutError
from process_manager        import ProcessManager
from project_paths          import ProjectPaths
from routers.dispatch       import HttpExchange, RouteTable, SubRouter
from run_launcher           import RunLauncher
from saved_run_store        import SavedRunStore
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver


class CatalogRouter(SubRouter):

    PROJECT = {
        "name"        : "DLR-TomoSAR",
        "tagline"     : "Neural SAR tomography control console",
        "description" : "Supervised deep learning that replaces per-pixel iterative optimisation in SAR tomographic parameter estimation, inferring all 3K Gaussian-mixture parameters of the elevation spectrum in one forward pass.",
        "pipelines"   : ["Processing", "Parameter Extraction", "Dataset", "Training", "Inference", "Tuning"],
    }

    def __init__(self, paths: ProjectPaths, catalog: ScriptCatalog, resolver: ScriptConfigResolver, layout: LaunchLayout, models: BackboneModelLibrary, launcher: RunLauncher) -> None:
        self.paths    = paths
        self.catalog  = catalog
        self.resolver = resolver
        self.layout   = layout
        self.models   = models
        self.launcher = launcher

        super().__init__(("/api/project", "/api/scripts"))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/project", self.project)
        table.add("GET", "/api/scripts", self.scripts)
        table.wildcard("GET", "/api/scripts/", "/config", self.script_config)
        table.wildcard("GET", "/api/scripts/", "",        self.script_detail)

    def project(self, exchange: HttpExchange) -> None:
        interpreters = self.paths.discover_interpreters()
        model_names  = [model["name"] for family in self.models.collect() for model in family["models"]]

        exchange.send_json({
            **self.PROJECT,
            "models"       : model_names,
            "repo_root"    : str(self.paths.repo_root),
            "interpreters" : interpreters,
            "preferred"    : self.paths.preferred_interpreter(interpreters),
            "counts"       : {
                "scripts"   : len(self.catalog.list_scripts()),
                "models"    : len(model_names),
                "pipelines" : len(self.PROJECT["pipelines"]),
            },
        })

    def scripts(self, exchange: HttpExchange) -> None:
        exchange.send_json({"scripts": self.catalog.list_scripts()})

    def script_config(self, exchange: HttpExchange, key: str) -> None:
        if not self.paths.has_script(key):
            exchange.send_json({"error": f"unknown script '{key}'"}, 404)
            return

        result = self.resolver.resolve(key, self.launcher.preferred_interpreter(key))
        if result.get("ok"):
            try:
                result = {**result, "layout": self.layout.build(key, result["leaves"])}
            except LayoutError as exc:
                result = {"ok": False, "error": str(exc)}

        exchange.send_result(result)

    def script_detail(self, exchange: HttpExchange, key: str) -> None:
        if not self.paths.has_script(key):
            exchange.send_json({"error": f"unknown script '{key}'"}, 404)
            return

        detail = self.catalog.get_script(key)
        if detail is None:
            exchange.not_found()
            return

        exchange.send_json(detail)


class JobRouter(SubRouter):

    def __init__(self, paths: ProjectPaths, processes: ProcessManager, launcher: RunLauncher) -> None:
        self.paths     = paths
        self.processes = processes
        self.launcher  = launcher

        super().__init__(("/api/run", "/api/jobs"))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/jobs", self.jobs)
        table.add("POST", "/api/run",  self.run)
        table.wildcard("GET",  "/api/jobs/", "/gpus",     self.gpu_pool)
        table.wildcard("GET",  "/api/jobs/", "/progress", self.progress)
        table.wildcard("GET",  "/api/jobs/", "/stream",   self.stream)
        table.wildcard("POST", "/api/jobs/", "/stop",     self.stop)
        table.wildcard("POST", "/api/jobs/", "/gpus",     self.set_gpus)

    def jobs(self, exchange: HttpExchange) -> None:
        self.processes.adopt_orphans()
        exchange.send_json({"jobs": self.processes.list_jobs()})

    def run(self, exchange: HttpExchange) -> None:
        body = exchange.body
        key  = body.get("script_key", "")

        if not self.paths.has_script(key):
            exchange.send_json({"error": f"unknown script '{key}'"}, 404)
            return

        interpreter = body.get("interpreter") or self.launcher.preferred_interpreter(key)
        result      = self.launcher.execute(key, interpreter, body.get("overrides", {}), body.get("follow_up") or None, bool(body.get("detach")), bool(body.get("queue")))
        exchange.send_result(result)

    def gpu_pool(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self.processes.gpu_pool(job_id))

    def progress(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self.processes.progress(job_id))

    def stream(self, exchange: HttpExchange, job_id: str) -> None:
        stream = self.processes.get_stream(job_id)
        if stream is None:
            exchange.send_json({"error": "unknown job"}, 404)
            return

        handler = exchange.handler
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        sub = stream.subscribe()
        try:
            while True:
                try:
                    event = sub.get(timeout=15)
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue

                payload = json.dumps(event)
                handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                handler.wfile.flush()

                if event.get("type") == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.unsubscribe(sub)

    def stop(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self.processes.stop(job_id))

    def set_gpus(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self.processes.set_gpus(job_id, exchange.body.get("gpus"), park=bool(exchange.body.get("park"))))


class SavedRunRouter(SubRouter):

    def __init__(self, paths: ProjectPaths, saved_runs: SavedRunStore, launcher: RunLauncher) -> None:
        self.paths      = paths
        self.saved_runs = saved_runs
        self.launcher   = launcher

        super().__init__(("/api/saved-runs",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/saved-runs", self.listing)
        table.add("POST", "/api/saved-runs", self.save)
        table.wildcard("POST", "/api/saved-runs/", "/run",    self.launch)
        table.wildcard("POST", "/api/saved-runs/", "/delete", self.remove)

    def listing(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.saved_runs.list())

    def save(self, exchange: HttpExchange) -> None:
        body        = exchange.body
        key         = body.get("script_key", "")
        interpreter = body.get("interpreter", "")

        if not interpreter and self.paths.has_script(key):
            interpreter = self.launcher.preferred_interpreter(key)

        error = self.launcher.interpreter_error(interpreter)
        if error:
            exchange.send_json({"ok": False, "error": error}, 400)
            return

        exchange.send_result(self.saved_runs.save({**body, "interpreter": interpreter}))

    def launch(self, exchange: HttpExchange, saved_id: str) -> None:
        entry = self.saved_runs.get(saved_id)
        if entry is None:
            exchange.send_json({"ok": False, "error": "saved run not found"}, 404)
            return

        result = self.launcher.execute(entry["script"], entry["interpreter"], entry["overrides"], entry["follow_up"], entry["detach"], bool(exchange.body.get("queue")))
        exchange.send_result(result)

    def remove(self, exchange: HttpExchange, saved_id: str) -> None:
        exchange.send_result(self.saved_runs.delete(saved_id))
