
from __future__ import annotations

import json

from backbone_model_library import BackboneModelLibrary
from config_template_store  import ConfigTemplateStore
from launch_layout          import LaunchLayout, LayoutError
from process_manager        import ProcessManager
from project_paths          import ProjectPaths
from routers.dispatch       import HttpExchange, RouteTable, SubRouter
from run_config_reader      import RunConfigReader
from run_launcher           import RunLauncher
from saved_run_store        import SavedRunStore
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver
from stream_hub             import StreamChannel, StreamHub


class CatalogRouter(SubRouter):

    PROJECT = {
        "name"        : "ML Template",
        "tagline"     : "Deep learning project template and control console",
        "description" : "A reusable scaffold for supervised deep learning projects: dataset assembly, model zoo, training, inference, experiments and analysis, all wired to this console. The bundled example task regresses two target channels from four input channels of a synthetic dataset on a 2-D grid.",
        "pipelines"   : ["Parameter Extraction", "Dataset", "Training", "Inference", "Tuning"],
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
            "roots"        : self.paths.roots(),
            "profile"      : self.paths.profile.summary(),
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

        result = self.resolver.resolve(key, self.launcher.preferred_interpreter())
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


class RunConfigRouter(SubRouter):

    def __init__(self, reader: RunConfigReader) -> None:
        self.reader = reader

        super().__init__(("/api/run-config",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/run-config/runs", self.runs)
        table.add("GET", "/api/run-config",      self.config)

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.reader.list_runs(exchange.text("base")))

    def config(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.reader.read(exchange.text("path")))


class JobRouter(SubRouter):

    KEEPALIVE_S = 15.0

    def __init__(self, paths: ProjectPaths, processes: ProcessManager, launcher: RunLauncher, hub: StreamHub) -> None:
        self.paths     = paths
        self.processes = processes
        self.launcher  = launcher
        self.hub       = hub

        super().__init__(("/api/run", "/api/jobs", "/api/stream"))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/jobs",   self.jobs)
        table.add("POST", "/api/run",    self.run)
        table.add("GET",  "/api/stream", self.stream)
        table.wildcard("GET",  "/api/jobs/",   "/gpus",        self.gpu_pool)
        table.wildcard("GET",  "/api/jobs/",   "/progress",    self.progress)
        table.wildcard("GET",  "/api/jobs/",   "/log",         self.log)
        table.wildcard("POST", "/api/jobs/",   "/stop",        self.stop)
        table.wildcard("POST", "/api/jobs/",   "/gpus",        self.set_gpus)
        table.wildcard("POST", "/api/jobs/",   "/tensorboard", self.tensorboard_bridge)
        table.wildcard("POST", "/api/stream/", "/subscribe",   self.subscribe)
        table.wildcard("POST", "/api/stream/", "/unsubscribe", self.unsubscribe)

    def jobs(self, exchange: HttpExchange) -> None:
        self.processes.adopt_orphans()

        chains = {chain["job_id"]: chain["steps"][chain["index"]:] for chain in self.launcher.finalizer.pending()}
        local  = self.processes.list_jobs()
        for record in local:
            record["tensorboard"] = self.launcher.tensorboard_capable(record["script"])
            record["finalize"]    = chains.get(record["job_id"], [])

        merged = list(local)
        merged.sort(key=lambda record: record["started"], reverse=True)

        exchange.send_json({"jobs": merged})

    def run(self, exchange: HttpExchange) -> None:
        body = exchange.body
        key  = body.get("script_key", "")

        if not self.paths.has_script(key):
            exchange.send_json({"error": f"unknown script '{key}'"}, 404)
            return

        interpreter = body.get("interpreter") or self.launcher.preferred_interpreter()
        result      = self.launcher.execute(key, interpreter, body.get("overrides", {}), body.get("follow_up") or None, bool(body.get("detach")), bool(body.get("queue")), body.get("finalize"))
        exchange.send_result(result)

    def gpu_pool(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self._backend(job_id).gpu_pool(job_id))

    def progress(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self._backend(job_id).progress(job_id))

    def log(self, exchange: HttpExchange, job_id: str) -> None:
        unit = self._unit_index(exchange)
        if unit == "invalid":
            return

        exchange.send_result(self._backend(job_id).job_log(job_id, unit))

    def _unit_index(self, exchange: HttpExchange) -> int | str | None:
        raw = exchange.text("unit")
        if not raw:
            return None

        try:
            return int(raw)
        except ValueError:
            exchange.send_json({"ok": False, "error": f"invalid unit index '{raw}'"}, 400)
            return "invalid"

    def stream(self, exchange: HttpExchange) -> None:
        channel_id = exchange.text("channel")

        if not channel_id:
            exchange.send_json({"ok": False, "error": "missing channel parameter"}, 400)
            return

        channel = self.hub.open(channel_id)
        handler = exchange.handler

        exchange.status = 200
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()

        try:
            self._write_event(handler, {"type": "hello", "channel": channel_id, "boot": self.hub.boot})
            self._pump_channel(handler, channel)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.hub.release(channel)

    def _pump_channel(self, handler, channel: StreamChannel) -> None:
        while True:
            event = channel.next(self.KEEPALIVE_S)

            if event is None:
                handler.wfile.write(b": keepalive\n\n")
                handler.wfile.flush()
                continue

            if event is StreamChannel.CLOSE:
                return

            self._write_event(handler, event)

    @staticmethod
    def _write_event(handler, event: dict) -> None:
        handler.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        handler.wfile.flush()

    def subscribe(self, exchange: HttpExchange, channel_id: str) -> None:
        request = self._channel_request(exchange)
        if request is None:
            return

        job_id, unit = request
        after        = exchange.body.get("after", 0)

        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            exchange.send_json({"ok": False, "error": f"after must be a non-negative integer event id, got {after!r}"}, 400)
            return

        exchange.send_result(self.hub.subscribe(channel_id, self._backend(job_id), job_id, unit, after), 404)

    def unsubscribe(self, exchange: HttpExchange, channel_id: str) -> None:
        request = self._channel_request(exchange)
        if request is None:
            return

        job_id, unit = request
        exchange.send_result(self.hub.unsubscribe(channel_id, job_id, unit), 404)

    def _channel_request(self, exchange: HttpExchange) -> tuple[str, int | None] | None:
        if not isinstance(exchange.body, dict):
            exchange.send_json({"ok": False, "error": "the request body must be a JSON object"}, 400)
            return None

        job_id = exchange.body.get("job_id")
        unit   = exchange.body.get("unit")

        if not isinstance(job_id, str) or not job_id:
            exchange.send_json({"ok": False, "error": "job_id must be a non-empty string"}, 400)
            return None

        if unit is not None and (not isinstance(unit, int) or isinstance(unit, bool)):
            exchange.send_json({"ok": False, "error": f"unit must be an integer index or null, got {unit!r}"}, 400)
            return None

        return job_id, unit

    def stop(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self._backend(job_id).stop(job_id))

    def set_gpus(self, exchange: HttpExchange, job_id: str) -> None:
        exchange.send_result(self._backend(job_id).set_gpus(job_id, exchange.body.get("gpus"), park=bool(exchange.body.get("park")), preempt=exchange.body.get("preempt")))

    def _backend(self, job_id: str):
        return self.processes

    def tensorboard_bridge(self, exchange: HttpExchange, job_id: str) -> None:
        record = self.processes.job_record(job_id)
        if record is None:
            exchange.send_json({"ok": False, "error": "unknown job"}, 404)
            return

        exchange.send_result(self.launcher.job_tensorboard(record))


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
            interpreter = self.launcher.preferred_interpreter()

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

        missing = [key for key in (entry["script"], entry["follow_up"]) if key and not self.paths.has_script(key)]
        if missing:
            exchange.send_json({"ok": False, "error": f"saved run points at removed entry point(s): {', '.join(missing)}"}, 404)
            return

        result = self.launcher.execute(entry["script"], entry["interpreter"], entry["overrides"], entry["follow_up"], entry["detach"], bool(exchange.body.get("queue")), entry["finalize"])
        exchange.send_result(result)

    def remove(self, exchange: HttpExchange, saved_id: str) -> None:
        exchange.send_result(self.saved_runs.delete(saved_id))


class ConfigTemplateRouter(SubRouter):

    def __init__(self, templates: ConfigTemplateStore) -> None:
        self.templates = templates

        super().__init__(("/api/config-templates",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/config-templates", self.listing)
        table.add("POST", "/api/config-templates", self.save)
        table.wildcard("POST", "/api/config-templates/", "/rename", self.rename)
        table.wildcard("POST", "/api/config-templates/", "/delete", self.remove)

    def listing(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.templates.list())

    def save(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.templates.save(exchange.body))

    def rename(self, exchange: HttpExchange, template_id: str) -> None:
        exchange.send_result(self.templates.rename(template_id, exchange.body.get("name", "")))

    def remove(self, exchange: HttpExchange, template_id: str) -> None:
        exchange.send_result(self.templates.delete(template_id))
