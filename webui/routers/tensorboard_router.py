from __future__ import annotations

import http.client

from routers.dispatch    import HttpExchange, RouteTable, SubRouter
from run_launcher        import RunLauncher
from tensorboard_manager import TensorboardManager


class TensorboardRouter(SubRouter):

    PASSTHRU_REQUEST  = ("Content-Type", "Accept", "X-XSRF-Protected")
    PASSTHRU_RESPONSE = ("Content-Type", "Content-Encoding", "Cache-Control", "Location")

    def __init__(self, tensorboard: TensorboardManager, launcher: RunLauncher) -> None:
        self.tensorboard = tensorboard
        self.launcher    = launcher

        super().__init__(("/api/tensorboard",), ("/tb/",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/tensorboard",         self.instances)
        table.add("GET",  "/api/tensorboard/logdirs", self.logdirs)
        table.add("POST", "/api/tensorboard/start",   self.start)
        table.wildcard("POST", "/api/tensorboard/", "/stop", self.stop)

    def instances(self, exchange: HttpExchange) -> None:
        exchange.send_json({"instances": self.tensorboard.list_instances()})

    def logdirs(self, exchange: HttpExchange) -> None:
        interpreter = self.launcher.preferred_interpreter("train_backbone")
        runs_root   = self.launcher.runs_root("train_backbone", interpreter)

        if not runs_root:
            exchange.send_json({"ok": False, "error": "could not resolve runs root"}, 400)
            return

        exchange.send_result(self.tensorboard.list_logdirs(runs_root))

    def start(self, exchange: HttpExchange) -> None:
        body        = exchange.body
        interpreter = body.get("interpreter") or self.launcher.preferred_interpreter(body.get("script_key", "train_backbone"))

        error = self.launcher.interpreter_error(interpreter)
        if error:
            exchange.send_json({"ok": False, "error": error}, 400)
            return

        logdir = body.get("logdir") or self.launcher.training_logdir(body.get("script_key", "train_backbone"), {}, interpreter)

        if not logdir:
            exchange.send_json({"ok": False, "error": "could not resolve a training log directory"}, 400)
            return

        exchange.send_result(self.tensorboard.ensure(logdir, interpreter))

    def stop(self, exchange: HttpExchange, tb_id: str) -> None:
        exchange.send_result(self.tensorboard.stop(tb_id))

    def handle_raw(self, exchange: HttpExchange) -> None:
        handler  = exchange.handler
        segments = handler.path.split("/")
        tb_id    = segments[2].split("?")[0] if len(segments) > 2 else ""
        record   = self.tensorboard.get(tb_id)

        if record is None:
            exchange.send_json({"error": "unknown tensorboard instance"}, 404)
            return

        length = int(handler.headers.get("Content-Length", 0) or 0)
        body   = handler.rfile.read(length) if length > 0 else None

        headers = {"Host": f"127.0.0.1:{record['port']}", "Accept-Encoding": "identity"}
        for name in self.PASSTHRU_REQUEST:
            value = handler.headers.get(name)
            if value:
                headers[name] = value

        connection = http.client.HTTPConnection("127.0.0.1", record["port"], timeout=60)
        try:
            connection.request(handler.command, handler.path, body=body, headers=headers)
            response = connection.getresponse()
            payload  = response.read()
            status   = response.status
            passthru = {name: response.getheader(name) for name in self.PASSTHRU_RESPONSE}
        except OSError as exc:
            exchange.send_json({"error": f"tensorboard unreachable: {exc}"}, 502)
            return
        finally:
            connection.close()

        handler.send_response(status)
        for name, value in passthru.items():
            if value:
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
