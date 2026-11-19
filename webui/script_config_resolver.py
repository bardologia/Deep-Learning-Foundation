
from __future__ import annotations

import ast
import json
import subprocess
import threading

from project_paths import ProjectPaths


class ScriptConfigResolver:

    BOOTSTRAP = (
        "import json, sys\n"
        "repo = sys.argv[1]\n"
        "sys.path.insert(0, repo)\n"
        "from enum import Enum\n"
        "from tools.runtime.config_cli import ConfigCli\n"
        "from webui.config_value_render import ConfigValueRenderer\n"
        "module = __import__(sys.argv[2], fromlist=[sys.argv[3]])\n"
        "cli = ConfigCli(getattr(module, sys.argv[3])())\n"
        "config = cli.config\n"
        "leaves = []\n"
        "for path, value, section, section_class in ConfigCli.detailed_leaves(config):\n"
        "    editable = ConfigCli.is_editable(value)\n"
        "    rendered = ConfigValueRenderer.render(value)\n"
        "    kind = 'none' if value is None else type(value).__name__\n"
        "    choices = [e.value for e in type(value)] if isinstance(value, Enum) else None\n"
        "    source = 'profile' if path in cli.profile_overrides else 'default'\n"
        "    leaves.append({'path': path, 'value': rendered, 'type': kind, 'editable': editable, 'choices': choices, 'section': section, 'section_class': section_class, 'source': source})\n"
        "print(json.dumps(leaves))\n"
    )

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths    = paths
        self.cache    = {}
        self.entries  = {}
        self.inflight = {}
        self.lock     = threading.Lock()

    def entry_config(self, key: str) -> dict | None:
        entry = self.paths.script_entry(key)

        if entry["config_module"] and entry["config_class"]:
            return {"module": entry["config_module"], "class": entry["config_class"]}

        path = entry["path"]
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None

        with self.lock:
            hit = self.entries.get(key)
            if hit is not None and hit[0] == stamp:
                return hit[1]

        detected = self._detect_entry_config(path)

        with self.lock:
            self.entries[key] = (stamp, detected)
        return detected

    def _detect_entry_config(self, path) -> dict | None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return None

        used = self._config_cli_class(tree)
        if used is None:
            return None

        located = self._locate_import(tree, used)
        if located is None:
            return None

        module, real_name = located
        return {"module": module, "class": real_name}

    def _config_cli_class(self, tree: ast.Module) -> str | None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "ConfigCli":
                continue
            if not node.args:
                continue

            first = node.args[0]
            if isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
                return first.func.id
        return None

    def _locate_import(self, tree: ast.Module, used: str) -> tuple[str, str] | None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("configuration"):
                continue
            for alias in node.names:
                if (alias.asname or alias.name) == used:
                    return node.module, alias.name
        return None

    def _signature(self, key: str) -> tuple:
        watched = sorted(self.paths.config_dir.rglob("*.py"))
        watched.append(self.paths.script_entry(key)["path"])
        watched.append(self.paths.repo_root / "tools" / "runtime" / "config_cli.py")
        watched.append(self.paths.repo_root / "tools" / "runtime" / "machine_profile.py")
        watched.append(self.paths.repo_root / "webui" / "config_value_render.py")

        return self.paths.tree_signature(watched) + self.paths.profile_signature()

    def _run_bootstrap(self, entry: dict, interpreter: str) -> dict:
        argv = [interpreter, "-c", self.BOOTSTRAP, str(self.paths.repo_root), entry["module"], entry["class"]]

        try:
            proc = subprocess.run(argv, cwd=str(self.paths.repo_root), capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": f"config resolution failed under {interpreter}: {exc}"}

        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-4:])
            return {"ok": False, "error": f"config resolution failed under {interpreter}:\n{tail}"}

        try:
            leaves = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "config resolution produced no output"}

        return {"ok": True, "module": entry["module"], "config_class": entry["class"], "leaves": leaves}

    def resolve(self, key: str, interpreter: str) -> dict:
        entry = self.entry_config(key)
        if entry is None:
            return {"ok": False, "error": "no entry configuration detected"}

        signature    = self._signature(key)
        cache_key    = (key, interpreter)
        inflight_key = (key, interpreter, signature)

        with self.lock:
            hit = self.cache.get(cache_key)
            if hit is not None and hit[0] == signature:
                return hit[1]

            pending = self.inflight.get(inflight_key)
            leader  = pending is None
            if leader:
                pending                     = {"event": threading.Event(), "result": None, "failure": None}
                self.inflight[inflight_key] = pending

        if not leader:
            pending["event"].wait()
            if pending["failure"] is not None:
                raise pending["failure"]
            return pending["result"]

        try:
            result            = self._run_bootstrap(entry, interpreter)
            pending["result"] = result
        except BaseException as exc:
            pending["failure"] = exc
            raise
        finally:
            with self.lock:
                self.inflight.pop(inflight_key, None)
            pending["event"].set()

        if result.get("ok"):
            with self.lock:
                self.cache[cache_key] = (signature, result)

        return result
