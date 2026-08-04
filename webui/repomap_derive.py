from __future__ import annotations

import ast
import json
from pathlib import Path


class ImportGraph:

    ROOTS = ("main", "pipelines", "tools", "models", "configuration", "webui")

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def _modules(self) -> list[Path]:
        modules = []
        for root in self.ROOTS:
            base = self.repo_root / root
            if not base.is_dir():
                raise FileNotFoundError(f"Source root {base} does not exist; the import graph cannot cover {root}")
            modules += [path for path in base.rglob("*.py") if "__pycache__" not in path.parts and "node_modules" not in path.parts]

        return sorted(modules)

    def _module_name(self, path: Path) -> str:
        return str(path.relative_to(self.repo_root).with_suffix("")).replace("/", ".")

    @staticmethod
    def _package_parts(module_name: str) -> list[str]:
        parts = module_name.split(".")
        return parts[:-1] if parts[-1] != "__init__" else parts[:-2] + [parts[-2]]

    def _imports_of(self, path: Path, module_name: str) -> set[str]:
        tree    = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    imports.add(node.module)
                elif node.level > 0:
                    package = self._package_parts(module_name)
                    base    = package[: len(package) - (node.level - 1)]
                    if not base:
                        continue
                    if node.module:
                        imports.add(".".join(base + node.module.split(".")))
                    else:
                        imports.update(".".join(base + [alias.name]) for alias in node.names)

        return imports

    def build(self) -> dict[str, list[str]]:
        modules  = self._modules()
        by_name  = {self._module_name(path): path for path in modules}
        prefixes = tuple(f"{root}." for root in self.ROOTS)

        graph = {}
        for name, path in by_name.items():
            internal = set()
            for imported in self._imports_of(path, name):
                if name.startswith("webui.") and not imported.startswith(prefixes) and f"webui.{imported}" in by_name:
                    imported = f"webui.{imported}"

                if not (imported in by_name or imported.startswith(prefixes) or imported in self.ROOTS):
                    continue

                target = imported
                while target and target not in by_name:
                    target = target.rpartition(".")[0]
                if target:
                    internal.add(target)

            graph[name] = sorted(internal - {name})

        return graph


class RepoMapDeriver:

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.graph     = ImportGraph(repo_root)

    def skeleton(self) -> dict:
        edges   = self.graph.build()
        folders = {}

        for module, targets in edges.items():
            folder = module.split(".")[0]
            entry  = folders.setdefault(folder, {"folder": folder, "nodes": [], "edges": []})

            entry["nodes"].append({"id": module, "module": module.replace(".", "/") + ".py"})
            entry["edges"] += [{"from": module, "to": target} for target in targets if target.split(".")[0] == folder]

        return {"folders": [folders[name] for name in sorted(folders)]}

    def drift(self, curated: dict) -> dict:
        missing_files = []
        for folder in curated["folders"]:
            for diagram in folder["diagrams"]:
                for node in diagram["nodes"]:
                    module = node.get("module")
                    if not module:
                        continue

                    path       = self.repo_root / module
                    as_package = path.with_suffix("") if module.endswith(".py") else None

                    if not path.exists() and not (as_package is not None and as_package.is_dir()):
                        missing_files.append({"folder": folder["folder"], "diagram": diagram["key"], "node": node["id"], "module": module})

        curated_modules = {
            node.get("module")
            for folder in curated["folders"]
            for diagram in folder["diagrams"]
            for node in diagram["nodes"]
            if node.get("module")
        }

        uncovered = []
        for module in self.graph.build():
            path = module.replace(".", "/") + ".py"
            top  = module.split(".")[0]
            if top == "webui" or module.endswith("__init__"):
                continue
            if path not in curated_modules:
                uncovered.append(path)

        return {"missing_files": missing_files, "uncovered_modules": sorted(uncovered)}

    def write_skeleton(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.skeleton(), indent=2), encoding="utf-8")

        return output_path
