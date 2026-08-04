from __future__ import annotations

import ast
import json
from pathlib import Path

from project_paths import ProjectPaths


class ConfigRegistry:

    DESCRIPTIONS_FILE = "config_descriptions.json"

    SECTION_TITLES = {
        "sar"              : "SAR Processing",
        "param_extraction" : "Parameter Extraction",
        "normalization"    : "Normalization",
        "dataset"          : "Dataset",
        "architectures"    : "Model Architectures",
        "training"         : "Training",
        "inference"        : "Inference",
        "benchmark"        : "Benchmark",
        "cross_validation" : "Cross-validation",
        "patch_sweep"      : "Patch Sweep",
        "tuning"           : "Tuning",
        "comparison"       : "Run Comparison",
        "diagnostics"      : "Diagnostics",
    }

    SECTION_ORDER = ["sar", "param_extraction", "normalization", "dataset", "architectures", "training", "inference", "benchmark", "cross_validation", "patch_sweep", "tuning", "comparison", "diagnostics"]

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths        = paths
        self.descriptions = self._load_descriptions()

    def _load_descriptions(self) -> dict:
        path = Path(__file__).resolve().parent / self.DESCRIPTIONS_FILE
        return json.loads(path.read_text(encoding="utf-8"))

    def _parse_module(self, path: Path) -> list[dict]:
        tree = ast.parse(path.read_text(encoding="utf-8"))

        classes = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not self._is_dataclass(node):
                continue
            fields = self._parse_fields(node)
            if not fields:
                continue
            classes.append({"name": node.name, "fields": fields})
        return classes

    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(target, ast.Name) and target.id == "dataclass":
                return True
            if isinstance(target, ast.Attribute) and target.attr == "dataclass":
                return True
        return False

    def _parse_fields(self, node: ast.ClassDef) -> list[dict]:
        fields   = []
        group    = 0
        prev_end = None
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue

            if prev_end is not None and item.lineno - prev_end > 1:
                group += 1
            prev_end = item.end_lineno

            name       = item.target.id
            annotation = self._safe_unparse(item.annotation)
            default    = self._format_default(item.value)

            fields.append({
                "name"    : name,
                "type"    : annotation,
                "default" : default,
                "group"   : group,
            })
        return fields

    def _format_default(self, value: ast.expr | None) -> str:
        if value is None:
            return "required"
        if isinstance(value, ast.Call):
            func     = value.func
            is_field = (isinstance(func, ast.Name) and func.id == "field") or (isinstance(func, ast.Attribute) and func.attr == "field")
            if is_field:
                for kw in value.keywords:
                    if kw.arg == "default_factory":
                        inner = self._safe_unparse(kw.value)
                        return f"{inner}()"
                    if kw.arg == "default":
                        return self._safe_unparse(kw.value)
                return "factory"
        return self._safe_unparse(value)

    def _safe_unparse(self, node: ast.expr | None) -> str:
        if node is None:
            return ""
        return ast.unparse(node)

    def _section_units(self) -> list[tuple[str, list[Path]]]:
        root  = self.paths.config_dir
        dirs  = [p for p in sorted(root.iterdir()) if p.is_dir()  and not p.name.startswith(("_", "."))]
        files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"]

        units: dict[str, list[Path]] = {}
        for d in dirs:
            units[d.name] = [p for p in sorted(d.rglob("*.py")) if p.name != "__init__.py"]
        for f in files:
            units[f.stem] = [f]

        known = [s for s in self.SECTION_ORDER if s in units]
        extra = [s for s in units if s not in self.SECTION_ORDER]
        return [(s, units[s]) for s in known + extra]

    def _rel_module(self, path: Path) -> str:
        return path.relative_to(self.paths.config_dir).with_suffix("").as_posix()

    def _attach_descriptions(self, cls: dict) -> None:
        entry       = self.descriptions["configs"][f"{cls['module']}::{cls['name']}"]
        cls["desc"] = entry["summary"]
        field_descs = entry["fields"]
        for field in cls["fields"]:
            field["desc"] = field_descs[field["name"]]

    def collect(self) -> list[dict]:
        groups = []
        for section, files in self._section_units():
            classes = []
            for path in files:
                for cls in self._parse_module(path):
                    cls["module"] = self._rel_module(path)
                    self._attach_descriptions(cls)
                    classes.append(cls)
            if not classes:
                continue
            groups.append({
                "module"  : section,
                "title"   : self.SECTION_TITLES.get(section, section.replace("_", " ").title()),
                "desc"    : self.descriptions["sections"][section],
                "classes" : classes,
            })
        return groups
