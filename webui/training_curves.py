from __future__ import annotations

import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from catalog_roots import CatalogRoots
from web_logger    import WebLogger


class TrainingCurves:

    MAX_POINTS   = 1000
    DEFAULT_TAGS = ("loss/val", "loss/validation", "val/loss")

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.roots  = CatalogRoots()
        self.cache  = {}

    def runs(self, base: str) -> dict:
        root, error = self.roots.open(base)
        if error:
            return {"ok": False, "error": error, "runs": []}

        run_dirs = {}
        for event_file in root.rglob("tensorboard/events.out.tfevents.*"):
            run_dir = event_file.parent.parent
            mtime   = event_file.stat().st_mtime
            key     = str(run_dir)
            if key not in run_dirs or mtime > run_dirs[key]:
                run_dirs[key] = mtime

        runs = [{"id": key, "name": str(Path(key).relative_to(root)), "mtime": mtime} for key, mtime in run_dirs.items()]
        runs.sort(key=lambda run: run["mtime"], reverse=True)

        self.logger.info(f"curves: {len(runs)} training runs under {root}")
        return {"ok": True, "root": str(root), "runs": runs}

    def curves(self, run_ids: list, tag: str) -> dict:
        loaded = []
        for run_id in run_ids:
            run_dir = self._run_dir(run_id)
            if run_dir is None:
                return {"ok": False, "error": f"unknown training run: {run_id}"}
            loaded.append((run_dir, self._scalars(run_dir)))

        all_tags = sorted({name for _, scalars in loaded for name in scalars})
        if not all_tags:
            return {"ok": False, "error": "the selected runs contain no scalar series"}

        if tag not in all_tags:
            tag = self._default_tag(all_tags)

        series = []
        for run_dir, scalars in loaded:
            points = scalars.get(tag)
            if points is None:
                continue
            steps, values = self._downsample(points)
            series.append({"id": str(run_dir), "name": run_dir.name, "steps": steps, "values": values})

        return {"ok": True, "tag": tag, "tags": all_tags, "series": series}

    def _scalars(self, run_dir: Path) -> dict:
        tb_dir = run_dir / "tensorboard"
        stamp  = max((f.stat().st_mtime for f in tb_dir.glob("events.out.tfevents.*")), default=0.0)
        key    = str(run_dir)

        cached = self.cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]

        accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
        accumulator.Reload()

        scalars = {}
        for tag in accumulator.Tags()["scalars"]:
            events       = accumulator.Scalars(tag)
            scalars[tag] = [(event.step, event.value) for event in events]

        self.cache[key] = (stamp, scalars)
        return scalars

    def _downsample(self, points: list) -> tuple[list, list]:
        stride = max(1, math.ceil(len(points) / self.MAX_POINTS))
        kept   = points[::stride]
        if points and kept[-1] != points[-1]:
            kept.append(points[-1])

        steps  = [int(step) for step, _ in kept]
        values = [float(value) for _, value in kept]
        return steps, values

    @classmethod
    def _default_tag(cls, all_tags: list) -> str:
        for preferred in cls.DEFAULT_TAGS:
            if preferred in all_tags:
                return preferred
        for tag in all_tags:
            if "val" in tag.lower() and "loss" in tag.lower():
                return tag
        for tag in all_tags:
            if "loss" in tag.lower():
                return tag
        return all_tags[0]

    def _run_dir(self, raw: str) -> Path | None:
        if not raw:
            return None

        run_dir = Path(raw).resolve()
        if not self.roots.contains(run_dir):
            return None
        if not (run_dir / "tensorboard").is_dir():
            return None
        return run_dir
