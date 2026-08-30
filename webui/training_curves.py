"""Scalar training curves read from run TensorBoard event files for the web UI."""

from __future__ import annotations

import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from catalog_roots import CatalogRoots
from lru_cache      import LruCache
from web_logger     import WebLogger


class TrainingCurves:
    """Lists training runs and serves their downsampled scalar series.

    Attributes:
        logger : Console logger for discovery messages.
        roots  : Catalogued runs roots used to resolve and sandbox run paths.
        cache  : LRU cache of parsed scalar series keyed by run directory.
    """


    MAX_POINTS   = 1000
    CACHE_CAP    = 16
    DEFAULT_TAGS = ("loss/val", "loss/validation", "val/loss")

    def __init__(self, logger: WebLogger) -> None:
        """Stores the logger and prepares the runs-root resolver and scalar cache.

        Args:
            logger : Console logger for discovery messages.
        """

        self.logger = logger
        self.roots  = CatalogRoots()
        self.cache  = LruCache(self.CACHE_CAP)

    def runs(self, base: str) -> dict:
        """Lists runs holding TensorBoard event files under a catalogued root.

        Args:
            base : Runs root key or path accepted by the catalogue.

        Returns:
            Dictionary with ``ok``, the resolved ``root`` and a ``runs`` list of
            entries with absolute id, root-relative name and event mtime, newest
            first; ``ok`` is False with an ``error`` when the root cannot be opened.
        """

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
        """Returns downsampled scalar series for the requested runs and tag.

        When the requested tag is absent from every run a default loss-like tag is
        chosen instead.

        Args:
            run_ids : Absolute run directory paths to load.
            tag     : Preferred scalar tag to plot.

        Returns:
            Dictionary with ``ok``, the resolved ``tag``, the union of available
            ``tags``, and a ``series`` list of per-run step and value lists; ``ok`` is
            False with an ``error`` for unknown runs or runs without scalars.
        """

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
        """Returns the scalar series of a run, reloading only when events changed.

        Args:
            run_dir : Run directory holding a ``tensorboard`` subdirectory.

        Returns:
            Mapping of scalar tag to a list of (step, value) pairs.
        """

        tb_dir = run_dir / "tensorboard"
        stamp  = max((f.stat().st_mtime for f in tb_dir.glob("events.out.tfevents.*")), default=0.0)
        key    = str(run_dir)

        cached = self.cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
        accumulator.Reload()

        scalars = {}
        for tag in accumulator.Tags()["scalars"]:
            events       = accumulator.Scalars(tag)
            scalars[tag] = [(event.step, event.value) for event in events]

        self.cache.put(key, (stamp, scalars))
        return scalars

    def _downsample(self, points: list) -> tuple[list, list]:
        """Strides a series down to at most ``MAX_POINTS`` samples.

        The final point is always retained so the curve keeps its true endpoint.

        Args:
            points : List of (step, value) pairs in recording order.

        Returns:
            Tuple of integer steps and float values of the kept samples.
        """

        stride = max(1, math.ceil(len(points) / self.MAX_POINTS))
        kept   = points[::stride]
        if points and kept[-1] != points[-1]:
            kept.append(points[-1])

        steps  = [int(step) for step, _ in kept]
        values = [float(value) for _, value in kept]
        return steps, values

    @classmethod
    def _default_tag(cls, all_tags: list) -> str:
        """Returns the preferred plotting tag among the available scalar tags.

        Preference falls from the configured validation-loss names, to any tag naming
        both validation and loss, to any loss tag, to the first tag available.

        Args:
            all_tags : Sorted union of scalar tags across the selected runs.

        Returns:
            Tag name to plot by default.
        """

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
        """Returns the run directory for a request path, or None when not usable.

        Args:
            raw : Candidate run directory path from the client.

        Returns:
            Resolved path when it lies inside a catalogued root and holds a
            ``tensorboard`` directory, otherwise None.
        """

        if not raw:
            return None

        run_dir = Path(raw).resolve()
        if not self.roots.contains(run_dir):
            return None
        if not (run_dir / "tensorboard").is_dir():
            return None
        return run_dir
