from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.baselines         import TrackBaselines
from tools.monitoring.logger import Logger
from tools.sar               import StepParameterResolver, TrackParameterCollector, TrackParameters


class TrackParameterBackfiller:
    TRACK_DEPTH = 2

    def __init__(self, run_directory, logger: Logger, remap=None, overwrite=False, dry_run=False) -> None:
        self.run_directory  = Path(run_directory)
        self.meta_directory = self.run_directory / "meta"
        self.logger         = logger
        self.remap          = remap
        self.overwrite      = overwrite
        self.dry_run        = dry_run

    def _load_baselines(self) -> TrackBaselines:
        path = self.meta_directory / TrackBaselines.FILENAME

        if not path.is_file():
            raise FileNotFoundError(f"No {TrackBaselines.FILENAME} under {self.meta_directory}; cannot recover pass directories.")

        baselines = TrackBaselines.load(path)

        if not baselines.track_files:
            raise ValueError(f"{path} has no track_files; the dataset predates track-file persistence and cannot be backfilled from it.")

        return baselines

    def _load_polarisation(self) -> str:
        path = self.meta_directory / "config_state.json"

        if not path.is_file():
            raise FileNotFoundError(f"No config_state.json under {self.meta_directory}; cannot recover the dataset polarisation.")

        polarisation = json.loads(path.read_text(encoding="utf-8")).get("tomogram_config", {}).get("polarisation")

        if not polarisation:
            raise ValueError(f"{path} has no tomogram_config.polarisation; cannot select the matching pp_*.xml.")

        return str(polarisation)

    def _remap_path(self, path: str) -> str:
        if self.remap is None:
            return path

        old, new = self.remap
        if not path.startswith(old):
            raise ValueError(f"Remap prefix '{old}' does not match track path '{path}'.")

        return new + path[len(old):]

    def _pass_directory(self, track_file: str) -> Path:
        return Path(self._remap_path(track_file)).parents[self.TRACK_DEPTH]

    def _resolve_parameter_files(self, baselines: TrackBaselines, polarisation: str) -> dict:
        resolver    = StepParameterResolver()
        track_paths = {}

        for label, track_file in zip(baselines.labels, baselines.track_files):
            track_paths[label] = resolver.resolve_for_polarisation(self._pass_directory(track_file), polarisation)

        return track_paths

    def _collect(self, track_paths: dict) -> TrackParameters:
        return TrackParameterCollector(track_paths).collect()

    def run(self) -> bool:
        out_path = self.meta_directory / TrackParameters.FILENAME

        if out_path.is_file() and not self.overwrite:
            self.logger.info(f"skip {out_path} already present (use --overwrite to regenerate).")
            return True

        baselines    = self._load_baselines()
        polarisation = self._load_polarisation()
        track_paths  = self._resolve_parameter_files(baselines, polarisation)
        parameters   = self._collect(track_paths)

        self.logger.info(f"  {'Polarisation':20s}: {polarisation}")
        for line, value in parameters.describe().items():
            self.logger.info(f"  {line:20s}: {value}")

        if self.dry_run:
            self.logger.info(f"dry-run  would write {out_path}")
            return True

        parameters.save(out_path)
        self.logger.ok(f"wrote {out_path}  ({parameters.n_tracks} tracks)")

        return True


class BackfillBatch:
    def __init__(self, roots, remap=None, overwrite=False, dry_run=False) -> None:
        self.roots     = [Path(root) for root in roots]
        self.remap     = remap
        self.overwrite = overwrite
        self.dry_run   = dry_run
        self.logger    = Logger(log_dir="logs", name="backfill_track_parameters")

    def _discover_runs(self, root: Path) -> list:
        if (root / "meta" / TrackBaselines.FILENAME).is_file():
            return [root]

        return sorted({match.parent.parent for match in root.glob(f"**/meta/{TrackBaselines.FILENAME}")})

    def run(self) -> int:
        runs = [run for root in self.roots for run in self._discover_runs(root)]

        if not runs:
            self.logger.warning("No datasets with meta/baselines.json found under the given paths.")
            return 1

        failures = 0

        for run in runs:
            self.logger.subsection(str(run))
            try:
                TrackParameterBackfiller(run, self.logger, remap=self.remap, overwrite=self.overwrite, dry_run=self.dry_run).run()
            except Exception as error:
                failures += 1
                self.logger.error(f"FAIL {type(error).__name__}: {error}")

        self.logger.info(f"Done: {len(runs) - failures}/{len(runs)} datasets backfilled, {failures} failed.")

        return 1 if failures else 0


def _parse_remap(value: str | None) -> tuple | None:
    if value is None:
        return None

    if "=" not in value:
        raise argparse.ArgumentTypeError("--remap expects OLD=NEW, e.g. /ste/rnd=/home/bard/data")

    old, new = value.split("=", 1)
    return old, new


def main() -> None:
    parser = argparse.ArgumentParser(description="TEMPORARY: backfill meta/track_parameters.json for datasets generated before the track-parameter component existed, by re-reading the F-SAR INF/INF-RDP/pp_*.xml of each track recovered from meta/baselines.json.")
    parser.add_argument("paths", type=str, nargs="+", help="Dataset run directories, or a parent directory to scan recursively for meta/baselines.json.")
    parser.add_argument("--remap",     type=str, default=None, help="Translate the stored server prefix to where the data actually lives, as OLD=NEW (e.g. /ste/rnd=/home/bard/ste).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate track_parameters.json even if it already exists.")
    parser.add_argument("--dry-run",   action="store_true", help="Resolve and report without writing.")
    args = parser.parse_args()

    exit_code = BackfillBatch(
        roots     = args.paths,
        remap     = _parse_remap(args.remap),
        overwrite = args.overwrite,
        dry_run   = args.dry_run,
    ).run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
