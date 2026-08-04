from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "webui") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "webui"))

from repomap_derive          import RepoMapDeriver
from tools.monitoring.logger import Logger


class SkeletonDeriver:

    def __init__(self) -> None:
        self.deriver = RepoMapDeriver(REPO_ROOT)
        self.out_dir = REPO_ROOT / "logs" / "repomap"
        self.logger  = Logger(log_dir="logs", name="derive_repomap_skeleton")

    def _write_skeleton(self) -> Path:
        return self.deriver.write_skeleton(self.out_dir / "derived_skeleton.json")

    def _write_drift(self) -> tuple[Path, dict]:
        curated = json.loads((REPO_ROOT / "webui" / "repomap_data.json").read_text(encoding="utf-8"))
        drift   = self.deriver.drift(curated)

        path = self.out_dir / "drift.json"
        path.write_text(json.dumps(drift, indent=2), encoding="utf-8")

        return path, drift

    def run(self) -> None:
        skeleton_path     = self._write_skeleton()
        drift_path, drift = self._write_drift()

        self.logger.info(f"skeleton  : {skeleton_path}")
        self.logger.info(f"drift     : {drift_path}")
        self.logger.info(f"missing   : {len(drift['missing_files'])} curated nodes point at files that no longer exist")
        self.logger.info(f"uncovered : {len(drift['uncovered_modules'])} modules not in any curated diagram")

        self.logger.close()


if __name__ == "__main__":
    SkeletonDeriver().run()
