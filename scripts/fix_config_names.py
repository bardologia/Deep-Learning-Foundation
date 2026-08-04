from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.monitoring.logger import Logger


class ConfigNameKeyFixer:
    TARGET_KEY = "model_name"
    OLD_KEYS   = ("backbone_name", "ae_model_name", "image_ae_model_name")
    FILENAMES  = ("model_config.json", "profile_autoencoder_config.json", "image_autoencoder_config.json")

    def __init__(self, root: Path) -> None:
        self.root   = Path(root)
        self.logger = Logger(log_dir="logs", name="fix_config_names")

    def discover(self) -> list:
        paths = []
        for name in self.FILENAMES:
            paths.extend(sorted(self.root.rglob(name)))
        return paths

    def fix_file(self, path: Path) -> bool:
        payload = json.loads(path.read_text(encoding="utf-8"))

        if self.TARGET_KEY in payload:
            return False

        present = [key for key in self.OLD_KEYS if key in payload]
        if not present:
            raise KeyError(f"{path} has no recognizable name key (expected one of {(self.TARGET_KEY,) + self.OLD_KEYS})")

        reordered = {self.TARGET_KEY: payload.pop(present[0]), **payload}
        path.write_text(json.dumps(reordered, indent=4), encoding="utf-8")

        return True

    def run(self) -> None:
        paths = self.discover()

        fixed   = 0
        skipped = 0
        for path in paths:
            if self.fix_file(path):
                fixed += 1
                self.logger.ok(f"fixed {path}")
            else:
                skipped += 1
                self.logger.info(f"ok {path}")

        self.logger.info(f"{fixed} renamed, {skipped} already on '{self.TARGET_KEY}', {len(paths)} files total under {self.root}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    ConfigNameKeyFixer(root).run()
