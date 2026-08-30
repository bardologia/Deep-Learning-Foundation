"""One-off migration of the legacy model-name key in persisted model configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.monitoring.logger import Logger


class ConfigNameKeyFixer:
    """Renames the legacy model-name key to `model_name` in persisted model configs.

    Attributes:
        root   : Runs root scanned recursively.
        apply  : Whether the files are rewritten instead of only reported.
        logger : Logger for the plan and results.
    """

    TARGET_KEY = "model_name"
    OLD_KEYS   = ("backbone_name", "ae_model_name", "image_ae_model_name")
    FILENAMES  = ("model_config.json", "profile_autoencoder_config.json", "image_autoencoder_config.json")

    def __init__(self, root: Path, apply: bool = False) -> None:
        """Binds the migration to a runs root.

        Args:
            root  : Runs root, or a single run directory.
            apply : Rewrite the files instead of only printing the plan.

        Raises:
            NotADirectoryError : If the root is not a directory.
        """
        self.root   = Path(root)
        self.apply  = apply
        self.logger = Logger(log_dir="logs", name="fix_config_names")

        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}; point the migration at the runs root (or a single run) whose {' / '.join(self.FILENAMES)} must be rewritten")

    def discover(self) -> list[Path]:
        """Returns the persisted model config files found under the root, sorted."""
        wanted = set(self.FILENAMES)

        return sorted(path for path in self.root.rglob("*") if path.name in wanted)

    def plan(self, paths: list[Path]) -> tuple[list[tuple[Path, dict]], int]:
        """Decides which files need the rename and builds their new payloads.

        Args:
            paths : Discovered config files.

        Returns:
            Tuple of (list of (path, rewritten payload) pairs, number of files already
            on the target key).

        Raises:
            KeyError : If any file carries neither the target key nor a legacy one, in which case nothing is rewritten.
        """
        pending   = []
        offenders = []
        skipped   = 0

        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))

            if self.TARGET_KEY in payload:
                skipped += 1
                self.logger.info(f"ok {path}")
                continue

            present = [key for key in self.OLD_KEYS if key in payload]

            if not present:
                offenders.append(str(path))
                continue

            pending.append((path, {self.TARGET_KEY: payload.pop(present[0]), **payload}))

        if offenders:
            raise KeyError(f"{len(offenders)} of {len(paths)} discovered file(s) carry no recognizable name key (expected one of {(self.TARGET_KEY,) + self.OLD_KEYS}): {', '.join(offenders)}. Nothing was rewritten; move or repair them and run the migration again")

        return pending, skipped

    def rewrite(self, pending: list[tuple[Path, dict]]) -> None:
        """Writes the planned payloads, or only reports them outside apply mode.

        Args:
            pending : Pairs of (path, rewritten payload).
        """
        for path, payload in pending:
            if self.apply:
                path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

            self.logger.ok(f"{'fixed' if self.apply else 'would fix'} {path}")

    def run(self) -> None:
        """Discovers, plans and applies the rename, then reports the counts."""
        paths            = self.discover()
        pending, skipped = self.plan(paths)

        self.rewrite(pending)

        self.logger.info(f"{len(pending)} {'renamed' if self.apply else 'to rename'}, {skipped} already on '{self.TARGET_KEY}', {len(paths)} files total under {self.root}")

        if not self.apply:
            self.logger.info("dry run only; re-run with --apply to rewrite the files")


def main() -> None:
    """Parses the command line and runs the key migration."""
    parser = argparse.ArgumentParser(description="TEMPORARY: rename the legacy backbone_name / ae_model_name / image_ae_model_name key to model_name in the persisted model configs of existing runs. Without --apply the script only reports what it would rewrite.")
    parser.add_argument("root", type=Path, help="Runs root (or a single run directory) scanned recursively for the persisted model configs.")
    parser.add_argument("--apply", action="store_true", help="Rewrite the files; without it the script only prints the plan.")
    args = parser.parse_args()

    ConfigNameKeyFixer(args.root, apply=args.apply).run()


if __name__ == "__main__":
    main()
