"""Repository revision reporting, recorded alongside run outputs for provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitState:
    """Reports the revision of the checkout this code is running from."""

    @staticmethod
    def commit() -> str:
        """Returns the short HEAD hash of the repository, annotated when unclean.

        Returns:
            The 12-character commit hash, suffixed with '(dirty working tree)'
            when uncommitted changes exist, or an 'unavailable' string when git
            cannot be run or the source is not a checkout.
        """
        repo_root = Path(__file__).resolve().parents[2]

        try:
            head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable (git not runnable)"

        if head.returncode != 0:
            return "unavailable (not a git checkout)"

        commit = head.stdout.strip()

        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, timeout=5)
        if status.returncode == 0 and status.stdout.strip():
            return f"{commit} (dirty working tree)"

        return commit

    @staticmethod
    def dirty_state() -> dict:
        """Returns the working-tree state beyond the commit hash, for provenance capture.

        Returns:
            Mapping with "dirty" (whether uncommitted tracked changes or untracked files exist), "diff" (unified diff of the tracked changes against HEAD, empty when clean) and "untracked" (paths git does not track); all falsy when git cannot be run or the source is not a checkout.
        """
        repo_root = Path(__file__).resolve().parents[2]

        try:
            status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return {"dirty": False, "diff": "", "untracked": []}

        if status.returncode != 0:
            return {"dirty": False, "diff": "", "untracked": []}

        lines     = [line for line in status.stdout.splitlines() if line.strip()]
        untracked = [line[3:] for line in lines if line.startswith("??")]

        if not lines:
            return {"dirty": False, "diff": "", "untracked": []}

        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=15)

        return {"dirty": True, "diff": diff.stdout if diff.returncode == 0 else "", "untracked": untracked}
