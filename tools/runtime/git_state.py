from __future__ import annotations

import subprocess
from pathlib import Path


class GitState:
    @staticmethod
    def commit() -> str:
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
