
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from tools.runtime.config_cli      import ConfigCli
from tools.runtime.machine_profile import MachineProfile


class GitState:

    REPO_ROOT    = Path(__file__).resolve().parents[2]
    DIRTY_SUFFIX = "(dirty working tree)"

    @classmethod
    def _git(cls, *args: str, repo_root: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=repo_root if repo_root is not None else cls.REPO_ROOT, capture_output=True, text=True, timeout=10)

    @classmethod
    def commit(cls, repo_root: Path | None = None) -> str:
        if repo_root is not None and not Path(repo_root).is_dir():
            return f"unavailable (no such directory: {repo_root})"

        try:
            head = cls._git("rev-parse", "--short=12", "HEAD", repo_root=repo_root)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unavailable (git not runnable: {type(exc).__name__})"

        if head.returncode != 0:
            reason = head.stderr.strip().splitlines()[-1] if head.stderr.strip() else "not a git checkout"
            return f"unavailable ({reason})"

        commit = head.stdout.strip()

        try:
            status = cls._git("status", "--porcelain", "--untracked-files=no", repo_root=repo_root)
        except subprocess.TimeoutExpired:
            return f"{commit} (working tree state unknown)"

        if status.returncode != 0:
            return f"{commit} (working tree state unknown)"
        if status.stdout.strip():
            return f"{commit} {cls.DIRTY_SUFFIX}"

        return commit

    @classmethod
    def dirty_state(cls) -> dict:
        try:
            status = cls._git("status", "--porcelain")
        except (OSError, subprocess.TimeoutExpired):
            return {"dirty": False, "diff": "", "untracked": []}

        if status.returncode != 0:
            return {"dirty": False, "diff": "", "untracked": []}

        lines     = [line for line in status.stdout.splitlines() if line.strip()]
        untracked = [line[3:] for line in lines if line.startswith("??")]
        tracked   = [line for line in lines if not line.startswith("??")]

        if not tracked:
            return {"dirty": False, "diff": "", "untracked": untracked}

        diff = cls._git("diff", "HEAD")

        return {"dirty": True, "diff": diff.stdout, "untracked": untracked}


class RunProvenance:

    @staticmethod
    def collect() -> dict:
        numpy   = sys.modules.get("numpy")
        torch   = sys.modules.get("torch")
        profile = MachineProfile.location()
        parsed  = ConfigCli.parsed

        return {
            "git_commit"      : GitState.commit(),
            "host"            : socket.gethostname(),
            "python"          : platform.python_version(),
            "conda_env"       : os.environ.get("CONDA_DEFAULT_ENV"),
            "machine_profile" : str(profile) if profile is not None and profile.is_file() else None,
            "numpy"           : numpy.__version__ if numpy is not None else None,
            "torch"           : torch.__version__ if torch is not None else None,
            "cuda"            : torch.version.cuda if torch is not None else None,
            "config_sources"  : parsed.sources_mapping() if parsed is not None else None,
        }
