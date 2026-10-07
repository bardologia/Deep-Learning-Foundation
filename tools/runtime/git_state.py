"""Repository revision and host provenance, recorded alongside run outputs.

Provides GitState, which reports the commit a process runs from and the
working-tree state of a dirty checkout, and RunProvenance, which gathers the
commit, host, interpreter, library versions and configuration sources every
completion marker carries.
"""

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
    """Reports the revision of the checkout this code is running from.

    Attributes:
        REPO_ROOT    : Root of the checkout the git commands run in.
        DIRTY_SUFFIX : Annotation commit() appends when tracked files carry uncommitted changes.
    """

    REPO_ROOT    = Path(__file__).resolve().parents[2]
    DIRTY_SUFFIX = "(dirty working tree)"

    @classmethod
    def _git(cls, *args: str, repo_root: Path | None = None) -> subprocess.CompletedProcess:
        """Runs one git command in a checkout, this repository's by default, and returns its completed process."""
        return subprocess.run(["git", *args], cwd=repo_root if repo_root is not None else cls.REPO_ROOT, capture_output=True, text=True, timeout=10)

    @classmethod
    def commit(cls, repo_root: Path | None = None) -> str:
        """Returns the short HEAD hash of a checkout, annotated when unclean.

        Only tracked files decide cleanliness, so untracked outputs and notes
        inside the checkout do not mark every run as dirty.

        Args:
            repo_root : Checkout to read, or None for this repository; a dependency checkout is read the same way.

        Returns:
            The 12-character commit hash, suffixed with '(dirty working tree)' when tracked files carry uncommitted changes or '(working tree state unknown)' when git status fails or times out, or an 'unavailable (...)' string naming why no commit could be read.
        """
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
        """Returns the working-tree state beyond the commit hash, for provenance capture.

        Tracked files decide whether the tree is dirty, as they do for
        commit(), so the diff saved with a dirty run rebuilds the code it
        executed from the recorded commit; untracked paths are listed either way.

        Returns:
            Mapping with "dirty" (whether tracked files carry uncommitted changes), "diff" (unified diff of those changes against HEAD, empty when clean) and "untracked" (paths git does not track); all falsy when git cannot be run or the source is not a checkout.

        Raises:
            RuntimeError : If git diff fails on a checkout whose tracked files carry changes.
        """
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

        if diff.returncode != 0:
            raise RuntimeError(f"git diff HEAD failed in {cls.REPO_ROOT}: {diff.stderr.strip()}")

        return {"dirty": True, "diff": diff.stdout, "untracked": untracked}


class RunProvenance:
    """Gathers where and with what code and libraries a process produced its outputs."""

    @staticmethod
    def collect() -> dict:
        """Returns the commit, host, interpreter, environment, library versions and configuration sources of this process.

        Library versions are read only for modules the process has already
        imported, so recording provenance never pulls torch into a process
        that does not use it. The configuration sources come from the ConfigCli
        that parsed this process's command line: the machine profile that
        applied, the defaults it rewrote and the typed overrides.

        Returns:
            Mapping with git_commit, host, python, conda_env, machine_profile, numpy, torch, cuda and config_sources; absent libraries map to None, and so do the sources of a process whose configuration no ConfigCli parsed.
        """
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
