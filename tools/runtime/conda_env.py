
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing  import List, Optional

from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class CondaEnv:

    @staticmethod
    def _candidates(env_name: str) -> List[Path]:
        executable = Path(sys.executable).resolve()
        candidates : List[Path] = []

        for parent in executable.parents:
            if parent.name == "envs":
                candidates.append(parent / env_name / "bin" / "python")
                break

        for base in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path.home() / ".conda"):
            candidates.append(base / "envs" / env_name / "bin" / "python")

        return candidates

    @staticmethod
    def interpreter(env_name: str) -> Path:
        candidates = CondaEnv._candidates(env_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Conda environment '{env_name}' interpreter not found (searched: {searched}); it is required to dispatch the job.")


class CondaJobDispatcher:

    def __init__(self, env_name: str, logger: Logger, repo_root: Optional[Path] = None) -> None:
        self.env_name  = env_name
        self.logger    = logger
        self.repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]

    @staticmethod
    def _library_dir(interpreter: Path) -> str:
        return str(interpreter.parent.parent / "lib")

    @classmethod
    def _already_on_path(cls, interpreter: Path) -> bool:
        return cls._library_dir(interpreter) in os.environ.get("LD_LIBRARY_PATH", "").split(":")

    @classmethod
    def _runtime_env(cls, interpreter: Path) -> dict:
        env          = dict(os.environ)
        library_dir  = cls._library_dir(interpreter)
        library_path = env.get("LD_LIBRARY_PATH", "")

        if not cls._already_on_path(interpreter):
            env["LD_LIBRARY_PATH"] = library_dir + (":" + library_path if library_path else "")

        return env

    def dispatch(self, entry_relative_path: str, spec_payload: dict, spec_path: Path) -> None:
        FileIO.save_json(spec_payload, spec_path)

        interpreter = CondaEnv.interpreter(self.env_name)
        entry       = self.repo_root / entry_relative_path
        command     = [str(interpreter), str(entry), "--spec", str(spec_path)]
        env         = self._runtime_env(interpreter)
        state       = "already on LD_LIBRARY_PATH" if self._already_on_path(interpreter) else "prepended to LD_LIBRARY_PATH"

        self.logger.subsection(f"Dispatching '{entry_relative_path}' in env '{self.env_name}': {' '.join(command)}")
        self.logger.info(f"{self._library_dir(interpreter)} {state}; the child runs with LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']} and writes its own log beside the spec in {spec_path.parent}")
        subprocess.run(command, check=True, cwd=str(self.repo_root), env=env)
