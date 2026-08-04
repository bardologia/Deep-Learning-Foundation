from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from tools.runtime.run_tag import RunTag


class Detacher:

    ENV_FLAG = "TOMOSAR_DETACHED"
    FLAGS    = ("--detach", "--nohup")

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)

    def requested(self, argv: list[str] | None = None) -> bool:
        argv = sys.argv[1:] if argv is None else argv
        return any(flag in argv for flag in self.FLAGS)

    def active(self) -> bool:
        return os.environ.get(self.ENV_FLAG) == "1"

    def ensure(self) -> None:
        if self.active():
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            return

        if not self.requested():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        stamp    = RunTag.now()
        name     = Path(sys.argv[0]).stem
        log_path = self.log_dir / f"{name}_{stamp}.out"

        env                     = dict(os.environ)
        env[self.ENV_FLAG]      = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("FORCE_COLOR", "1")
        env.setdefault("COLUMNS", "120")

        with open(log_path, "ab") as sink:
            process = subprocess.Popen(
                [sys.executable, "-u", *sys.argv],
                cwd               = os.getcwd(),
                stdout            = sink,
                stderr            = subprocess.STDOUT,
                stdin             = subprocess.DEVNULL,
                env               = env,
                start_new_session = True,
            )

        print(f"detached {name} as pid {process.pid}, immune to hangup")
        print(f"output: {log_path}")
        print(f"follow: tail -f {log_path}")

        raise SystemExit(0)
