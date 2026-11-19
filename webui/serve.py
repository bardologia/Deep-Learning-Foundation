
from __future__ import annotations

import argparse
import faulthandler
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_ui_server import WebUIServer


class ServeEntry:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="ML Template control console")
        self.parser.add_argument("--host", default="127.0.0.1")
        self.parser.add_argument("--port", type=int, default=8772)
        self.parser.add_argument("--root", type=Path, default=None, help="project root the console operates on (jobs, logs, runs); defaults to the repository holding this file")

    def run(self) -> None:
        faulthandler.register(signal.SIGUSR1, all_threads=True)

        args   = self.parser.parse_args()
        server = WebUIServer(host=args.host, port=args.port, root=args.root)
        server.serve()


if __name__ == "__main__":
    ServeEntry().run()
