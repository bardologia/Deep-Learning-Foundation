from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_ui_server import WebUIServer


class ServeEntry:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="DLR-TomoSAR control console")
        self.parser.add_argument("--host", default="127.0.0.1")
        self.parser.add_argument("--port", type=int, default=8765)

    def run(self) -> None:
        args   = self.parser.parse_args()
        server = WebUIServer(host=args.host, port=args.port)
        server.serve()


if __name__ == "__main__":
    ServeEntry().run()
