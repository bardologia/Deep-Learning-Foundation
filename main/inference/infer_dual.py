from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path


def main() -> None:
    from _inference_entry import InferenceEntry, RunType
    InferenceEntry(entry_script=Path(__file__).resolve(), run_type=RunType.DUAL).run()


if __name__ == "__main__":
    main()
