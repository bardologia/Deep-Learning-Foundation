"""Entry point running inference over backbone and JEPA runs under the runs directory.

Delegates to the shared ``InferenceEntry``, which handles the scheduler/worker split.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path


def main() -> None:
    """Runs the shared inference entry bound to the backbone run type."""
    from _inference_entry import InferenceEntry, RunType
    InferenceEntry(entry_script=Path(__file__).resolve(), run_type=RunType.BACKBONE).run()


if __name__ == "__main__":
    main()
