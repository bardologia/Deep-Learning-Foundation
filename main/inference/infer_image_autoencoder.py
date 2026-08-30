"""Command-line entry point for inference on trained image-autoencoder runs.

Delegates to the shared ``InferenceEntry`` driver with the image autoencoder run
type, which resolves the run directory, rebuilds the model and executes the
image-autoencoder inference pipeline.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path


def main() -> None:
    """Runs the image-autoencoder inference entry for the requested run."""
    from _inference_entry import InferenceEntry, RunType
    InferenceEntry(entry_script=Path(__file__).resolve(), run_type=RunType.IMAGE_AE).run()


if __name__ == "__main__":
    main()
