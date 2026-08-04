from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.models_baseline.test_state_dict_baseline import BASELINE_PATH, MODEL_FAMILIES, StateDictSignature, _build_case
from tools.monitoring.logger                        import Logger


class BaselineGenerator:
    def __init__(self):
        self.logger = Logger(log_dir="logs", name="state_dict_baseline")

    def run(self) -> None:
        baseline = {}

        for family, (cases, factory) in MODEL_FAMILIES.items():
            baseline[family] = {}

            for key in sorted(cases):
                model, _config        = _build_case(factory, cases[key])
                baseline[family][key] = StateDictSignature.compute(model)
                self.logger.info(f"{family}/{key}: {baseline[family][key]['num_keys']} keys, {baseline[family][key]['num_parameters']} parameters")

        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        self.logger.info(f"Baseline written to {BASELINE_PATH}")


if __name__ == "__main__":
    BaselineGenerator().run()
