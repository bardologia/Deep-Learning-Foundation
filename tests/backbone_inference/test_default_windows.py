from __future__ import annotations

import pytest

from configuration.diagnostics                               import InputAttributionConfig, ReceptiveFieldConfig
from configuration.generation                                import GenerationConfig
from pipelines.backbone.inference.analysis.input_attribution import InputAttributionRun
from pipelines.backbone.inference.analysis.receptive_field   import RunReceptiveField
from pipelines.processing.generation                         import DatasetGenerator

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import SilentLogger


@pytest.mark.slow
def test_the_window_analyses_run_at_their_defaults_on_the_example_grid(tmp_path):
    config = GenerationConfig(output_directory=tmp_path / "synthetic")
    DatasetGenerator(config, logger=SilentLogger()).run()
    run_dir = build_run_directory(tmp_path, config.output_directory)

    RunReceptiveField(run_dir, ReceptiveFieldConfig(), StageLogger()).run()
    InputAttributionRun(run_dir, InputAttributionConfig(), StageLogger()).run()

    assert (run_dir / ReceptiveFieldConfig().output_subdir).is_dir()
    assert (run_dir / InputAttributionConfig().output_subdir).is_dir()
