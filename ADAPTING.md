# Adapting the template

`README.md` runs the template on its example task. This file is about replacing that task
with yours. The steps come first, in the order a new project usually takes them; each names
the one file to touch. The sections after them are reference material for the steps.

## Steps

1. **Write your dataset.** In an importer script of your own, build a `DatasetFiles` from
   `pipelines/shared/dataset/dataset_files.py` and call its `write(inputs, targets)`. The
   contract is described below; nothing downstream needs to know where the arrays came from.
2. **Point the defaults at it.** Set `ReferenceLayout.DATASET_DIR` in
   `configuration/reference_layout.py`. On one machine only, rerun
   `./install.sh --skip-install` and pick the dataset in the data step instead, or pass
   `--paths.dataset_path <dir>` to a single launch.
3. **Pick a model.** Set `backbone_name` in `configuration/training/backbone.py`, or pass
   `--backbone_name <key>`. The keys are listed in `models/backbone/__init__.py` and
   compared in `notes/models/Model Zoo.md`. A new architecture is described under "Adding a
   model" below.
4. **Change the loss.** Edit `Loss.elementwise` in `pipelines/backbone/training/loss.py`.
   A new loss kind also needs its name in `LossKind`
   (`configuration/training/general/loss.py`); the launch forms and the benchmark's loss
   sweep read the kinds from there.
5. **Change the metrics.** Edit `Metrics.channel` and `Metrics.CHANNEL_KEYS` in
   `pipelines/backbone/inference/metrics.py`. A metric whose key starts with `rmse_`,
   `mae_`, `mse_` or `r2_`, or holds `loss`, is ranked without further work; any other
   prefix needs an orientation rule in `MetricOrientation` (`tools/metrics/scoring.py`).
6. **Adjust the training defaults.** Epochs, batch size, patch size and stride and the
   learning-rate scaling are the fields of `TrainingQueueConfig` in
   `configuration/training/general/run.py`.
7. **Rename the project** when it is no longer a template, using the list at the end.

Keep the example generator until your own data is in place: the test suite builds its
dataset with it (`test_data_dir` in `tests/conftest.py`). When you remove it, point that
fixture at a small copy of your data first.

## The dataset contract

```
<dataset>/
  dataset.json
  inputs.npy     (input_channels, rows, cols)   float32
  targets.npy    (target_channels, rows, cols)  float32
```

`dataset.json` holds seven keys, and `DatasetFiles.read` reads every one, so a missing key
stops the reader:

- `name`: a free string, usually the directory name.
- `input_channels`, `target_channels`: one name per channel of each array, in array order.
  The backbone is built with as many input and output channels as these lists hold, and the
  names label the normalization statistics, the per-channel losses, the metrics and the
  figure folders.
- `rows`, `cols`: the grid size both arrays share.
- `splits`: a map from `train`, `val` and `test` to a list of regions
  `[row_start, row_end, col_start, col_end]` in pixels. A split may hold several disjoint
  regions; each is cut into patches on its own. Cross-validation replaces the splits with
  its fold blocks.
- `generator`: a free mapping recording how the dataset was made.

`DatasetFiles.write` creates the directory, which must not exist yet, saves both arrays as
float32 and writes `dataset.json` last, so a half-written dataset never has a layout file.
A minimal importer:

```python
from pathlib import Path

import numpy as np

from pipelines.shared.dataset.dataset_files import DatasetFiles
from tools.data.regions                     import CropRegion, SplitRegions

inputs  = np.load("my_inputs.npy")
targets = np.load("my_targets.npy")
rows    = inputs.shape[1]
cols    = inputs.shape[2]

splits = SplitRegions(
    train = CropRegion(row_start=0,                row_end=int(rows * 0.7), col_start=0, col_end=cols),
    val   = CropRegion(row_start=int(rows * 0.7),  row_end=int(rows * 0.85), col_start=0, col_end=cols),
    test  = CropRegion(row_start=int(rows * 0.85), row_end=rows,             col_start=0, col_end=cols),
)

DatasetFiles(Path("datasets/mine"), "mine", ["a", "b"], ["y"], rows, cols, splits, {"source": "my_inputs.npy"}).write(inputs, targets)
```

The contract has no mask. Fill invalid pixels with values the loss can take, or multiply
the weight map of `PatchDataset` (`pipelines/backbone/dataset/datasets.py`) by a mask you
store beside the arrays: the weight map is the one place a pixel's share of the loss is
decided. Padding weighs 0, a train pixel 1, and a validation or test pixel 1 over the
number of patches covering it, so every scored pixel counts once.

## Loss, metrics and normalization in more detail

`Loss` compares the prediction with the normalized targets elementwise, averages each
target channel over the pixel weights of its patch and combines the channels with
`loss.channel_weights` (empty weighs every channel 1). It returns a `WeightedMean`
(`tools/training/aggregation.py`) per channel and for the total, which lets a whole split
be pooled as one per-pixel mean. A loss that is not elementwise, for example one on
spatial gradients, belongs in `Loss.__call__` next to the channel means and returns its own
`WeightedMean`.

Inference scores the stitched prediction in target units. `Metrics.compute` writes flat
keys, `<metric>_<channel>` for every channel and `<metric>_mean` for the keys in
`MEAN_KEYS`, and every comparison, seed aggregate and result table reads them. The
leaderboards rank on `HEADLINE_METRICS` in `pipelines/shared/comparison/comparison_report.py`.

Each input and target channel is normalized with its own location and scale, fitted on the
train split by `StatsComputer` (`pipelines/backbone/dataset/stats_computer.py`) with the
method `NormMethod` names: `zscore` (mean and standard deviation), `robust_iqr` (median and
interquartile range) or `min_max` (0.1 and 99.9 percentiles). The statistics are saved in
the run's `meta/normalization_stats.json` and reloaded by inference. A new method is a new
`NormMethod` value with its fit in `NormMethod.fit`, in
`configuration/normalization/general.py`. Augmentation (flips, quarter turns, input noise)
is `AugmentationConfig` in `configuration/dataset/general/dataset.py`.

## Adding a model

1. The module under `models/backbone/`. Reuse the blocks in `models/blocks.py`, mix in
   `OutputHeadMixin`, set `embedding_channels` and call `_build_output_head()` in the
   constructor, and end `forward` with `_head_forward(...)`.
2. Its config dataclass in `configuration/architectures/backbone.py`, with `in_channels`,
   `out_channels`, `shape_logger_types`, `tunable_lr_params`, `tunable_arch_params` and
   `get_param_groups`, following the existing configs.
3. One entry in `BACKBONE_MODEL_REGISTRY` and one in `BACKBONE_CONFIG_REGISTRY` in
   `models/backbone/__init__.py`, under the same key. A model that needs a fixed input size
   also goes into `BACKBONE_IMAGE_SIZE_MODELS`.
4. A card in `BackboneModelLibrary._families()` (`webui/backbone_model_library.py`) and a
   note `notes/models/<card name>.md`, which the Architectures page shows.
5. Its entry in `tests/state_dict_baseline.json` (see the gates below).

`tests/webui/test_backbone_model_library.py` checks that every registered model has a card
and a note, and `tests/models_backbone/test_backbone_architectures.py` runs the forward pass
and the optimizer groups of every model.

## Adding an entry point

`main/` holds wiring only. `main/processing/generate_dataset.py` is the short form: build
the entry config with `ConfigCli(<EntryConfig>()).apply()` and hand it to a session class.
`main/training/train_backbone.py` shows the GPU form, which pins `CUDA_VISIBLE_DEVICES`
before torch is imported. The console needs four registrations:

1. `ProjectPaths.SCRIPT_DIRS` in `webui/project_paths.py`, the key and its directory under
   `main/`.
2. `ScriptCatalog.META` in `webui/script_catalog.py`: title, category and purpose.
3. A layout in `LaunchLayout.LAYOUTS` (`webui/launch_layout.py`) that claims every field of
   the entry config.
4. A description for every new config field in `webui/config_descriptions.json`.

A `JobDescriber.SPECS` row in `webui/job_describer.py` is optional and names the job in the
console and in notifications. A per-run analysis subclasses `AnalysisRun` in
`pipelines/backbone/inference/analysis/run_batch.py`; `main/analysis/probe_layers.py` is
the short example.

## The gates

These tests fail loudly when the code and the console or the documentation drift apart:

- **Every config field has a launch control.** `LaunchLayout` refuses a page whose layout
  leaves a field unclaimed, claims it twice or names an unknown one
  (`tests/webui/test_launch_layout.py`). Add the field to a section, or to a `hidden` panel.
- **Every config field has a description, and every description a field**
  (`tests/webui/test_config_registry_descriptions.py`). Edit `config_descriptions.json`.
- **Old checkpoints still load.** `tests/state_dict_baseline.json` records the state-dict
  keys, shapes and parameter count of every model
  (`tests/models_baseline/test_state_dict_baseline.py`). Regenerate it with
  `scripts/generate_state_dict_baseline.py` only for an intentional architecture change,
  in its own commit.
- **Every path default sits below a reference root**
  (`tests/configuration/test_reference_layout.py`), so the machine profile can rebase it.
  Write path defaults relative to the repository, below a `ReferenceLayout` root.
- **Every metric key has an orientation.** A key no rule of `MetricOrientation` covers stops
  the ranking (`tests/loss_metrics/test_scoring.py`).
- **Every figure a page names exists** (`tests/webui/test_hairline_figures.py`). The console
  figures are registered in `ProjectFigures.CATALOG` in `webui/static/js/hairline_project.js`.

## Renaming the project

The name is written out in these places:

- `pyproject.toml` (`name`, `description`).
- The console: `webui/serve.py`, `webui/web_ui_server.py` (start-up banner), `webui/run.sh`,
  `webui/notifier.py` (test notification, asserted in `tests/webui/test_notifier.py`) and
  `webui/static/index.html` (title, brand, footer).
- The environment name `ml-template`: `install.sh` and `MachineProfile.DEFAULTS` in
  `tools/runtime/machine_profile.py`.
- The profile location `~/.config/ml-template/` and the `MLTEMPLATE_` variables:
  `tools/runtime/machine_profile.py`, `tools/runtime/detacher.py`,
  `tools/runtime/cuda_mask.py`, `install.sh` and `webui/run.sh`.
- The console port 8772: `ReferenceLayout.CONSOLE_PORT` in
  `configuration/reference_layout.py`, `webui/serve.py`, and the screenshot harness in
  `webui/tools/visual/`.
- The systemd scope prefix `ml-template-webui` (`webui/process_manager.py`) and the browser
  storage prefix `mlt.` (`webui/static/js/configs.js`).
