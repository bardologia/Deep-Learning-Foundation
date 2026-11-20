# ML-Template

A starting point for supervised deep-learning projects. It provides the parts most
projects rebuild from scratch: a configuration layer with command-line overrides, logging,
run tracking and resource monitoring, training, inference and experiment pipelines, a zoo
of dense-prediction backbones, analysis tools, and a local web console that launches the
entry points, streams their output and browses their results.

The bundled example task is dense per-pixel regression on a 2-D grid: four float input
channels in, two float target channels out. A generator writes the example dataset in
under a second, so every part of the template runs end to end on a fresh clone without
external data. `ADAPTING.md` explains how to replace the example with your own data,
model, loss and metrics.

## Quick start

```bash
./install.sh                                     # conda env, dependencies, machine profile
conda activate ml-template
python main/processing/generate_dataset.py       # writes datasets/synthetic
python main/training/train_backbone.py --seeds "[0]" --training.epochs 5
python main/inference/infer_backbone.py --runs_dir runs/backbone
webui/run.sh                                     # console at http://127.0.0.1:8772
```

The rest of this file explains each step.

## 1. Environment

`./install.sh` finds or installs Miniconda, creates the conda environment `ml-template`
with Python 3.11, installs the torch build that matches the NVIDIA driver
(`cu130`, `cu128` or `cpu`), installs the pinned requirements and the optional extras, and
then starts the setup wizard (`scripts/setup_machine.py`). The wizard asks where datasets,
runs and results live, which GPUs the entry points may use, how many data-loader workers to
allow, the console address, and whether to send ntfy notifications or mail reports. Every
question has a detected default. The answers are written to the machine profile at
`~/.config/ml-template/machine.json` together with a shell twin, `machine.env`.

Run the installer again whenever something changes; each question then defaults to the
current answer. Useful options:

```bash
./install.sh --yes                 # accept every default
./install.sh --skip-install        # rerun only the questions
./install.sh --env myenv           # use another conda environment name
./install.sh --torch-index cpu     # choose the torch build by hand
```

When `nvidia-smi` is installed but fails, which usually means the driver no longer matches
the running kernel, the installer stops instead of guessing a build: fix the driver or pass
`--torch-index`. The wizard's final check imports torch in the new environment and quotes
torch's own reason when it cannot use a GPU the driver lists.

The same setup by hand:

```bash
conda create -n ml-template python=3.11
conda activate ml-template
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install onnx==1.22.0 pytest==9.0.3 pyflakes==3.4.0
./install.sh --skip-install
```

`requirements.txt` mirrors the pinned dependencies of `pyproject.toml`. The optional extras
are `export` (onnx, for the ONNX export entry; its tests skip without it) and `dev` (pytest
and pyflakes). Nothing has to be installed as a package: every entry point puts the
repository root on `sys.path`, so run scripts from the repository root.

Training falls back to the CPU with a warning when no CUDA device is visible. Inference
uses `--inference.device`, default `cuda`; pass `--inference.device cpu` on a machine
without a GPU.

## 2. The example dataset

```bash
python main/processing/generate_dataset.py
```

The generator writes `datasets/synthetic/` in the dataset contract every pipeline reads:

```
dataset.json   name, input and target channel names, grid size, split regions, generator settings
inputs.npy     (input_channels, rows, cols)   float32
targets.npy    (target_channels, rows, cols)  float32
```

The inputs are smooth random fields on a 256 x 256 grid. The target `pointwise` equals
tanh(x0 x1) + 0.5 x2, and the target `neighbourhood` is the window mean of x3 squared minus
the window mean of x0. The train, validation and test splits are row bands. The generator
refuses an existing output directory; delete it to generate again. The settings are the
fields of `GenerationConfig` in `configuration/generation.py`.

## 3. Configuration

Every default lives in a dataclass under `configuration/`. Each entry point builds its
entry config and hands it to `ConfigCli` (`tools/runtime/config_cli.py`), which exposes
every leaf field as a command-line flag:

```bash
python main/training/train_backbone.py --help-config
python main/training/train_backbone.py --backbone_name unet --training.batch_size 32 --seeds 0,1,2
```

Nested fields use dots, values are coerced to the declared type, and an unknown flag stops
the run. To change a default for good, edit the dataclass; the scripts under `main/` hold
no values.

The machine profile retargets the checkout-relative defaults: a path below one of the roots
named in `configuration/reference_layout.py` (`datasets/`, `runs/`, `results/`) is rebased
onto the profile's root, and the GPU lists and worker limits follow the profile. A
command-line flag always wins over the profile. `MLTEMPLATE_MACHINE_PROFILE` names another
profile file, and an empty value disables the profile, which is what the tests do.

Every run log opens with a table of where each setting came from: the default, the machine
profile or the command line.

## 4. Training

```bash
python main/training/train_backbone.py --seeds "[0]" --training.epochs 5
```

The default backbone is `resunet`, trained on 32 x 32 patches at stride 16 with batch size
64 and one run per seed in `seeds` (default `[0, 1]`, fanned out over the GPUs in `gpus`).
The loss (`loss.kind`: `mse`, `l1` or `huber`) is computed on normalized targets and
weighted per pixel, so the validation and test losses are per-pixel means over the whole
split. Inputs and targets are normalized per channel with statistics fitted on the train
split (`normalization.inputs`, `normalization.targets`: `zscore`, `robust_iqr` or
`min_max`).

A run lands in `runs/backbone/<model>-<loss>-<augmentation>_<timestamp>/`, for example
`resunet-mse-hv_20261120_101500`. It holds `best_model.pt`, `last.pt`, the TensorBoard
events, `complete.json` once training finished, the resolved configuration and provenance
under `docs/`, and the dataset configuration, normalization statistics and run summary
under `meta/`.

With `--resume` (on by default) a relaunch reuses a run that holds `complete.json` and
restarts one that does not. `--training.resume true` continues an unfinished run from
`last.pt` instead. `--resume_from <path>` finishes a broken training sweep, benchmark,
cross-validation or tuning run with the configuration it recorded, running only the
missing or unfinished units.

`--trials_enabled true --trials_mode <mode>` fans one launch out into a set of trials:
`architecture` (one run per listed backbone), `patch` (patch sizes), `augmentation`,
`stability` (warmup and gradient clipping), `ema` and `amp`.

## 5. Inference

```bash
python main/inference/infer_backbone.py --runs_dir runs/backbone
```

The entry finds every run below `--runs_dir` and dispatches one inference job per run over
the GPU pool. Each job stitches the patch predictions of the test split
(`--inference.split`) with a tapering window, inverts the target normalization and writes
`inference/<timestamp>/` into the run:

```
metrics.json     rmse_, mae_, bias_ and r2_<channel> per target channel, rmse_mean, mae_mean, r2_mean, n_pixels
report.md        the metrics and the run settings
maps/            predictions.npy and targets.npy in target units
figures/         target, prediction and absolute error maps per target channel
complete.json    the scored checkpoint
```

A run that already holds a completed inference of the requested split and checkpoint is
skipped; `--skip_completed false` scores it again.

## 6. Experiments and analysis

Three experiment entries under `main/experiments/` wrap training and inference:

- `benchmark.py` trains and scores several backbones and loss kinds and ranks them.
- `cross_validate.py` runs k-fold cross-validation over row blocks of the dataset.
- `tune.py` runs an Optuna study over learning rates, architecture fields and batch size.

Their defaults are sized for a real study: the benchmark trains all twenty backbones, the
cross-validation ten folds and the tuning a hundred trials per backbone, each under two
seeds or for tens of epochs. A first launch on the example dataset can be much smaller:

```bash
python main/experiments/benchmark.py --seeds "[0]" --training.epochs 2 --skip_models unet_skip,resunet,attention_unet,unetplusplus,linknet,swin_unet,transunet,unetr,deeplabv3plus,segformer,convnext_unet,dense_unet,hrnet,multires_unet,fpn,u2net,local_cnn,nafnet
python main/experiments/cross_validate.py --backbone_name unet --folds.n_folds 3 --seeds "[0]" --training.epochs 2
python main/experiments/tune.py --model unet --tuning.n_trials 3 --tuning.n_epochs 2
```

The benchmark line keeps unet and pixel_mlp; the console's model toggle builds the same
skip list by clicking.

The eighteen entries under `main/analysis/` read finished runs: trial, seed and run
comparisons, weight and activation x-rays, receptive field measurement, input attribution
and stress tests, layer probes, loss-landscape maps, training animations, a training-step
profiler, and exporters for reports, figures, TensorBoard plots, result tables and ONNX.
Every entry takes `--help-config`. The training animation needs epoch snapshots, which
training writes only when asked: train with `--training.snapshot_every_n_epochs 1` (or any
N) before running `animate_training.py`.

## 7. The web console

```bash
webui/run.sh          # http://127.0.0.1:8772
webui/run.sh 9000     # another port
```

`run.sh` starts the console with the interpreter recorded in the machine profile and stops
with a pointer to `./install.sh` when there is no profile. The console lists the entry
points, renders each configuration as a form with the profile's values in place, shows the
exact command, launches or queues it and streams the output live. It also carries server
and GPU telemetry with watchdogs, saved runs and config templates, experiment sequences,
a results browser, a TensorBoard page, the backbone architectures with their notes, and
model inspection pages. `webui/README.md` documents every tab.

## 8. Tests

```bash
python -m pytest tests --ignore=tests/webui -q
python -m pytest tests/webui -q
```

The suite holds about 3400 tests and builds its own small copy of the example dataset in a
temporary directory, so it passes before the generator has ever run. A few tests use the
GPU and skip without one. On a shared machine, check `nvidia-smi` before trusting a red
run: a card filled by another process fails those tests rather than skipping them.

## Repository map

```
install.sh      one-command setup: conda, environment, dependencies, machine profile
main/           entry points, one per task; wiring only
configuration/  dataclasses holding every default
pipelines/      the pipelines behind the entry points
models/         the backbones and their shared blocks
tools/          shared components: logger, tracker, monitors, training, metrics, reporting
webui/          the console: standard-library HTTP server and static frontend
notes/models/   one note per backbone, shown on the console's Architectures page
scripts/        setup wizard and maintenance scripts
tests/          the test suite, one directory per area
datasets/ runs/ results/ logs/   generated output, ignored by git
```

## Housekeeping

`./clean.sh` removes `__pycache__` directories and `.pytest_cache`. Formats change without
migration: datasets, runs and inference outputs written by an older checkout are
regenerated, not converted, because the readers stop on the first missing or renamed
field. `scripts/generate_state_dict_baseline.py` regenerates
`tests/state_dict_baseline.json`, the record that guards checkpoint compatibility; run it
only for an intentional architecture change and commit it on its own.
