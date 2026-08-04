# ML-Template

A project skeleton for supervised deep learning, lifted out of a working research
codebase. It carries the configuration layer, the dataset, training and inference
pipelines, the experiment and analysis entry points, a local web console and about 4000
tests. The domain code is gone; what is left is the machinery around a model.

The example task is per-pixel Gaussian-mixture estimation along a z axis from a stack of
co-registered 2-D channels, and it comes with its own dataset generator. It is there so
that every part of the skeleton runs end to end on a fresh clone with no external data,
and so that each piece has a concrete thing to point at. Replacing it is the subject of
`ADAPTING.md`.

This file is the setup guide. Follow it top to bottom.

## What you need

A Linux machine with an NVIDIA GPU and driver for training and inference at a useful
speed. The tests, the dataset generator and the web console run fine on CPU. Training
falls back to CPU on its own when no CUDA device is visible; inference has to be told,
with `--inference.device cpu`. You also need conda.

## 1. Environment

```bash
conda create -n Dune python=3.11
conda activate Dune

pip install -r requirements.txt
pip install "jax>=0.4.30" pytest==9.0.3 pyflakes==3.4.0
```

Keep the name `Dune` unless you have a reason not to. `webui/run.sh` looks for
`~/miniconda3/envs/Dune/bin/python` first and only then falls back to conda base and
system `python3`, and `ProjectPaths.PRIORITY` in `webui/project_paths.py` names the same
env as the console's preferred interpreter. A differently named env means the console
may start on the wrong one; section 7 says where to change it.

`requirements.txt` pins `torch==2.11.0`, and the reference machine runs the `cu130`
build. If your driver needs a different CUDA build, install torch from the PyTorch index
first and then run the requirements file, which will leave the existing torch alone.

The second pip line covers the optional extras declared in `pyproject.toml`: `fitting`
installs `jax` for the Gaussian-fit kernels under
`pipelines/processing/param_extraction/sigma/` and the console fit lab, and `dev`
installs `pytest` and `pyflakes`. Everything else in the repository works without jax;
the tests that need it skip themselves.

Check the result:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You do not have to install the project itself. Every entry point puts the repository
root on `sys.path`, so run scripts from the repository root and imports resolve.
`pip install -e . --no-deps` works if you want the package on the path anyway.

## 2. Generate the example dataset

```bash
python main/processing/generate_dataset.py
```

Five seconds of CPU work and about 100 MB on disk. It writes `datasets/synthetic_small/`:

```
data/primary.npy        (rows, cols)                complex64
data/secondaries.npy    (n_secondaries, rows, cols) complex64
data/derived.npy        (n_secondaries, rows, cols) complex64
data/aux_full.npy       (rows, cols)                float32
data/cube_full.npy      (n_z, rows, cols)           complex64
data/dataset.json       the layout that names those files
meta/config_state.json  the crop, the z axis and the config that produced them
params/params_k2/       parameters.npy, fit_diagnostics.npz, param_extraction_meta.json
```

Everything downstream reads that directory, so this is the first thing to run on a fresh
clone. The tests are the exception: they synthesize their own copy of the scene into a
temporary directory, so they pass before you ever run the generator.

## 3. Run the tests

The suite splits in two. The backend half:

```bash
python -m pytest tests --ignore=tests/webui -q
```

3375 passed and 4 skipped in about 8 minutes on the reference machine. Three of the skips
are VRAM reservation tests that stand down when the GPU has no memory to spare, the
fourth is an architecture test whose config has no dropout field.

The console half needs `webui/` on `sys.path`, which `tests/webui/conftest.py` arranges:

```bash
python -m pytest tests/webui -q
```

614 passed in about a minute. Plain `python -m pytest -q` runs both halves, since
`testpaths` in `pyproject.toml` points at `tests`. A green run is your proof the
environment is correct.

One caveat on a shared machine: a handful of tests allocate on the GPU, and pinned-memory
batches need a working CUDA context. If another process has filled the card, those tests
fail rather than skip. Check `nvidia-smi` before believing a red run.

## 4. Point the configuration at your machine

There is no config file to edit before a run. Each entry point builds a configuration
dataclass and hands it to `ConfigCli`, which exposes every leaf field as a command-line
flag.

Ask any script what it accepts:

```bash
python main/training/train_backbone.py --help-config
```

Then override what you need. Nested fields use dots, values are coerced to the declared
type, lists and tuples are parsed with `ast.literal_eval`, and an unrecognised flag stops
the run instead of being ignored. Besides the config fields there is a fixed set of
bootstrap flags, listed as `ConfigCli.BOOTSTRAP_FLAGS` in `tools/runtime/config_cli.py`:
`--help-config`, `--detach` (alias `--nohup`), `--gpu`, `--trial`, `--worker`,
`--resume`, `--model`, `--n-trials`, `--study-name`, `--storage-url`, `--run-tag`,
`--run-dir`, `--fold`, `--split`.

To change a default for good, edit the dataclass under `configuration/`. Never put
values in the `main/` scripts.

## 5. Train the example model

The defaults already point at the dataset you generated in section 2, so a short
training run is one command:

```bash
python main/training/train_backbone.py \
  --seeds "[0]" \
  --training.epochs 5 \
  --curriculum.swap_epoch 2 \
  --gpu 0
```

The defaults are sized for the bundled scene: `training.batch_size` is 64, which fits
the 209-patch training split of `datasets/synthetic_small`, and `gpus` is `[0]`. The two
overrides exist because this is a shortened demonstration run:

- `--seeds "[0]"` trains one model. The default is `[0, 1]`, which fans one job per seed
  across the GPU pool in `--gpus` and doubles the runtime of a demo.
- `--curriculum.swap_epoch 2` keeps the loss curriculum inside the run. The warmup loss
  swaps for the complete loss at that epoch, the default is 15 (sized for the default 60
  epochs), and the trainer refuses to start unless it is below `training.epochs`.

Overrides work the same way for anything else:

```bash
python main/training/train_backbone.py \
  --seeds "[0]" \
  --backbone_name unet \
  --backbone_head conv \
  --training.epochs 5 \
  --curriculum.swap_epoch 2 \
  --gpu 0
```

Runs land under `--logdir`, default `runs/backbone`. The directory name is composed from
the configuration by `RunNaming` in `pipelines/shared/training/run_naming.py`, so a run
carries its own settings:
`resunet-set_pred-hungarian-K_2-hv-AB-param_l1_1-cosine_curve_0.05_20260805_004432`. It
holds `best_model.pt`, `last.pt`, `complete.json`, TensorBoard events, the resolved entry
config and model documentation under `docs/`, and the crop, patch, normalization and
metric metadata under `meta/`.

Resume is driven by `complete.json`. `tools/training/trainer.py` writes the marker when
training finishes, and with `--resume` (on by default) `UnitResume` reuses a run
directory that has one and deletes a run directory that does not before restarting it.
`--training.resume` is the other kind of resume: it continues an unfinished run from
`last.pt` instead of deleting it.

## 6. Run inference

```bash
python main/inference/infer_backbone.py --runs_dir runs/backbone --gpus "[0]"
```

The entry walks `--runs_dir` at any depth, finds every backbone run under it and
dispatches one inference job per run across the GPU pool. Each job writes an
`inference/<timestamp>/` directory into the run: `metrics.json`, `report.md`, the
stitched prediction cubes under `cubes/`, plus figures, animations and its own
`complete.json`. Those directories are what the console's results, leaderboard, cube,
triage and autopsy tabs read.

`--inference.device` defaults to `cuda` and is used as given, so without a GPU the job
fails with `No CUDA GPUs are available`. Pass `--inference.device cpu` on a machine
without one. Add `--inference.save_animations False` if you do not want to wait for the
GIFs, which are the slowest part of a small run.

## 7. Start the web console

```bash
webui/run.sh          # http://127.0.0.1:8772
webui/run.sh 9000     # different port
```

The HTTP layer is standard library only, but the console runs inside the project
environment: the cube explorer, fit lab, model probe and training-curve readers are
imported at boot, so numpy, matplotlib, scipy, torch, scikit-image, tensorboard and
psutil all have to be there. `run.sh` picks the `Dune` interpreter for exactly that
reason, and it prints the one it picked. To point the console at a different env, edit
the candidate list in `webui/run.sh` and `ProjectPaths.PRIORITY` in
`webui/project_paths.py`; each launch page also has an interpreter picker filled from the
conda envs the server discovers.

The console lists the 34 `main/` entry points, renders each one's configuration fields as
typed controls, shows the exact command it will run, and streams the output back live. It
also carries the run leaderboard, the results and cube browsers, the fit lab, the model
microscope and server telemetry. `webui/README.md` documents every tab.

The frontend pulls MathJax, GSAP, highlight.js and xterm.js from a CDN. Offline it still
works, with plain text where the equations would be.

## 8. Experiments and analysis

Five experiment entries under `main/experiments/` wrap the training loop:
`benchmark.py` sweeps models, heads and losses; `cross_validate.py` runs k-fold over the
scene; `sweep_patches.py` sweeps patch geometry; `tune.py` drives an Optuna study;
`tune_dataloader.py` searches batch size, worker count and prefetch depth for throughput.

Eighteen analysis entries under `main/analysis/` read finished runs rather than
producing them: trial, seed and run comparisons, weight and activation x-rays, receptive
field measurement, input attribution and stress tests, layer probes, representation
comparison, loss-landscape maps, training animations, and the report, figure and
TensorBoard exporters. Each one takes `--help-config` like everything else, and each one
has a card in the console's Scripts tab.

## Repository map

```
main/           entry points, one per verb and model type; nothing but wiring
configuration/  dataclasses holding every default
pipelines/      the pipelines behind the entry points
models/         backbones, heads, blocks
tools/          shared components (logger, training, metrics, monitoring)
webui/          the control console, stdlib HTTP server plus static frontend
tests/          about 4000 tests, self-contained
scripts/        maintenance utilities
notes/          model and design notes, read by the console
datasets/       generated example datasets, gitignored
runs/           run outputs, gitignored
logs/           entry-point logs and console state, gitignored
```

## Making it your own

`ADAPTING.md` is the guide for turning this into a real project: the dataset contract
your importer has to emit, how to change the target specification and the loss to match
it, where the normalization strategies live, the strict gates that will fail loudly if
you add a config field or an entry point without wiring it up, and the rename checklist.

## Housekeeping

`./clean.sh` removes `__pycache__` and `.pytest_cache`, which git never cleans for you.

Generated output stays out of git: `runs/`, `logs/`, `results/` and `datasets/` are all
ignored. Keep large run outputs outside the repository.

`scripts/generate_state_dict_baseline.py` regenerates `tests/state_dict_baseline.json`,
the file that guarantees old checkpoints still load. Run it only for an intentional
architecture change, and commit it on its own.
