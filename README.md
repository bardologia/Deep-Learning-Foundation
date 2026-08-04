# ML-Template

A dense per-pixel regression template: deep-learning estimation of Gaussian-mixture
profiles along a z axis from a stack of co-registered 2-D channels. The repository holds
the dataset synthesis, training and inference pipelines, the configuration layer that
drives them, a local web console for launching and watching runs, and about 4000 tests.

The bundled synthetic dataset generator means the whole stack runs end to end on a fresh
clone with no external data.

This file is the setup guide. Follow it top to bottom.

## What you need

A Linux machine with an NVIDIA GPU and driver for anything that trains, infers or fits
parameters. The tests, the dataset generator and the web console run fine on CPU. You
also need conda.

## 1. Environment

```bash
git clone git@github.com:bardologia/ML-Template.git
cd ML-Template

conda create -n Dune python=3.11
conda activate Dune

pip install -r requirements.txt
pip install "h5py>=3.11" "jax>=0.4.30" pytest==9.0.3 pyflakes==3.4.0
```

Keep the name `Dune` unless you have a reason not to. `webui/run.sh` looks for
`~/miniconda3/envs/Dune/bin/python` first and only then falls back to conda base and
system `python3`, so a differently named env means the console may start on the wrong
interpreter.

`requirements.txt` pins `torch==2.11.0`, and the reference machine runs the `cu130`
build. If your driver needs a different CUDA build, install torch from the PyTorch
index first and then run the requirements file, which will leave the existing torch
alone.

The second pip line covers the optional extras declared in `pyproject.toml`: `h5py` for
the processing pipeline, `jax` for the sigma-fitting kernel, `pytest` and `pyflakes`
for development.

Check the result:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You do not have to install the project itself. Every entry point puts the repository
root on `sys.path`, so run scripts from the repository root and imports resolve.

## 2. Run the tests

```bash
python -m pytest -q
```

Around 4000 tests, all self-contained. That run is your proof the environment is
correct.

## 3. Generate the example dataset

```bash
python main/processing/generate_dataset.py
```

This writes `datasets/synthetic_small/`: a primary channel, four secondary channels,
their derived channels, an auxiliary field, the reference cube, and a parameter run with
the Gaussian-mixture targets and their fit diagnostics. Everything downstream reads that
directory, so this is the first thing to run on a fresh clone.

## 4. Point the configuration at your machine

There is no config file to edit before a run. Each entry point builds a configuration
dataclass and hands it to `ConfigCli`, which exposes every leaf field as a command-line
flag.

Ask any script what it accepts:

```bash
python main/training/train_backbone.py --help-config
```

Then override what you need:

```bash
python main/training/train_backbone.py \
  --paths.dataset_path datasets/synthetic_small \
  --paths.parameters_path datasets/synthetic_small/params/params_k2/parameters.npy \
  --logdir runs/backbone \
  --training.epochs 10 \
  --gpu 0
```

Nested fields use dots, values are coerced to the declared type, and an unrecognised
flag stops the run instead of being ignored. Besides the config fields there is a fixed
set of bootstrap flags: `--help-config`, `--detach` (alias `--nohup`), `--gpu`,
`--resume`, `--trial`, `--worker`, `--model`, `--n-trials`, `--study-name`,
`--storage-url`, `--run-tag`, `--run-dir`, `--fold`, `--split`.

To change a default for good, edit the dataclass under `configuration/`. Never put
values in the `main/` scripts.

## 5. Start the web console

```bash
webui/run.sh          # http://127.0.0.1:8772
webui/run.sh 9000     # different port
```

The HTTP layer is standard library only, but the console runs inside the project
environment: the cube explorer, fit lab, model probe and training-curve readers are
imported at boot, so numpy, matplotlib, scipy, torch, scikit-image, tensorboard and
psutil all have to be there. `run.sh` picks the `Dune` interpreter for exactly that
reason. The console lists the `main/` entry points, renders each one's configuration
fields as typed controls, shows the exact command it will run, and streams the output
back live. It also carries the run leaderboard, the results and cube browsers, the fit
lab, and server telemetry. `webui/README.md` documents every tab.

The frontend pulls MathJax, GSAP, highlight.js and xterm.js from a CDN. Offline it still
works, with plain text where the equations would be.

## 6. The pipeline order

1. `main/processing/generate_dataset.py` writes the bundled example dataset described
   above. Swap it for your own writer when you bring real data: the contract is the
   `data/dataset.json` layout plus the artifacts it names.
2. `main/processing/extract_params.py` runs the GPU Gaussian fit over every profile and
   writes `parameters.npy` together with `param_extraction_meta.json` beside it.
   Downstream code reads `k_max` and the preprocessing constants from that meta file and
   raises if it is absent, so never move a parameters file away from it. The generator
   already writes one parameter run, so this step is only needed for new fit settings.
3. `main/training/train_backbone.py` trains on the (dataset, parameters) pair. The same
   shape applies to `train_dual.py`, `train_jepa.py` and the two autoencoder entries.
4. `main/inference/infer_backbone.py` and its siblings write cubes, `metrics.json` and a
   `report.md` into the run directory, which the console then browses.

## Repository map

```
main/           entry points, one per verb and model type; nothing but wiring
configuration/  dataclasses holding every default
pipelines/      the pipelines behind the entry points
models/         backbones, heads, blocks
tools/          shared components (logger, training, metrics, monitoring)
webui/          the control console, stdlib HTTP server plus static frontend
tests/          about 4000 tests
scripts/        maintenance and migration utilities
notes/          model and design notes
datasets/       generated example datasets, gitignored
```

## Housekeeping

`./clean.sh` removes `__pycache__` and `.pytest_cache`, which git never cleans for you.

Generated output stays out of git: `runs/`, `logs/`, `results/` and `datasets/` are all
ignored. Keep run outputs outside the repository.

`scripts/generate_state_dict_baseline.py` regenerates `tests/state_dict_baseline.json`.
Run it only for an intentional architecture change, and commit it on its own.
