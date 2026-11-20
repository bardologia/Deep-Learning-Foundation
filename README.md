# ML-Template

A project skeleton for supervised deep learning. It carries the configuration layer, the
dataset, training and inference pipelines, the experiment and analysis entry points, a
local web console and about 3500 tests, with a small synthetic dense regression task as
the worked example: the machinery around a model, ready for your data.

The example task is dense per-pixel regression on a 2-D grid: a few float input channels
in, a few float target channels out, every target pixel a known nonlinear and spatially
local function of the inputs around it. It comes with its own dataset generator, so that
every part of the skeleton runs end to end on a fresh clone with no external data and
each piece has a concrete thing to point at. Replacing it with your own data is the
subject of `ADAPTING.md`.

This file is the setup guide. The quick start is enough for most people; the numbered
sections explain what the installer does, for anyone who prefers to do it by hand or has
to debug a step.

## Quick start

```bash
./install.sh
```

`install.sh` finds or installs Miniconda, creates the `ml-template` environment with the torch
build that matches your NVIDIA driver, installs the pinned dependencies and the optional
extras, and then asks eight questions: where your datasets are, where runs and results
should be written, which GPUs the entry points may use, how many workers to allow, and
whether you want notifications. Every question has a detected default, so pressing
Enter through it is a valid setup. The answers become the machine profile at
`~/.config/ml-template/machine.json`; the entry points and the web console read it, so
no default in `configuration/` ever has to be edited.

Run `./install.sh` again whenever something changes; each question defaults to the
current answer. `./install.sh --yes` accepts every default without asking,
`--skip-install` reruns only the questions, `--torch-index cu130`, `cu128` or `cpu`
picks the torch build by hand instead of from the driver, and `--answers FILE` takes the
answers from a JSON file keyed by question. On a desktop session the installer also
offers an application-menu entry that starts the console and opens it in the browser.

When `nvidia-smi` is installed but fails, which usually means the driver no longer
matches the running kernel, the installer stops instead of guessing a build: fix the
driver or pass `--torch-index`. The setup wizard (`scripts/setup_machine.py`) reports
that case apart from a missing driver and suggests installing the `linux-modules-nvidia`
package for the new kernel. Its verification step imports torch in the new environment
and, when torch cannot use a GPU the driver lists, quotes torch's own reason.

Then generate the example dataset and start the console:

```bash
python main/processing/generate_dataset.py
webui/run.sh          # http://127.0.0.1:8772
```

## What you need

A Linux machine with an NVIDIA GPU and driver for training and inference at a useful
speed. The tests, the dataset generator and the web console run fine on CPU. Training
falls back to CPU on its own, with a warning, when no CUDA device is visible; inference
has to be told, with `--inference.device cpu`. You also need conda.

## 1. Environment

```bash
conda create -n ml-template python=3.11
conda activate ml-template

pip install -r requirements.txt
pip install onnx==1.22.0 pytest==9.0.3 pyflakes==3.4.0
```

Keep the name `ml-template` unless you have a reason not to. The installer records the
interpreter of the environment in the machine profile, and `webui/run.sh` starts the
console with exactly that interpreter; a hand-made environment under another name works
as long as the installer is told about it (`./install.sh --skip-install --env <name>`).

`requirements.txt` pins `torch==2.11.0`, and the reference machine runs the `cu130`
build. If your driver needs a different CUDA build, install torch from the PyTorch index
first and then run the requirements file, which will leave the existing torch alone.
Progress bars come from the project `Logger`, which draws them with rich, so tqdm is not
a dependency.

The second pip line covers the optional extras declared in `pyproject.toml`: `export`
installs `onnx` for the ONNX export entry, and `dev` installs `pytest` and `pyflakes`.
Everything else in the repository works without onnx; the tests that need it skip
themselves.

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

Under a second of CPU work and about 1.5 MB on disk. It writes `datasets/synthetic/`, a
directory in the dataset contract every pipeline reads:

```
dataset.json   name, input and target channel names, grid size, split regions, generator provenance
inputs.npy     (input_channels, rows, cols)   float32
targets.npy    (target_channels, rows, cols)  float32
docs/          resolved_entry_config.json and provenance.json of the generating run
```

The split regions are lists of `[row_start, row_end, col_start, col_end]` per split, in
dataset pixels. The generator draws `generation.input_channels` smooth random fields
(white noise smoothed by a normal kernel of `generation.smoothing_pixels`, standardized, plus
`generation.input_noise`) and computes two targets from them: `pointwise`, equal to
tanh(x0 x1) + 0.5 x2, and `neighbourhood`, the mean of x3 squared minus the mean of x0
over a square window of `generation.neighbourhood` pixels. Both carry
`generation.target_noise`. Only x0 to x3 enter the targets, so any further input channel
is a distractor. The train, validation and test splits are row bands of
`generation.val_fraction` and `generation.test_fraction` at the bottom of the grid.

Everything downstream reads that directory, so this is the first thing to run on a fresh
clone. The tests are the exception: they generate their own small copy into a temporary
directory, so they pass before you ever run the generator.

`pipelines/shared/dataset/dataset_files.py` holds the contract: `DatasetFiles.read`
reads `dataset.json` strictly, and a missing key stops the reader. The generator refuses
an output directory that already exists; delete it to generate again. It writes
`dataset.json` after the two arrays, and only once the dataset is complete does it
record the resolved entry config under `docs/`, so a failed run leaves nothing for the
console's Copy configs control to offer.

## 3. Run the tests

The suite splits in two. The backend half:

```bash
python -m pytest tests --ignore=tests/webui -q
```

5354 passed and 1 skipped in about 11 minutes on the reference machine with its GPU
visible. The skip is an architecture test whose config has no dropout field. With
`CUDA_VISIBLE_DEVICES=""` the tests that need a CUDA device skip as well, and the VRAM
reservation tests stand down when the GPU has no memory to spare.

The console half needs `webui/` on `sys.path`, which `tests/webui/conftest.py` arranges:

```bash
python -m pytest tests/webui -q
```

1319 passed in under three minutes. Plain `python -m pytest -q` runs both halves, since
`testpaths` in `pyproject.toml` points at `tests`. A green run is your proof the
environment is correct. The markers (`dataset`, `slow`) are declared in `pyproject.toml`
under `[tool.pytest.ini_options]` with `strict_markers`, so a mistyped marker fails
collection instead of quietly running in the wrong lane.

One caveat on a shared machine: a handful of tests allocate on the GPU, and pinned-memory
batches need a working CUDA context. If another process has filled the card, those tests
fail rather than skip. Check `nvidia-smi` before believing a red run.

## 4. Point the configuration at your machine

The defaults in `configuration/` describe the checkout itself: every path default is
relative to the repository root and sits below one of the roots named in
`configuration/reference_layout.py` (`datasets/`, `runs/`, `results/` and the example
dataset), the GPU lists name device 0, and the worker counts are
plain numbers. On any machine where the data or the runs should live elsewhere, the
installer's machine profile retargets them.

The profile lives at `~/.config/ml-template/machine.json`, or wherever
`MLTEMPLATE_MACHINE_PROFILE` points. An empty value disables it, which is what the test
suite does; a value naming a file that does not exist stops the entry point with
`FileNotFoundError` instead of quietly running without a profile. `ConfigCli` applies
the profile before parsing any flag: path defaults below a reference root are rebased
onto the profile's root, the most specific root winning, so a default such as
`runs/benchmark` becomes `<your runs root>/benchmark` with no per-field bookkeeping;
`gpus` and `gpu` take the profile's GPU pool; loader worker counts are
capped at its limit; the report-mail identity follows it too. A `config_overrides`
section of the form `{"ClassName.field": value}` handles anything else.
`--help-config` marks every rewritten default with `[machine profile]`, and the
console's launch pages mark the same fields.

Nothing needs a profile: without one the checkout-relative defaults apply unchanged, so
a clone works from its own `datasets/` and `runs/` directories. Each entry point builds a
configuration dataclass and hands it to `ConfigCli`, which exposes every leaf field as a
command-line flag; a flag always wins over the profile.

Ask any script what it accepts:

```bash
python main/training/train_backbone.py --help-config
```

Then override what you need. Nested fields use dots, values are coerced to the declared
type, and an unrecognised flag stops the run instead of being ignored. Every override
takes exactly one value. Lists and tuples are parsed with `ast.literal_eval` or as
comma-separated tokens, so a list is written `--seeds 0,1,2` or `--seeds "[0, 1, 2]"`;
`--seeds 0 1 2` is refused with a message naming the option. Besides the config fields
there is a fixed set of bootstrap flags, listed as `ConfigCli.BOOTSTRAP_FLAGS` in
`tools/runtime/config_cli.py`: `--help-config`, `--detach` (alias `--nohup`), `--gpu`,
`--trial`, `--worker`, `--resume`, `--model`, `--n-trials`, `--study-name`,
`--storage-url`, `--run-tag`, `--run-dir`, `--run-name`, `--fold`, `--split`.

`Logger` (`tools/monitoring/logger.py`) appends to an existing log of the same name by
default, so relaunching an entry adds to its earlier log. A run's log opens with the
host, the process id, the command line and, for a `--detach` launch, the output file,
followed by a Configuration sources table: whether a machine profile applied, every
default it rewrote and every command-line override. A run that ends on an exception
closes with an `ABORTED` banner and the traceback instead of `END`.

To change a default for good, edit the dataclass under `configuration/`. Never put
values in the `main/` scripts.

## 5. Train the example model

The defaults already point at the dataset you generated in section 2, so a short
training run is one command:

```bash
python main/training/train_backbone.py \
  --seeds "[0]" \
  --training.epochs 5 \
  --gpu 0
```

The defaults are sized for the bundled scene: `training.batch_size` is 64 with 32 x 32
patches at stride 16, which the 165-patch training split of `datasets/synthetic` fills,
and `gpus` is `[0]`. The
seed override exists because this is a shortened demonstration run:

- `--seeds "[0]"` trains one model. The default is `[0, 1]`, which fans one job per seed
  across the GPU pool in `--gpus` and doubles the runtime of a demo.

Overrides work the same way for anything else:

```bash
python main/training/train_backbone.py \
  --seeds "[0]" \
  --backbone_name unet \
  --training.epochs 5 \
  --gpu 0
```

Runs land under `--logdir`, default `runs/backbone`. The directory name is composed from
the configuration by `RunNaming` in `pipelines/shared/training/run_naming.py`, so a run
carries its own settings:
`resunet-mse-hv_20260805_004432` (model, loss kind, augmentation). It holds
`best_model.pt`, `last.pt`, `complete.json`, TensorBoard events, the resolved entry
config and model documentation under `docs/`, and the dataset configuration, patch grid,
normalization statistics and run summary (input and target channel names) under `meta/`. With `--training.use_swa True` the last quarter of
training (configurable) is additionally averaged into `swa_model.pt`, scored on
validation with refreshed batch-norm statistics.

Every run also records its provenance. `meta/` holds an environment snapshot (library
versions, CUDA stack, GPUs) and sha256 hashes of the dataset artifacts. `docs/` holds the
git state, with the diff of tracked changes whenever the working tree was dirty, and
`docs/provenance.json` beside the resolved entry config: the commit, host, Python,
conda environment, machine profile, numpy, torch and CUDA versions, and where every
non-default setting came from. `complete.json` and `meta/run_summary.json` carry the
same provenance block, and the run summary also records the configured and the effective
patch geometry, which differ when a square-only backbone squares the patch.

Resume is driven by `complete.json`. `tools/training/trainer.py` writes the marker when
training finishes, and with `--resume` (on by default) `UnitResume` reuses a run
directory that has one and deletes a run directory that does not before restarting it.
A reused run's recorded configuration and commit are compared with the launch, and every
leaf that differs is warned about. The deletion warning says what was discarded,
including the size and save time of a `last.pt` that `--training.resume` could have
continued. With `--resume false` an unfinished attempt's checkpoints, trainer state,
inference stamps and event files are deleted when training starts and its logs are
kept, while a completed unit is refused rather than retrained in place: the training run
will not start in its directory, and the benchmark and cross-validation workers exit
naming it. Relaunch under a new run tag, or delete the directory, to train it again.

`--training.resume` is the other kind of resume: it continues an unfinished run from
`last.pt` instead of deleting it. `last.pt` records the batch geometry (micro-batch,
accumulation steps, rank count and learning-rate scale), and a resume under other values,
or from a state saved before the geometry was recorded, is refused. A state that had
already stopped early goes straight to the final checkpoint, the best-weight restore and
the test evaluation.

The loss (`loss.kind`: `mse`, `l1` or `huber`) is computed on the normalized target
channels, weighted per channel by `loss.channel_weights` (empty weighs every channel 1)
and per pixel by the weight map each patch carries: padding weighs 0, a training pixel 1
for every patch covering it, and a validation or test pixel 1 over the number of patches
covering it, so every scored pixel counts once. Each loss is the weighted sum over the
weight sum, and a split pools its batches the same way, so the validation and test
losses are per-pixel means over the whole split. The inputs and the targets are each
normalized per channel with a method fitted on the train split (`normalization.inputs`,
`normalization.targets`: `zscore`, `robust_iqr` or `min_max`). Attention backbones run
torch's fused `scaled_dot_product_attention`.

### Finishing a run that broke

The staged experiments (`cross_validate`, `benchmark`, `tune`) and the training entries
take `--resume_from <path>`, any path at or inside an existing run: the configuration that
run recorded is loaded, the overrides typed on the new launch are applied on top, the GPU
fields follow the machine it runs on, and only the missing or unfinished units run.
Nothing has to be retyped and no directory has to be named the way the entry would name
it today:

```bash
python main/experiments/cross_validate.py --resume_from runs/sequences/plan/cv1
python main/experiments/benchmark.py      --resume_from runs/benchmark/arch/training
python main/training/train_backbone.py    --resume_from runs/backbone/augmentation
python main/training/train_backbone.py    --resume_from runs/backbone/resunet_myrun/seed3
```

A training path may be one run, a seed sweep root or a trial sweep root, or any run
inside them. A trial sweep is recognised from the recording sitting in its own
`batch_train_logs`, so the sweep mode and its arms need not be named again. The entry
loads the recorded configuration (`pipeline/resolved_config.json` for the staged
experiments, `batch_train_logs/resolved_config.json` for a training sweep,
`docs/resolved_entry_config.json` for one run) and prints a table of every leaf that
differs from the recording. Then it lists each unit with what it already holds: complete
units are reused, unfinished ones continue from their trainer state when they have one
and restart otherwise, missing ones are created. Units of the staged experiments are
matched to their directories through the manifest each worker writes first
(`meta/unit.json`), so a fold or seed found under any directory name keeps that
directory. A directory no planned unit accounts for is reported and left alone. `--stage prepare` on
the staged experiments prints the plan and the status of every unit without launching
any, which is how a resume is checked before it burns GPU hours; the console offers the
same choice beside `resume_from` on the benchmark, cross-validation and tuning forms, and
its `resume_from` picker lists the runs under the log directory that recorded a
configuration.

A benchmark or cross-validation run whose units partly failed still writes its reports,
then logs the unit totals and each failed unit and exits non-zero
(`Partial: k of n units failed`). An exception marks the stage it ended in, and the
pipeline, as failed in `pipeline/state.json`. Every launch appends a record to
`pipeline/launches.jsonl` (start time, stage, resume source, provenance); a relaunch by
`run_tag` whose settings differ from the recording logs the changed leaves and moves the
earlier recording to `pipeline/config_history/`.

Resumed seed and trial sweeps of the training entries re-infer every reused unit that
lacks an inference for the split, sort unfinished units into resumable (a `last.pt` is
there) and restarting, and keep the earlier launches' entries in their training and
inference results files.

Inference works the same way from the other side. Point an inference entry at any path
and it scores every run of its family found below it:

```bash
python main/inference/infer_backbone.py --runs_dir runs/sequences/plan/augmentation
```

A run is skipped only when a stamp on the requested split records the requested
`checkpoint_name`; a stamp scored with another checkpoint does not count. The backbone
entry scores a split stored as several disjoint regions region by region, one
stamp per region named `<stamp>_region<i>`, and counts the split as scored only when
every region has its stamp. The scheduler names the stamps itself, so leave
`output_subdir` unset and `run_directory` at `.` in the family block; it refuses either
when set. A cross-validation fold is scored over the row block its manifest records. To
re-score everything, pass `--skip_completed false`.

Mailing the results takes one path as well:

```bash
python main/analysis/mail_experiment_reports.py --experiments_dir runs/sequences/plan
```

Every experiment found below the path is mailed as one message: a sweep of trial runs, a
cross-validation run, a benchmark run, or a workspace holding several of them. The scan
continues below a sweep, so a runs tree that holds standalone runs beside sweeps,
cross-validation and benchmark runs mails all of them. The mailer writes the
seed-aggregate report of a seeded trial, the trial comparison, and a cross-validation
run's cross-fold report when they are missing, then attaches them together with the
configuration the experiment ran with. Each trial is mailed from its newest completed
inference on the split its comparison scores: `test` for sweeps and cross-validation
folds, the recorded `inference.split` for a benchmark run. A trial without one stops the
mailing with a message naming the splits it does have. Reused aggregates, comparisons
and cross-fold reports are named in the log, with a warning when a run completed an
inference after them, when they miss a trial or seed, or when they scored another split;
`--regenerate_reports true` rewrites them.

To read a run rather than compare it, mail the run itself:

```bash
python main/analysis/mail_experiment_reports.py --target runs --runs_dir runs/backbone --run_filter unet-mse-hv_demo
```

Each selected run travels whole, packed into zip volumes that each stay under the
attachment cap: the configuration and metadata it recorded, its logs and docs, the
TensorBoard events, and every report and figure its inference wrote. What is too heavy to
mail is named by `run_excluded_paths`, `run_excluded_suffixes` and `run_max_file_mb`,
whose defaults leave the checkpoints and the saved prediction maps behind; a warning lists the files the size cap left out, largest first. Unpacking every
volume of a run into one directory rebuilds its tree.

Every mailing logs its effective settings and writes `mail_<target>_<tag>.json` into
`staging_dir` with each source's subjects, staged paths and counts. With `--dry_run true`
everything is staged and the log says nothing was sent. `smtp_timeout_s` (300 s) bounds
the SMTP connect, login and upload, and each message is logged with its encoded size and
send time.

## 6. Run inference

```bash
python main/inference/infer_backbone.py --runs_dir runs/backbone --gpus "[0]"
```

The entry walks `--runs_dir` at any depth, finds every backbone run under it and
dispatches one inference job per run across the GPU pool. Each job writes an
`inference/<timestamp>/` directory into the run: `metrics.json`, `report.md`,
`inference_config.json` with the resolved settings and region override, the stitched
predicted and target maps under `maps/` (`predictions.npy` and `targets.npy`, in target
units), one target, prediction and absolute-error map per target channel under
`figures/<channel>/`, and its own `complete.json`, which records the scored checkpoint
(name, epoch, best epoch and best validation loss). `metrics.json` holds flat keys: `rmse_<channel>`,
`mae_<channel>`, `bias_<channel>` and `r2_<channel>` for every target channel, their means
`rmse_mean`, `mae_mean` and `r2_mean`, `n_pixels`, and the checkpoint name, its one-based
epoch and the code commit. Benchmarks, cross-validation, trial and seed comparisons and the
result tables rank on these keys; the leaderboards use the three means. R2 is NaN for a channel whose targets are constant
over the region. Those directories are what the console's results tab reads.

The predictor runs every patch of the split region through the model, inverts the target
normalization and overlap-adds the patches with a tapering window
(`--inference.stitch_window`: `hann`, `uniform` or `triangular`).
`--inference.device` defaults to `cuda` and is used as given, so without a GPU the job
fails with `No CUDA GPUs are available`. Pass `--inference.device cpu` on a machine
without one. `--inference.target.enabled True --inference.target.dataset_path <dir>`
scores the run on the test split of another dataset with the same channels and writes
the stamp under `<run>/cross_dataset/<dataset>/`.

## 7. Start the web console

```bash
webui/run.sh          # http://127.0.0.1:8772
webui/run.sh 9000     # different port
```

`run.sh` reads the interpreter, bind address and port from the machine profile's shell
twin (`~/.config/ml-template/machine.env`) and refuses to start without one, pointing at
`./install.sh`. The HTTP layer is standard library only, but the console runs inside the
project environment: the model probe is imported at boot, so numpy, matplotlib, scipy, torch, tensorboard and
psutil all have to be there, which is why the profile records the project environment's
interpreter. Each launch page also has an interpreter picker filled from the conda envs
the server discovers, with the profile's environment preselected.

The console lists the 24 `main/` entry points, renders each one's configuration fields as
typed controls with the profile's values already in place, shows the exact command it
will run, and streams the output back live. A launch page stores its configuration as a
saved run to launch later, or as a named config template under `logs/config_templates/`
whose overrides import into the form of any entry point from the Templates tab or the
launch rail. It also carries the results browser, the model microscope and server
telemetry; the browse roots of the results tabs
start at the profile's data and runs roots. `webui/README.md` documents every tab.

The console uses one dark theme. Every wait on the backend shows a short status line, and
every page carries a line figure that follows the view; on the Home board several of
them draw live state (the watchdog, the GPU week schedule, the job queue, the GPU
guard). The Home board is a set of cards for the resource watchdog, notifications, the
GPU week, the CUDA devices, memory and storage, GPU activity, neighbour impact, host
history, jobs, the GPU guard, the server log and a 30-day GPU heatmap. Its watchdog card
holds the kill front end button, which stops every console server on the same project
root; running and detached jobs survive it and are adopted again on the next start.

On the launch pages the dataset field is a picker over the dataset folders that hold a
`dataset.json`, badged with their input and target channel counts, and the
cross-validation form shows the architecture fields of the selected backbone. An ended job's console tile has a Retry
button that reopens the launch page with the job's overrides, interpreter and finalize
switches. A dropped job stream reconnects and replays from its last event. TensorBoard
runs are picked from a folder tree.
Finalize steps chain onto blocks of plain sequences as well as isolated ones, and a
tuning block records its best configuration as its verdict while the sequence board
shows its study progress.

The frontend pulls MathJax, GSAP, highlight.js and xterm.js from a CDN. Offline it still
works, with plain text where the architecture notes' equations would be.

## 8. Experiments and analysis

Three experiment entries under `main/experiments/` wrap the training loop:
`benchmark.py` sweeps models and loss kinds; `cross_validate.py` runs k-fold over
the dataset rows; `tune.py` drives an Optuna study. Patch-shape sweeps run through the
training entries themselves as `trials_mode=patch`, alongside the other fan-out modes
(one run per listed architecture, augmentation, stability, EMA and AMP ladders). The
patch sweep report follows the planned seeds and names the seeds a trial mean drops.

`benchmark.py` lists the selected and skipped models at the start, and checks its comparison
against the planned units, warning about each missing or unplanned run. A resumed
benchmark reuses a completed inference of the scored split. Comparison summaries record
per model the seeds each metric mean used (`n_used`) and the seeds without metrics
(`missing_seeds`), and the log warns when a mean covers fewer seeds than were trained.
`cv_summary.json` records `fold_status`, the seeds each fold used, the missing seeds and
the provenance; the cross-fold report shows Status and Seeds columns and names any
metric that is non-finite in every fold. The cross-validation collector logs, per
inference split, which fold runs hold a completed inference.

`tune.py` samples three groups of settings, each with its own switch:
`tuning.search_lr` (learning rates, weight decays and dropout), `tuning.search_arch`
(architecture fields) and `tuning.search_batch` (a batch size from `tuning.batch_sizes`,
with the optimizer rescaled by the training batch rule). `plateau_patience` and `plateau_min_delta` stop a study whose best
value has stalled, and `time_budget_h` stops workers from starting new trials; these may
change on resume, the search groups may not. The workers of one model share its study
in the run's SQLite database (`optuna.db`), whose heartbeat lets a dead worker's trials
be named and failed when the study is next opened; `n_workers` may exceed the GPU count,
and each worker is seeded and named by its index (`<model>/workerNN/`). Per model the
run keeps `best_config.json` and `study_progress.json` (trial counts, best value, stall
count, rate and time to the target). A trial's checkpoints are deleted once its
objective is read unless `keep_trial_checkpoints` is on. `tuning_results.json` holds
each model's status (`DONE`, `PARTIAL` or `FAILED`) and stop reason, and the entry exits
non-zero when a reported model is not `DONE`. The `prepare` and `full` stages refuse an
existing study without `--resume` or `resume_from`.

Eighteen analysis entries under `main/analysis/` read finished runs rather than
producing them: trial, seed and run comparisons (the seed comparison also reports 95%
confidence intervals and scores the across-seed mean prediction as a deep ensemble),
weight and activation x-rays, receptive field measurement, input attribution and
stress tests, layer probes, loss-landscape maps, training
animations, a torch.profiler capture of full training steps, and the report, figure,
TensorBoard, result-table (CSV/LaTeX/markdown) and ONNX exporters. Each one takes
`--help-config` like everything else, and each one has a card in the console's
Scripts tab.

Every per-run analysis pass writes `analysis_config.json`, `analysis_provenance.json`
and a completion marker into its output subdirectory and warns before it replaces an
earlier result; a batch closes with a summary table of its runs. The seed comparison,
the paper figure pack and the result-table export read only completed inference stamps
of their `split` (default `test`), and `compare_runs.py` takes `--inference_split`. Trial comparisons
write `comparison_summary.json` and state their split, seed means show the seeds they
cover as used/total, and leaderboards name the smallest p-value the seed count allows.
The weight x-ray records its checkpoint entry, thresholds and provenance and reads
transposed convolutions on their output axis; the activation x-ray records the leaf
modules that are reused or never fire.

## Repository map

```
install.sh      one-command setup: conda, environment, dependencies, guided machine profile
main/           entry points, one per verb and model type; nothing but wiring
configuration/  dataclasses holding every default; reference_layout.py names the roots the profile rebases
pipelines/      the pipelines behind the entry points
models/         backbones and their shared blocks
tools/          shared components (logger, training, metrics, monitoring)
webui/          the control console, stdlib HTTP server plus static frontend
tests/          about 3500 tests, self-contained
scripts/        maintenance utilities and the setup wizard
notes/          model and design notes, read by the console
datasets/       generated example datasets, gitignored
runs/           run outputs, gitignored
logs/           entry-point logs and console state, gitignored
```

## Making it your own

`ADAPTING.md` is the guide for turning this into a real project: the dataset contract
your importer has to emit, how to change the loss and the metrics to match your targets,
where the normalization methods live, the strict gates that will fail loudly if
you add a config field or an entry point without wiring it up, the rename checklist, and
where the console's figures and palette live.

## Artifacts from older checkouts

Formats change without migration scripts. Datasets, runs and inference stamps written by
an older checkout are regenerated, not converted: the readers are strict and stop on the
first missing or renamed field.

## Housekeeping

`./clean.sh` removes `__pycache__` and `.pytest_cache`, which git never cleans for you.

Generated output stays out of git: `runs/`, `logs/`, `results/` and `datasets/` are all
ignored. Keep large run outputs outside the repository.

`scripts/generate_state_dict_baseline.py` regenerates `tests/state_dict_baseline.json`,
the file that guarantees old checkpoints still load. Run it only for an intentional
architecture change, and commit it on its own.
