# ML-Template

A project skeleton for supervised deep learning, lifted out of a working research
codebase. It carries the configuration layer, the dataset, training and inference
pipelines, the experiment and analysis entry points, a local web console and about 6700
tests. The domain code is gone; what is left is the machinery around a model.

The example task is per-pixel Gaussian-mixture estimation along a z axis from a stack of
co-registered 2-D channels, and it comes with its own dataset generator. It is there so
that every part of the skeleton runs end to end on a fresh clone with no external data,
and so that each piece has a concrete thing to point at. Replacing it is the subject of
`ADAPTING.md`.

This file is the setup guide. The quick start is enough for most people; the numbered
sections explain what the installer does, for anyone who prefers to do it by hand or has
to debug a step.

## Quick start

```bash
./install.sh
```

`install.sh` finds or installs Miniconda, creates the `Dune` environment with the torch
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
has to be told, with `--inference.device cpu`. Fitting parameter runs with
`extract_params` needs a GPU that JAX can see. You also need conda.

## 1. Environment

```bash
conda create -n Dune python=3.11
conda activate Dune

pip install -r requirements.txt
pip install "jax>=0.4.30" pytest==9.0.3 pyflakes==3.4.0
```

Keep the name `Dune` unless you have a reason not to. The installer records the
interpreter of the environment in the machine profile, and `webui/run.sh` starts the
console with exactly that interpreter; a hand-made environment under another name works
as long as the installer is told about it (`./install.sh --skip-install --env <name>`).

`requirements.txt` pins `torch==2.11.0`, and the reference machine runs the `cu130`
build. If your driver needs a different CUDA build, install torch from the PyTorch index
first and then run the requirements file, which will leave the existing torch alone.
Progress bars come from the project `Logger`, which draws them with rich, so tqdm is not
a dependency.

The second pip line covers the optional extras declared in `pyproject.toml`: `fitting`
installs `jax` for the Gaussian-fit kernels under
`pipelines/processing/param_extraction/sigma/` and the console fit lab, `export`
installs `onnx` for the ONNX export entry, and `dev` installs `pytest` and
`pyflakes`. Everything else in the repository works without jax or onnx; the tests
that need them skip themselves.

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
docs/                   resolved_entry_config.json and provenance.json of the generating run
```

Everything downstream reads that directory, so this is the first thing to run on a fresh
clone. The tests are the exception: they synthesize their own copy of the scene into a
temporary directory, so they pass before you ever run the generator.

The generator refuses an output directory that already holds a finished dataset (a
`data/dataset.json`, a `docs/resolved_entry_config.json` or any parameter run under
`params/`); delete the directory to generate it again. It writes `dataset.json` last and
records the generating commit in `dataset.json`, `meta/config_state.json` and the
parameter run's metadata. Only once the dataset is complete does it record the resolved
entry config under `docs/`, so a failed run leaves nothing for the console's Copy
configs control to offer, and a finished dataset can be copied from on the
`generate_dataset` launch page.

The parameter run also fixes the activity threshold, the amplitude above which a
Gaussian slot counts as active. It is chosen once, by `SynthesisConfig.activity_threshold`
when the generator writes the labels or by `fit_activity_threshold` when `extract_params`
fits them (1e-3 by default in both), and recorded in `param_extraction_meta.json`.
Training (loss masks, slot matching, slot diagnostics, output normalization statistics),
inference, the profile autoencoder, the seed comparison and the console all read it from
there, and none of them has a threshold setting of its own. A slot is active only when
its amplitude is strictly above the threshold.

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
`configuration/reference_layout.py` (`datasets/`, `runs/`, `results/`, the example
dataset and its parameter run), the GPU lists name device 0, and the worker counts are
plain numbers. On any machine where the data or the runs should live elsewhere, the
installer's machine profile retargets them.

The profile lives at `~/.config/ml-template/machine.json`, or wherever
`MLTEMPLATE_MACHINE_PROFILE` points. An empty value disables it, which is what the test
suite does; a value naming a file that does not exist stops the entry point with
`FileNotFoundError` instead of quietly running without a profile. `ConfigCli` applies
the profile before parsing any flag: path defaults below a reference root are rebased
onto the profile's root, the most specific root winning, so a default such as
`runs/benchmark` becomes `<your runs root>/benchmark` with no per-field bookkeeping;
`gpus`, `gpu_device_ids` and `gpu` take the profile's GPU pool; `vram_budget_gb`,
`cpu_workers` and `gif_workers` take the profile's values; loader worker counts are
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
metric metadata under `meta/`. With `--training.use_swa True` the last quarter of
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

The pre-run probes `pretrain.find_batch_size`, `pretrain.tune_loader` and
`pretrain.find_lr` each switch their own flag off once they have run, so the resolved
entry config records the measured batch size, loader settings and learning-rate
multiplier instead of the request to measure them. They are refused on a run that
resumes from `last.pt`. The evidence stays in the run under `pretrain/`:
`lr_range_test.json` with the settings, every rung and the selection statistics, and
`loader_sweep.csv`. The preflight warns when the GPU is already occupied (a CUDA context
above 1.5 GB) or when the dataset size lowers the batch ceiling.

The loss refuses a curriculum stage with no enabled term or with weights summing to zero,
warns about enabled terms of weight zero, and refuses a z axis with fewer than two
samples. The normalization fit refuses a non-finite location or scale. `prior_bias_init`
(on `train_backbone` and `cross_validate`, off by default) starts the conv head's biases
at the training-label prior in normalized output space and writes
`meta/head_prior_bias.json`; it needs `--backbone_head conv`. Attention backbones run
torch's fused `scaled_dot_product_attention`; only the attention-capture analysis
switches to the explicit path that exposes the weights.

The autoencoder and JEPA entries follow the same resume and provenance rules. A JEPA run
couples only finished autoencoder runs (a `complete.json` in the run directory), records
each imported checkpoint and its dataset provenance in `meta/coupled_autoencoders.json`,
and refuses an image autoencoder trained on another input selection or other secondary
labels; a different preprocessing run, parameter run or input statistics only warn. The
JEPA curriculum swap resets the best-checkpoint baseline, so `best_model.pt` comes from
the main loss stage. The profile autoencoder takes `pixel_subsample` in (0, 1] and
`keep_empty_frac` in [0, 1], and refuses a training split whose sampled curves are
near-constant.

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
directory. A run that trained before manifests existed carries none, and is adopted by
`scripts/backfill_unit_manifests.py <run>`, which writes each unit's manifest from the
run's own recording and matches units to directories by the name that recording composes.
A directory no planned unit accounts for is reported and left alone. `--stage prepare` on
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
inference results files. A trial sweep or multi-seed fan-out with the loss-scale probe
set to exit after its report (`probe.enabled` with `probe.exit_after`) is refused,
since every unit would stop with nothing trained.

Inference works the same way from the other side. Point an inference entry at any path
and it scores every run of its family found below it:

```bash
python main/inference/infer_backbone.py --runs_dir runs/sequences/plan/augmentation
```

A run is skipped only when a stamp on the requested split records the requested
`checkpoint_name`; a stamp scored with another checkpoint does not count. The backbone
and dual entries score a split stored as several disjoint regions region by region, one
stamp per region named `<stamp>_region<i>`, and count the split as scored only when
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
python main/analysis/mail_experiment_reports.py --target runs --runs_dir runs/backbone --run_filter "['unet-conv-sorted_gt-K_2']"
```

Each selected run travels whole, packed into zip volumes that each stay under the
attachment cap: the configuration and metadata it recorded, its logs and docs, the
TensorBoard events, and every report and figure its inference wrote. What is too heavy to
mail is named by `run_excluded_paths`, `run_excluded_suffixes` and `run_max_file_mb`,
whose defaults leave the checkpoints, the reconstructed cubes and the walk animations
behind; a warning lists the files the size cap left out, largest first. Unpacking every
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
prediction cubes under `cubes/`, plus figures, animations and its own `complete.json`,
which records the scored checkpoint (name, epoch, best epoch and best validation loss).
`metrics.json` carries the checkpoint name, its one-based epoch, the code commit and the
activity threshold it scored with. Those directories are what the console's results,
leaderboard, cube, triage and autopsy tabs read.

`--inference.device` defaults to `cuda` and is used as given, so without a GPU the job
fails with `No CUDA GPUs are available`. Pass `--inference.device cpu` on a machine
without one. Add `--inference.save_animations False` if you do not want to wait for the
GIFs, which are the slowest part of a small run. `--inference.gif_duration` sets the
total playback length of each GIF and overrides `gif_fps`, and
`--inference.gif_separate_panels True` also writes each cube as its own single-panel GIF;
the benchmark's inference block has both fields too.

The loader refuses a checkpoint whose z axis differs from the dataset's and logs the
checkpoint epoch, both axes, the region and where the region came from. The metrics count
what they leave out: pixel R² is NaN where the reference profile is constant and those
pixels are counted, pixels with an empty ground truth are counted and left out of the
cosine and peak errors (the `*_nonempty` statistics), the SSIM means record how many
slices they scored, and the normalization audit and the stratification tables record
non-finite inputs and dropped bins.

`scripts/export_slice_ssim.py <metrics.json>...` saves the per-slice SSIM of a stamp as
`ssim_gt_col.npy` and `ssim_gt_row.npy` beside each file, in slice order with
non-finite slices kept as NaN.

## 7. Start the web console

```bash
webui/run.sh          # http://127.0.0.1:8772
webui/run.sh 9000     # different port
```

`run.sh` reads the interpreter, bind address and port from the machine profile's shell
twin (`~/.config/ml-template/machine.env`) and refuses to start without one, pointing at
`./install.sh`. The HTTP layer is standard library only, but the console runs inside the
project environment: the cube explorer, fit lab, model probe and training-curve readers
are imported at boot, so numpy, matplotlib, scipy, torch, scikit-image, tensorboard and
psutil all have to be there, which is why the profile records the project environment's
interpreter. Each launch page also has an interpreter picker filled from the conda envs
the server discovers, with the profile's environment preselected.

The console lists the 37 `main/` entry points, renders each one's configuration fields as
typed controls with the profile's values already in place, shows the exact command it
will run, and streams the output back live. A launch page stores its configuration as a
saved run to launch later, or as a named config template under `logs/config_templates/`
whose overrides import into the form of any entry point from the Templates tab or the
launch rail. It also carries the run leaderboard, the results and cube browsers, the fit
lab, the model microscope and server telemetry; the browse roots of the results tabs
start at the profile's data and runs roots. `webui/README.md` documents every tab.

The console uses one dark theme. Every wait on the backend shows an animated orb, and
every page carries a line figure that follows the view; on the Home board several of
them draw live state (the watchdog, the GPU week schedule, the job queue, the GPU
guard). The Home board is a set of cards for the resource watchdog, notifications, the
GPU week, the CUDA devices, memory and storage, GPU activity, neighbour impact, host
history, jobs, the GPU guard, the server log and a 30-day GPU heatmap. Its watchdog card
holds the kill front end button, which stops every console server on the same project
root; running and detached jobs survive it and are adopted again on the next start.

On the launch pages the secondary-channel field is a picker filled from the selected
dataset's channel labels, and the cross-validation and JEPA forms show the architecture
fields of the model their training type selects. An ended job's console tile has a Retry
button that reopens the launch page with the job's overrides, interpreter and finalize
switches. A dropped job stream reconnects and replays from its last event. TensorBoard
runs are picked from a folder tree, and a data-parallel job opens rank zero's run.
Finalize steps chain onto blocks of plain sequences as well as isolated ones, and a
tuning block records its best configuration as its verdict while the sequence board
shows its study progress. The cube explorer reads cubes whole through a shared read
pool, shows the load in bytes and keeps the last eight cubes in memory; the slice
collector puts cuts in a basket that is saved as one folder and can be mailed
(`REPORT_MAIL_PASSWORD`).

The frontend pulls MathJax, GSAP, highlight.js and xterm.js from a CDN. Offline it still
works, with plain text where the equations would be.

## 8. Experiments and analysis

Four experiment entries under `main/experiments/` wrap the training loop:
`benchmark.py` sweeps models, heads and losses; `cross_validate.py` runs k-fold over the
scene; `tune.py` drives an Optuna study; `tune_dataloader.py` searches batch size,
worker count and prefetch depth for throughput. Patch-shape sweeps run through the
training entries themselves as `trials_mode=patch`, alongside the other fan-out modes
(curriculum, warmup, presence, pair, log-energy, secondary, input, context, reach, head,
augmentation, clamp, stability, EMA, AMP and ablation ladders). The patch sweep report
follows the planned seeds and names the seeds a trial mean drops.

`benchmark.py` refuses a `skip_models` entry that matches no model key and no backbone
name, lists the selected and skipped models at the start, and checks its comparison
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
with the optimizer rescaled by the training batch rule). JEPA studies search the
architecture only. `plateau_patience` and `plateau_min_delta` stop a study whose best
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

`tune_dataloader.py` writes each sweep into its own
`<output_dir>/<mode>/<model>_<timestamp>/` with its logs, figures, `resolved_config.json`
and a `results.json` that carries the status, the settings, the provenance and the GPU it
measured, found by its NVML UUID. `<mode>/work/` is scratch space the sweeps share.

`main/processing/extract_params.py` fits parameter runs onto a dataset. It refuses a
permutation whose `params/<run>` already exists unless `overwrite` is on, which deletes
and refits it; refuses repeated sweep values, a dataset queued twice and modes that
collide on one run name; refuses to fit when JAX sees no GPU; and raises when no pixel
clears the activity threshold. Its `resources` block samples RAM, VRAM and shared memory
around every group. Each parameter run records the commit, the fit outcome (active
pixels, truncation, backend, devices, duration), where its z range came from and the
share of slots that ended on the sigma and mean clamp bounds, and keeps its own copy of
the group log under `logs/`. `analyze_param_extraction` writes `fit_metrics_summary.json`
after every figure, so a run counts as analysed only once the analysis finished.

Twenty-two analysis entries under `main/analysis/` read finished runs rather than
producing them: trial, seed and run comparisons (the seed comparison also reports 95%
confidence intervals and scores the across-seed mean prediction as a deep ensemble),
weight and activation x-rays, receptive field measurement, input attribution and
stress tests, layer probes, representation comparison, loss-landscape maps, training
animations, a torch.profiler capture of full training steps, and the report, figure,
TensorBoard, result-table (CSV/LaTeX/markdown) and ONNX exporters. Each one takes
`--help-config` like everything else, and each one has a card in the console's
Scripts tab.

Every per-run analysis pass writes `analysis_config.json`, `analysis_provenance.json`
and a completion marker into its output subdirectory and warns before it replaces an
earlier result; a batch closes with a summary table of its runs. The seed comparison,
the paper figure pack and the result-table export read only completed inference stamps
of their `split` (default `test`), and `compare_runs.py` takes `--inference_split`. The
seed comparison refuses seeds scored at different activity thresholds. Trial comparisons
write `comparison_summary.json` and state their split, seed means show the seeds they
cover as used/total, and leaderboards name the smallest p-value the seed count allows.
The weight x-ray records its checkpoint entry, thresholds and provenance and reads
transposed convolutions on their output axis; the activation x-ray records the leaf
modules that are reused or never fire. `collect_reports.py` also gathers the reports of
profile and image autoencoder runs, which stamp under `inference/<family>/`.

## Repository map

```
install.sh      one-command setup: conda, environment, dependencies, guided machine profile
main/           entry points, one per verb and model type; nothing but wiring
configuration/  dataclasses holding every default; reference_layout.py names the roots the profile rebases
pipelines/      the pipelines behind the entry points
models/         backbones, heads, blocks
tools/          shared components (logger, training, metrics, monitoring)
webui/          the control console, stdlib HTTP server plus static frontend
tests/          about 6700 tests, self-contained
scripts/        maintenance utilities and the setup wizard
notes/          model and design notes, read by the console
datasets/       generated example datasets, gitignored
runs/           run outputs, gitignored
logs/           entry-point logs and console state, gitignored
```

## Making it your own

`ADAPTING.md` is the guide for turning this into a real project: the dataset contract
your importer has to emit, how to change the target specification and the loss to match
it, where the normalization strategies live, the strict gates that will fail loudly if
you add a config field or an entry point without wiring it up, the rename checklist, and
where the console's figures and palette live.

## Artifacts from older checkouts

Formats change without migration scripts; artifacts written by an older checkout are
regenerated, not converted. These changes, made in October 2026, affect existing files:

- Parameter runs need `activity_threshold` in `param_extraction_meta.json`. Generate or
  extract them again.
- Trainer states saved before `last.pt` recorded the batch geometry cannot be resumed,
  and profile autoencoder runs whose `profile_dataset_config.json` lacks
  `activity_threshold` cannot be loaded. Retrain them.
- Inference stamps scored before `metrics.json` recorded the activity threshold are
  refused by the seed comparison and by the cube explorer's parameter overlays, and
  stamps without a completion marker are ignored by the split-aware readers
  (`scripts/backfill_completion_markers.py` writes markers for older stamps). Metric
  values moved too: slot activity is strict, scoring uses the label run's threshold
  where it used a fixed 1e-4, and pixel R² is NaN at constant references. Run inference
  again before comparing old and new runs. Attention backbones now run on the fused
  attention kernel, so their GPU results differ from older runs at noise level.
- Saved launch configs, config templates and sequence plans that set `amp_zero_thr`,
  `active_amp_thr` or `col_chunk` are rejected by `ConfigCli`, and so is a command line
  that writes a list as space-separated values.
- Tuning runs from an older checkout cannot be resumed, because their recorded
  configuration lacks the new study keys, and an older `tuning_results.json` is refused.
  Start a new run tag.
- The mailer's reused seed aggregates need `seed_runs` and `seed_inference` records in
  their `metrics.json`; pass `--regenerate_reports true` to rewrite older ones.

## Housekeeping

`./clean.sh` removes `__pycache__` and `.pytest_cache`, which git never cleans for you.

Generated output stays out of git: `runs/`, `logs/`, `results/` and `datasets/` are all
ignored. Keep large run outputs outside the repository.

`scripts/generate_state_dict_baseline.py` regenerates `tests/state_dict_baseline.json`,
the file that guarantees old checkpoints still load. Run it only for an intentional
architecture change, and commit it on its own.

`scripts/rename_runs.py <roots>` plans the move of run directories from older naming
generations to the current run tag scheme, and performs it with `--apply`. It ends each
root and the whole run with an outcome totals table, and an abort logs how many groups
were already applied.
