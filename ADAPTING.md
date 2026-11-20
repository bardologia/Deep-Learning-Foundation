# Adapting the template

`README.md` gets the template running on its own example task. This file is about
replacing that task with yours: what the pipelines demand of a dataset, what to change
when your targets need another loss or other metrics, where the normalization decisions
live, which checks will stop you if you wire something up halfway, what to touch when
you rename the project, and where the console's figures and palette live.

Read it with the code open. Every claim below names the file it comes from.

## 1. Replace the data source

Nothing downstream of the dataset knows where the arrays came from. What it knows is the
dataset contract, and the one reader and writer of it is `DatasetFiles` in
`pipelines/shared/dataset/dataset_files.py`. Write an importer that produces this and the
rest of the repository works unchanged:

```
<dataset>/
  dataset.json
  inputs.npy     (input_channels, rows, cols)   float32
  targets.npy    (target_channels, rows, cols)  float32
```

`dataset.json` holds seven keys, and `DatasetFiles.read` reads every one of them, so a
missing key stops the reader:

- `name`, a free string, usually the directory name.
- `input_channels` and `target_channels`, one name per channel of the two arrays, in
  array order. The backbone is built with as many input and output channels as these
  lists hold; the names label the normalization statistics, the per-channel losses in
  TensorBoard, the inference metrics and the figure folders.
- `rows` and `cols`, the grid size both arrays share.
- `splits`, a map from `train`, `val` and `test` to a list of regions
  `[row_start, row_end, col_start, col_end]` in dataset pixels. A split may hold several
  disjoint regions; every region of a split is cut into patches on its own. The config
  factory (`pipelines/shared/config/config_factory.py`) takes the splits from here, and
  cross-validation replaces them with its fold blocks.
- `generator`, a free mapping recording how the dataset was made. The bundled generator
  writes its configuration and the run provenance; an importer can write anything.

`DatasetFiles.write` is the writer: it creates the directory, which must not exist yet,
saves the two arrays as float32 and writes `dataset.json` last. Use it from your importer,
so that a half-written dataset never has a layout file.

If your data has invalid pixels, the contract has no mask: fill them with values the loss
can take, or extend `PatchDataset` in `pipelines/backbone/dataset/datasets.py` to multiply
its weight map by a mask you store beside the arrays. The weight map is the one place a
pixel's contribution to the loss is decided.

Once your importer exists, the bundled generator can go: `main/processing/generate_dataset.py`,
`pipelines/processing/generation.py` and `configuration/generation.py`. Deleting it is not
a one-file job, because the console and the tests both know about it. Section 4 lists the
checks that will fail, and section 6 lists the registration sites in reverse. One of them
is easy to miss: `tests/conftest.py` builds the small scene every dataset fixture depends
on by calling `DatasetGenerator` into a temporary directory. Point that fixture at a small
copy of your own data, or at a generator of your own, before you delete this one, or the
suite loses its input.

## 2. Change the loss and the metrics

The backbone ends in a 1x1 convolution to as many channels as the dataset has targets, and
predicts every target channel at every pixel in normalized units. `Loss` in
`pipelines/backbone/training/loss.py` compares the prediction with the normalized targets
elementwise, with the kind chosen by `loss.kind` (`mse`, `l1` or `huber`, with
`loss.huber_delta`), averages every channel over the pixel weights of its patch and
combines the channels with `loss.channel_weights` (empty weighs every channel 1). It
returns a `WeightedMean` per channel and for the total (`tools/training/aggregation.py`),
which is what lets a whole split be pooled as one per-pixel mean.

To add a loss kind, add the value to `LossKind` in `configuration/training/general/loss.py`
and its elementwise formula to `Loss.elementwise`; the launch forms, the benchmark sweep
choices (`LaunchLayout.MULTI_LOSS_KINDS`) and the description file follow the enum. A loss
that is not elementwise, for example one on spatial gradients, belongs in `Loss.__call__`
next to the channel means, returning its own `WeightedMean` so the pooling stays exact.

Inference scores the stitched prediction in target units. `Metrics` in
`pipelines/backbone/inference/metrics.py` computes the RMSE, MAE, bias and R2 of every
target channel and their means; add a metric there and give its key an orientation
(section 4) so the comparison reports can rank it.

## 3. Normalization

Every input channel and every target channel is normalized with its own location and
scale, fitted on the train split by `StatsComputer` in
`pipelines/backbone/dataset/stats_computer.py`. `NormalizationConfig` in
`configuration/normalization/general.py` chooses the method for each side:
`normalization.inputs` and `normalization.targets`, each one of `NormMethod`:

```
zscore       mean and standard deviation
robust_iqr   median and interquartile range
min_max      0.1 and 99.9 percentiles
```

The scale is floored at 1e-8, so a constant channel does not divide by zero. The fitted
statistics land in `meta/normalization_stats.json` in the run directory, as one
`ChannelStats` per side: the method, and a name, a location and a scale per channel.
Inference reloads them from there, so a run is self-contained, and `Normalizer`
(`pipelines/backbone/dataset/normalizer.py`) applies and inverts the affine transform on
numpy arrays and torch tensors alike. A new method is a new `NormMethod` value with its
fit in `NormMethod.fit`.

## 4. The gates you have to keep green

The console and the curated documentation are not decorative: they are checked against the
code, and the checks fail loudly. Each one exists because the silent version of the failure
was worse.

### Every config field has a launch control

`LaunchLayout._validate` in
`webui/launch_layout.py` compares the fields a page claims against the leaves of its entry
config and raises `LayoutError` on any field that is unclaimed, claimed twice, or unknown,
plus any number widget without min and max bounds. Add a field to a config that a launch
page covers and the page stops building until you place it.

- Fails in: `tests/webui/test_launch_layout.py::test_training_layout_claims_every_config_field_exactly_once`,
  `::test_inference_layout_claims_every_config_field_exactly_once`, and
  `::test_every_script_layout_builds_against_its_entry_config`, which does the same for
  every key in `ProjectPaths.SCRIPT_DIRS`.
- Fix: add the field to the relevant `LaunchLayout.LAYOUTS` entry, or to a `hidden` panel
  if it should not be exposed. Hidden still counts as claimed.

### Every config field has a description, and every description has a field

`webui/config_descriptions.json` is parsed against the dataclasses under `configuration/`
by `webui/config_registry.py`, in both directions.

- Fails in: `tests/webui/test_config_registry_descriptions.py::test_every_field_has_a_description`
  for a new field with no text, `::test_no_stale_config_entries` for text left behind after
  a field is deleted or renamed, `::test_every_section_has_a_general_description` for a new
  `configuration/` subpackage, and `::test_descriptions_are_plain_ascii` for arrows and
  other symbols.
- Fix: edit `config_descriptions.json`. Section keys live under `sections`, class keys are
  `<module>::<ClassName>` under `configs`.

### Old checkpoints still load

`tests/state_dict_baseline.json` records the key set, shape signature and parameter count
of every model in every registry, at every head.

- Fails in: `tests/models_baseline/test_state_dict_baseline.py::test_state_dict_matches_baseline`
  when an architecture changes shape, and `::test_baseline_covers_full_registries` when a
  model is added or removed.
- Policy: adding a model means adding its entries. Hand-prune or hand-add the affected
  keys in the JSON when the change is additive, so the untouched models keep their
  recorded signatures as evidence. Regenerate the whole file with
  `scripts/generate_state_dict_baseline.py` only for an intentional architecture change,
  and commit that regeneration on its own, with a message that says what changed and that
  checkpoints trained before it will not load. A baseline regenerated inside a large
  commit proves nothing.

### Every path default sits below a reference root

`configuration/reference_layout.py` names the roots a checkout works from (`datasets/`,
`runs/`, `results/` and the example dataset), and the machine profile
written by `./install.sh` retargets them: `ConfigCli` rebases every `Path` leaf, and every
string leaf that equals a root or lies below one, onto the profile's root before any flag
is parsed. A default written against one machine's filesystem escapes that rebase and
points every other machine at a directory it does not have.

- Fails in: `tests/configuration/test_reference_layout.py::test_no_configuration_default_is_an_absolute_path`
  for an absolute default, and `::test_a_profile_setting_every_root_rebases_every_reference_default`
  for a path below a root that the rebase does not reach, which happens when a root is
  spelled differently from `ReferenceLayout` or the leaf is not a `Path` or a string.
- Fix: write the default relative to the repository root, below the matching
  `ReferenceLayout` root, as a `Path` or a plain string.

### Every metric key has an orientation

`MetricOrientation` in `tools/metrics/scoring.py` decides from a metric key whether a
larger value is better, worse or neither, and the comparison reports and the A/B autopsy
rank by it. A key that matches no declared rule raises `KeyError`
instead of being ranked as an error by default.

- Fails in: the `test_orientation_*` tests in `tests/loss_metrics/test_scoring.py` for
  the rules themselves.
- Fix: name the key so an existing rule covers it, or add it to `_HIGHER_OVERRIDES` or
  to the neutral, lower or higher patterns. Counts, settings and spreads belong in the
  neutral pattern, so they are reported but never ranked.

### Every figure a page names exists

The line figures of the console are named by string in `index.html` and the view modules
(the `name` attribute, `data-figure`, `HairlineFigure.html`, the `FIGURES` tables). The
names resolve against the vendored stock bundle in `webui/static/vendor/hairline-figures.js`
and `ProjectFigures.CATALOG` in `webui/static/js/hairline_project.js`, and an unknown name
throws only when its page renders.

- Fails in: `tests/webui/test_hairline_figures.py::test_every_named_figure_is_in_a_catalog`
  for a name nothing draws, `::test_every_catalog_figure_is_shown_somewhere` for a
  figure no page uses, and `::test_stock_and_project_catalogs_do_not_share_a_name`.
- Fix: register the figure in `ProjectFigures.CATALOG`, or remove it from the catalog
  together with its last use.

## 5. Add a model

Backbones are registered in `models/backbone/__init__.py`. A new one needs, in order:

1. The module under `models/backbone/`, reusing the shared pieces in `models/blocks.py`
   (`build_activation`, `build_norm2d`, `build_upsample`, `DropPath`,
   `initialize_weights`) and mixing in `OutputHeadsMixin` so it gets the four heads.
2. Its config dataclass in `configuration/architectures/backbone.py`.
3. Entries in `BACKBONE_MODEL_REGISTRY` and `BACKBONE_CONFIG_REGISTRY` under the same key,
   plus the names in `__all__`. If it needs a fixed input size, add it to
   `BACKBONE_IMAGE_SIZE_MODELS`.
4. A card in `BackboneModelLibrary._families()` (`webui/backbone_model_library.py`), in one
   of the five families, and an entry in its `CONFIG_CLASSES` and `NOTE_FILES` maps.
5. A note file under `notes/models/`, named by `NOTE_FILES`. The library reads it and the
   console renders it on the Architectures tab.
6. Baseline entries in `tests/state_dict_baseline.json`, per section 4.

`tests/webui/test_backbone_model_library.py` checks 3, 4 and 5 against each other:
`test_config_classes_cover_the_backbone_registry`, `test_note_files_cover_the_backbone_registry`,
`test_families_cover_the_backbone_registry_exactly_once` and `test_every_model_resolves_a_note`.
`tests/models_backbone/test_backbone_architectures.py` puts the model itself through every
head.

## 6. Add an entry point

`main/` holds wiring only, and the shape is fixed. `main/training/train_backbone.py` is
the short version: put the repository root on `sys.path`, import `_bootstrap`, read
`--gpu` with a throwaway parser, pin the environment with `EnvironmentPinner`, and only
then import the pipeline and run it. Heavy imports stay below the pin, because
`CUDA_VISIBLE_DEVICES` has to be set before torch initialises. `main/processing/generate_dataset.py`
is the version without a GPU: build the config with `ConfigCli(<EntryConfig>()).apply()`
and hand it to a session class. No defaults in `main/`, ever; they belong in
`configuration/`.

Registering it with the console takes five sites:

1. `ProjectPaths.SCRIPT_DIRS` in `webui/project_paths.py`, mapping the key to its
   subdirectory under `main/`. Add `ENTRY_OVERRIDES` too when the entry config cannot be
   found by parsing the script, which is the case whenever the config is not built in the
   file itself.
2. `ScriptCatalog.META` in `webui/script_catalog.py`: title, category and the purpose text
   the card shows. A per-family variant goes into `ScriptCatalog.GROUPS` instead of getting
   its own card.
3. A layout in `LaunchLayout.LAYOUTS`, claiming every field of the entry config.
4. Descriptions for any new config fields in `config_descriptions.json`.
5. A `JobDescriber.SPECS` entry in `webui/job_describer.py`, the short line that names the
   run in the console and in notifications. Each row is `(label, config path, kind)`, with
   kinds like `text`, `opt`, `opt_tail`, `list` and `flag`. Without one, the job is
   described by its overrides alone.

An entry that writes a run and can pick it up again takes two more:

6. A `resume_from` control in its layout with the `resume_runs` picker,
   `{"kind": "dataset", "mode": "resume_runs", "baseFrom": <log directory field>}`, so the
   launch page lists the runs under the log directory that recorded a configuration, and a
   `("resuming", "resume_from", "opt_tail")` row in its `SPECS` entry. The describer then
   reads the recorded configuration of the target, and the job line shows the values a
   resume runs on rather than the form's defaults.
7. A row in `FinalizeChain.TAGGED_SWEEPS` in `webui/finalize_chain.py` when the entry writes
   a run-tagged sweep of compared units, mapping its key to the sweep directory under
   `<log_base_dir>/<run_tag>`. That is what lets the launch page and the sequence board
   chain the seed comparison, the trial comparison and the report mail onto the finished
   run. A training entry needs no row: its sweep is found through `trials_mode`.

`tests/webui/test_launch_layout.py::test_every_registered_script_is_reachable_from_the_catalog`
holds 1, 2 and 3 together: every key in `SCRIPT_DIRS` has to be either a `META` card or a
`GROUPS` member, and every one of those needs a layout. Step 4 is held by the description
tests in section 4. Step 5 is the one nothing enforces: without a `SPECS` entry the job
still launches, it is just described by its raw overrides.
`tests/webui/test_job_describer.py` covers the rendering itself.

Nothing per entry is needed for saved runs, config templates or the machine profile: they work from the layout and the config
dataclass alone. A new console tab that shows images registers with the figure saver
instead, through a `figureTag()` method and `data-fig` labels, as `webui/README.md`
describes under "Saving figures".

A per-run analysis builds on `pipelines/backbone/inference/analysis/run_batch.py`.
Subclass `AnalysisRun`, set `SUMMARY_FILENAME`, and implement `_analyse(run)` returning
an `AnalysisOutcome(payload, headline, report_path, status)`. The base class loads the
run, refuses an output directory that resolves to the run itself, warns before it
replaces an earlier result, and writes `analysis_config.json`, `analysis_provenance.json`
and the completion marker. Its `RunBatch` subclass takes the entry's
`ConfigCli.profile_overrides` as its third argument, so the logged configuration table
tags the values the machine profile set; `main/analysis/probe_layers.py` is the short
version of such an entry. `StageLogger` in `tests/backbone_inference/_helpers.py`
records the timers and progress an analysis reports, for tests to assert on.

Two shared settings have one home each. The resource monitor's settings are
`ResourceConfig` in `configuration/resources.py`, outside the training package, so an
entry that does not train can carry a `resources` field without importing torch. The batch-size learning-rate rule is
`TrainingQueueConfig.lr_scale(batch_size)` in `configuration/training/general/run.py`;
`ConfigFactory` and the batch-size tuner both call it, and a new caller that changes the
batch size should too.

## 7. Rename the project

Nothing derives the name at runtime, so it is spelled out in a handful of places:

- `pyproject.toml`: `name` and `description`.
- Console strings: `webui/serve.py` (parser description), `webui/web_ui_server.py`
  (startup banner), `webui/routers/launch_routers.py` (`CatalogRouter.PROJECT`, the name,
  tagline and description the Home tab shows), `webui/run.sh` (the line it echoes),
  `webui/notifier.py` (the ntfy test notification), and `webui/static/index.html` (the
  `<title>`, the brand and logo labels, the footer).
- Report headings: `pipelines/backbone/inference/report.py` and
  `pipelines/backbone/inference/analysis/seed_comparison.py`.
  `tests/backbone_inference/test_report.py` and `tests/webui/test_notifier.py` assert the
  current text, so they change with it.
- The port, 8772: `ReferenceLayout.CONSOLE_PORT` in `configuration/reference_layout.py`
  seeds the machine profile, `webui/serve.py` holds the default, and
  `webui/tools/visual/refresh.sh` and `webui/tools/visual/snap_ui.js` hold the screenshot
  harness default.
- The interpreter and environment name: `MachineProfile.DEFAULTS["machine"]` in
  `tools/runtime/machine_profile.py` and `install.sh` both name the `Dune` env; the
  profile written by the installer carries the interpreter `webui/run.sh` starts from.
- The profile location, `~/.config/ml-template/` and `MLTEMPLATE_MACHINE_PROFILE`:
  `MachineProfile` in `tools/runtime/machine_profile.py`, `install.sh` and `webui/run.sh`.
  The variable must name an existing file or be empty; a missing file raises
  `FileNotFoundError` rather than running without a profile.
- The other `MLTEMPLATE_` variables: `MLTEMPLATE_DETACHED_OUT`, which hands a `--detach`
  child its output path (`Detacher.OUT_ENV` in `tools/runtime/detacher.py`),
  `MLTEMPLATE_INHERITED_CUDA_MASK` (`InheritedCudaMask.STASH_VAR` in
  `tools/runtime/cuda_mask.py`), and `MLTEMPLATE_ENV_NAME` and `MLTEMPLATE_CONDA_ROOT`,
  which `install.sh` reads and the profile's shell twin writes.
- The systemd scope the console's keep-alive switch moves the server into,
  `ml-template-webui-<pid>.scope`: `ServerDetacher.SCOPE_PREFIX` in
  `webui/process_manager.py`.
- The browser storage prefix `mlt.` in `webui/static/js/configs.js`
  (`mlt.config.open`, `mlt.config.active`). Changing it resets which config sections a
  returning user had open, which is the point if two consoles share a host.
- Scratch paths in the visual harness: `webui/tools/visual/refresh.sh`,
  `snap_ui.js` and `package.json`.

## 8. The console's figures and palette

Every console page carries a line figure, and the project's own figures draw its
subjects: `surface` (a smooth field with two bumps on a ridgeline surface) for training
and the dataset, `sampler` for the patch grid, `folds` for cross-validation and so on.
They are classes in `webui/static/js/hairline_project.js`, registered in
`ProjectFigures.CATALOG`; the stock
figures come from the vendored bundle in `webui/static/vendor/hairline-figures.js`.
Replace or add catalog entries for your own subject, then point the pages at them:

- `LaunchView.FIGURES` in `webui/static/js/launch.js` maps script keys to figures by
  pattern, and an unmatched key shows `surface`.
- `ResultsView.FIGURES` in `results.js` picks a figure per view mode.
- The Home board names its figures in `home.js`, and static pages name theirs in
  `index.html`.

The figure gate in section 4 catches a name nothing draws and a figure nothing shows.

The palette is the `:root` block at the top of `webui/static/css/styles.css`: `--bg`
and `--surface` for the page and its panels, the `--ink*` tokens for text, the
`--accent*` tokens, `--on-accent` for text on an accent fill, and `--frost` and
`--frost-panel` for the blurred bars and popovers. Most canvas charts and diagrams set
their own light series colours in their modules.
