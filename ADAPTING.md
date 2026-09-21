# Adapting the template

`README.md` gets the template running on its own example task. This file is about
replacing that task with yours: what the pipelines demand of a dataset, what has to
change when the target is not a Gaussian mixture, where the normalization decisions live,
which checks will stop you if you wire something up halfway, and what to touch when you
rename the project.

Read it with the code open. Every claim below names the file it comes from.

## 1. Replace the data source

Nothing downstream of the dataset knows where the arrays came from. What it knows is the
on-disk layout, and the reader for it is `Layout` in
`pipelines/shared/dataset/dataset_spatial.py`. Write an importer that produces this and
the rest of the repository works unchanged:

```
<dataset>/
  data/
    dataset.json
    primary.npy        (rows, cols)                 complex64
    secondaries.npy    (n_secondaries, rows, cols)  complex64
    derived.npy        (n_secondaries, rows, cols)  complex64
    aux_full.npy       (rows, cols)                 float32
    cube_full.npy      (n_z, rows, cols)            complex64
  meta/
    config_state.json
  params/<run name>/
    parameters.npy               (3 * k_max, rows, cols)  float32
    fit_diagnostics.npz
    param_extraction_meta.json
```

`data/dataset.json` is the entry point. Four of its keys are read:

- `global_crop`, four integers `[row_start, row_end, col_start, col_end]`. It is the
  absolute position of this scene in whatever larger frame it was cut from, and
  everything that names rows in absolute coordinates is measured against it:
  `ConfigFactory.split_regions` in `pipelines/shared/config/config_factory.py` intersects
  it with `training.train_rows` and its siblings, and the cube explorer subtracts it to
  place an inference region back inside the full cube
  (`webui/cube_explorer.py::_crop_bounds`). If your data has no such frame, write
  `[0, rows, 0, cols]` and keep the split rows inside it.
- `artifacts`, a map from artifact key to filename. The keys are fixed:
  `primary`, `secondaries`, `derived`, `aux_full`, `cube_full`. `Layout.artifact_path`
  resolves them, and the loaders in `pipelines/backbone/dataset/spatial.py` memory-map the
  results, so shapes have to agree across files.
- `channel_labels`, the primary label first and one per secondary after it. Selecting
  secondaries by label (`paths.secondary_labels`) resolves through this list, and
  `Layout.secondary_indices` raises if it is empty.
- `cube_tag`, a free string, logged with the layout.

`dataset_type`, `parameter_tag` and `max_amplitude_clip` are written by the bundled
generator and read by nothing. They are provenance. Keep or drop them.

`meta/config_state.json` carries the z axis: `cube_config.z_range` (two floats) and
`cube_config.n_z`. `GaussianConfig.from_dataset` in `configuration/dataset/gaussian.py`
reads the range, and `ExtractParamsEntryConfig` in `configuration/param_extraction.py`
falls back to it when no range is given on the command line. The same file also holds
`crop`, the same bounds as `global_crop` in record form.

The parameter run is the supervision. `parameters.npy` is `(3 * k_max, rows, cols)`,
interleaved as amplitude, mean, width per slot, sorted by mean, with unused slots zeroed.
`param_extraction_meta.json` sitting beside it is not optional: `GaussianConfig` reads
`k_max` from it and raises `FileNotFoundError` if it is missing, which is what stops a
parameter file from being separated from its own description. A dataset can hold several
parameter runs side by side under `params/`; `paths.parameters_path` picks one.

The channels themselves are complex because the example task treats magnitude and phase
as separate input channels. `InputConfig` in `configuration/dataset/general/dataset.py`
turns each source into channels through `Representation`
(`tools/data/representation.py`): magnitude only, angle only, real and imaginary, and so
on. If your data is real, store it as complex with a zero imaginary part or narrow
`Representation` to the real cases and adjust `InputConfig.channel_group_keys`, which is
what names the normalization slot of every produced channel.

Once your importer exists, the bundled generator can go: `main/processing/generate_dataset.py`,
`pipelines/processing/synthesis/`, `configuration/synthesis.py`. Deleting it is not a
one-file job, because the console and the tests both know about it. Section 4 lists the
checks that will fail, and section 6 lists the registration sites in reverse. One of them
is easy to miss: `tests/conftest.py` builds the scene every fixture depends on by calling
`SyntheticDatasetPipeline` into a temporary directory. Point that fixture at your own data
or at a small generator of your own before you delete this one, or the suite loses its
input.

## 2. Change the target specification

The example target is `k_max` Gaussians per pixel, three numbers each, evaluated on a z
axis of `n_z` samples. Three things define it: the axis and slot count, the roles that are
supervised, and the head that emits them.

`GaussianConfig` (`configuration/dataset/gaussian.py`)
holds `n_default_gaussians`, `x_min`, `x_max` and `params_per_gaussian`. It is built by
`ConfigFactory.gaussian_config()` from the dataset, never by hand, so the slot count
follows the parameter run and the axis follows the dataset metadata. Widening the axis or
changing K is a matter of writing a different parameter run, not of editing code.

Which of the three roles are supervised is `OutputConfig` in
`configuration/dataset/general/dataset.py`, through `use_amplitude`, `use_mu` and `use_sigma`,
and `selected_indices` turns them into column indices into `parameters.npy`. The dataset
(`pipelines/backbone/dataset/datasets.py`) and the statistics computer
(`pipelines/backbone/dataset/stats_computer.py`) are the only readers. Note the asymmetry
before you use it: `GaussianConfig.params_per_gaussian` stays at 3, and the loss indexes
each slot positionally as amplitude, mean, width (`_param_term` and `_physical_errors` in
`pipelines/backbone/training/loss.py`). Dropping a role from `OutputConfig` therefore also
means setting `params_per_gaussian` to match and rewriting those two methods.

The head that emits the parameters is chosen with `--backbone_head`, and
`models/blocks.py::OutputHeadsMixin` builds all four:

- `conv`, a single 1x1 convolution to `out_channels`. It imposes no slot structure and is
  the only head that does not require `out_channels` to divide by `params_per_gaussian`.
- `multihead`, three pixel MLPs, one each for amplitude, mean and width, interleaved back
  into slot order. It is written for three parameters per slot.
- `per_gaussian`, one pixel MLP per slot emitting that slot's parameters.
- `set_pred`, `per_gaussian` plus a sigmoid existence gate per slot that blends the
  slot's amplitude toward a learned off value, so a slot can be switched off rather than
  regressed to zero. This is the default, and the dual model
  (`models/dual/dual_resunet.py`) accepts nothing else.

Slot order is a convention, not a fact, so the loss aligns the two sides
before comparing them. `ParamMatcher.match` in `tools/loss/param_loss.py` runs on every
backward pass of the backbone loss, under `curriculum.<stage>.param_matching`:
`sorted_gt` sorts the ground-truth slots by mean and leaves the prediction alone;
`hungarian` sorts the ground truth and then assigns predicted slots to it by minimum
cost. The assignment is exhaustive over all `G!` permutations, which is why
`ParamMatcher.MAX_GAUSSIANS` is 6 and why the class raises above that. At `K = 1` both
choices are no-ops.

If your target is plain dense regression, say one value per pixel or a fixed vector
with no permutation freedom, the shape of the work is:

1. Use the `conv` head with `out_channels` set to your channel count.
2. Keep the curve terms and drop the parameter terms. The catalog in
   `pipelines/backbone/training/loss_terms.py` splits exactly along that line: the terms
   marked `denorm` (`mse_curve`, `l1_curve`, `huber_curve`, `charbonnier_curve`,
   `cosine_curve`, `total_power_relerr`, `moments`) compare reconstructed curves; the ones
   marked `norm` (`param_l1`, `param_huber`, `param_mse`, `smoothness_tv`) compare
   parameters. Turn the `use_param_*` flags off in both curriculum stages.
3. Replace `Loss` in `pipelines/backbone/training/loss.py`. Turning the flags off is not
   enough on its own: `Loss.__call__` reconstructs curves through `GaussianCurve`, calls
   `_match_params`, and computes `amp_mae`, `mu_mae` and `sigma_mae` on every batch
   regardless of which terms are active. That class is where the Gaussian assumption
   really lives, and it is one file with one class.

## 3. Normalization slots

Every channel, in and out, is normalized by a strategy chosen from its slot key. The
table is `_SLOT_STRATEGIES` at the bottom of `configuration/normalization/general.py`:

```
chan/mag           robust_iqr_log1p      derived/mag         robust_iqr_log1p
chan/raw_re_im     zscore                derived/raw_re_im   zscore
chan/norm_re_im    zscore                derived/norm_re_im  zscore
chan/phase         zscore                derived/phase       fixed_div_pi
out/amp            robust_iqr_log1p      aux/value           zscore
out/mu             zscore
out/sigma          robust_iqr_log1p
```

Input slot keys are produced by `InputConfig.channel_group_keys`, one per emitted channel:
the primary and the secondaries produce `chan/<kind>`, the derived products produce
`derived/<kind>`, where the kind comes from the `Representation`, and the auxiliary field
produces `aux/value`. The output keys are fixed at `out/amp`, `out/mu` and `out/sigma`.

`NormalizationConfig` in the same file is the override surface.
`input_strategy`/`output_strategy` set a single preset for the whole side, or `per_slot`
to use the table. The per-slot fields (`chan_mag`, `channel_phase`, `derived_mag`,
`derived_phase`, `out_amp`, `out_mu`, `out_sigma`, `aux`) override one slot each and take
precedence; `SLOT_FIELDS` maps a slot key to its field. The presets are listed in
`Presets.names()`. `fixed_bounds` is a special case: it cannot be fitted, and it reads its
range from `fixed_out_bounds_min`/`fixed_out_bounds_max`, one entry per output channel.

The fitted statistics land in `meta/normalization_stats.json` in the run directory, as
`ChannelStats`: a location, a scale, a name, the strategy and a clampable flag per
channel. Inference reloads them from there, so a run is self-contained. The fitting itself
is `pipelines/backbone/dataset/stats_computer.py`, over the training split only.

Adding a slot means three edits: emit its key from `channel_group_keys`, add an entry to
`_SLOT_STRATEGIES`, and add a field plus a `SLOT_FIELDS` entry if it should be
overridable from the command line. The third one drags the console gates in with it, which
is the next section.

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

### The flow catalog and the pipeline catalog describe the same pipelines

`webui/flow_library.py` and `webui/pipeline_library.py` are hand-written, and their keys
have to match one for one, in order.

- Fails in: `tests/webui/test_content_libraries.py::test_the_flow_catalog_and_the_pipeline_catalog_agree`.
  `::test_pipeline_scripts_point_at_real_entry_points` catches a pipeline card naming a
  script the console cannot launch.
- Fix: add or remove the entry in both files.

### The curated repo map points at files that exist

`webui/repomap_data.json` names modules by path, and `webui/repomap_derive.py` walks the
real import graph and reports drift.

- Fails in: `tests/webui/test_repomap_derive.py::test_curated_repomap_references_only_existing_files`.
  `tests/webui/test_repomap_library.py` separately checks that node ids are unique, grid
  cells do not collide, edges point at declared nodes and artifacts are well formed.
- Fix: update the folder entry, or delete it with the subsystem it described.

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
`runs/`, `results/`, the example dataset and its parameter run), and the machine profile
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
5. A note file under `notes/models/`, named by `NOTE_FILES`. The library reads it through
   `model_library_base.py` and the console renders it on the Architectures tab.
6. Baseline entries in `tests/state_dict_baseline.json`, per section 4.

`tests/webui/test_backbone_model_library.py` checks 3, 4 and 5 against each other:
`test_config_classes_cover_the_backbone_registry`, `test_note_files_cover_the_backbone_registry`,
`test_families_cover_the_backbone_registry_exactly_once` and `test_every_model_resolves_a_note`.
`tests/models_backbone/test_backbone_architectures.py` puts the model itself through every
head. The autoencoder, JEPA and dual families follow the same pattern in their own
`models/<family>/__init__.py` and model library.

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

The curated content is optional but goes stale silently if you skip it: a pipeline needs
its pair of entries in `flow_library.py` and `pipeline_library.py`, and a new subsystem
deserves a folder in `repomap_data.json`. Nothing per entry is needed for saved runs,
config templates or the machine profile: they work from the layout and the config
dataclass alone. A new console tab that shows images registers with the figure saver
instead, through a `figureTag()` method and `data-fig` labels, as `webui/README.md`
describes under "Saving figures".

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
- The browser storage prefix `mlt.` in `webui/static/js/configs.js`
  (`mlt.config.open`, `mlt.config.active`). Changing it resets which config sections a
  returning user had open, which is the point if two consoles share a host.
- Scratch paths in the visual harness: `webui/tools/visual/refresh.sh`,
  `snap_ui.js`, `snap_repomap.js`, `dump_frames.js` and `package.json`.
