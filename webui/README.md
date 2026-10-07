# ML Template control console

A single-page web console for the ML Template project. It documents the model and the
pipelines, launches and supervises the `main/` entry points, and browses everything those
runs produce: metrics, reports, figures, prediction cubes and TensorBoard curves.

## Run

```bash
webui/run.sh            # http://127.0.0.1:8772
webui/run.sh 9000       # custom port
```

`run.sh` sources the machine profile's shell twin (`~/.config/ml-template/machine.env`,
written by `./install.sh`) and executes `serve.py --host <host> --port <port>` under the
interpreter recorded there; without a profile it stops and points at the installer.
`serve.py` also takes `--root`, the project root the console operates on (job adoption,
logs, GPU pools, runs), default the repository that holds `serve.py`. Tests run with
`MLTEMPLATE_MACHINE_PROFILE` empty so no developer profile leaks into their expectations.

The profile also supplies the runs root the console browses, the data and runs roots the
results tabs start from (`/api/project` carries them as `roots`), the interpreter
preference of the launch pages, the GPU pool the guard schedule is seeded with and the
ntfy seed. Launch pages mark every field whose default the profile rewrote and count
them in the facts rail.

The HTTP transport is standard library only (`http.server` plus server-sent events), but
the console itself needs the project environment. `web_ui_server.py` imports the cube
explorer, fit lab, model probe, A/B autopsy, triage board, training-curve reader and
system monitor at boot, and those pull numpy, matplotlib, scipy, torch, scikit-image,
tensorboard and psutil. On a bare interpreter the server fails with an ImportError before
it binds the port, which is why `run.sh` starts from the interpreter the profile records.

The frontend loads MathJax 3.2.2 (tex-svg), highlight.js 11.9.0, GSAP 3.12.5 and xterm.js
5.5.0 with its fit addon from a CDN, so equation typesetting, syntax highlighting,
animation and the terminal view need a network connection. `marked` is vendored under
`static/vendor/`. Everything degrades to plain text offline.

## Tabs

Navigation is hash-based, `#/<route>`. Each tab is one view module under `static/js/`
talking to one group of JSON endpoints.

| Tab | Route | Backed by | What it does |
|---|---|---|---|
| Home | `#/home` | `/api/system`, `/api/jobs`, `/api/gpu-schedule`, `/api/gpu-guard/history`, `/api/notify/*` | Live host and GPU telemetry, per-user memory attribution, running-job tiles, the GPU guard history, the charity-mode schedule, the neighbour-impact alarm and the ntfy settings. |
| Model | `#/model` | `/api/equations`, `/api/flows` | The curated equation catalog, 6 groups and 111 items, rendered with MathJax, plus the flow walkthroughs: 15 animated step-by-step derivations from dataset synthesis and parameter extraction through training, inference, benchmarking and tuning. |
| Pipelines | `#/pipelines` | `/api/pipelines` | The 15 pipelines as staged flows; each maps to the entry point that runs it. |
| Repo Map | `#/repomap` | `/api/repomap` | Curated module-level schematics, 13 folders and 29 diagrams, each with nodes, edges and the artifacts a stage reads and writes. |
| Architectures | `#/architectures` | `/api/backbones`, `/api/profile-autoencoders`, `/api/image-autoencoders`, `/api/jepa-variants` | The 20 backbones across 5 families plus their 4 output heads and the autoencoder and JEPA variants, with selection guidance and per-model defaults. |
| Scripts | `#/scripts` | `/api/scripts` | The entry points as cards. `ScriptCatalog` describes 37 of them and folds the per-family variants into 4 groups (train, infer, analyze, compare). |
| Launch | `#/launch/<key>` | `/api/scripts/<key>/config` | Full-screen launch control per script: config sections grouped by dataclass, typed controls, the override manifest, the command preview and the interpreter. Launch now, queue after the current job, or save for later. Entries that write a sweep (the training entries with their trial modes, cross-validation and the benchmark) also carry a Finalize section: the seed comparison, the trial comparison and the report mail are chained onto the finished run one at a time, each starting only once the previous one exited cleanly. The chain is derived before the run starts, so a run tag left to the timestamp default or a missing SMTP password is refused at the button instead of failing hours later. |
| Ablation | `#/ablation` | `/api/backbones`, `/api/run` | Reached from a script card. Picks the entry, the features to ablate and their order, then launches a cumulative loop that trains the full model and degrades one feature at a time down to the baseline. |
| Saved | `#/saved` | `/api/saved-runs` | Configurations stored from launch pages, one JSON file each under `logs/saved_runs/`, finalize switches included. Each card can be launched, queued or deleted. |
| Templates | `#/templates` | `/api/config-templates` | Named sets of config overrides captured from a launch page or a sequence block, one JSON file each under `logs/config_templates/`. A template carries only the fields the form overrode, so it can be imported on any entry point: matching dotted paths apply and the rest are reported as skipped. Each card opens its entry point with the template already applied, renames or deletes it; the `Config templates` rail block on every launch page imports, saves and deletes them without leaving the form. Names are unique per entry point, and saving over one replaces it. |
| Sequences | `#/sequences` | `/api/sequences/*` | Experiment sequences as a drag-and-drop DAG board: launch blocks chained by predecessors with checkpoints, skip toggles, per-block decision axes, an isolated per-execution workspace, resume scans over the persisted workspace state, and per-block finalize chains carrying the same switches as the launch page, chained onto the block's own end inside the workspace without holding back its successors. |
| Survey | `#/survey` | `/api/survey/*` | Model survey over many trained runs at once: pick runs, and compare their weight statistics, spectra and health indicators side by side. |
| Configuration | `#/configuration` | `/api/configs` | Every configuration dataclass, field, type and default, parsed live from `configuration/`. |
| Console | `#/console` | `/api/jobs`, `/api/jobs/<id>/stream` | Real-time stdout of launched jobs over SSE with stop control. Fan-out experiments add a progress strip from `/api/jobs/<id>/progress`: units done of total, mean unit duration, ETA and projected finish. |
| TensorBoard | `#/tensorboard` | `/api/tensorboard/*` | Starts and supervises a TensorBoard process against any run directory. |
| Leaderboard | `#/leaderboard` | `/api/leaderboard`, `/api/curves` | Three modes. Runs: every saved inference `metrics.json` as one sortable, filterable table with run-name axes parsed into dropdowns and a metric plus resolved-config diff; seeded layouts also get one seed-mean row per unit. Trials: seeded runs aggregated to mean-and-std bars. Curves: native overlay of TensorBoard scalars across runs. |
| Results | `#/results` | `/api/results/*`, `/api/fs/*` | Browse any run or dataset directory: rendered markdown reports, figure galleries with compare pinning, configs, and inline log and text files. |
| Cube | `#/cube` | `/api/cubes/*` | Cube explorer over saved inference cubes: live cuts with profiles and SSIM, axis sweeps, Gaussian parameter maps with per-pixel slot readout, per-pixel metric overlays with thresholding, arbitrary-line transects, a 3D component point cloud, run-vs-run comparison, selectable colormaps and paper-style figure export. A load reads every source whole into memory through a process-wide read pool (`NpyReader`, up to 32 threads) that splits each file into 16 MB `pread` chunks and keeps them in flight together, since concurrent whole-file reads beat memory-mapped page faults on a network mount, and the sources after the prediction cube are fetched in parallel. The progress bar follows the bytes read against the byte plan of the load and names the files still in flight; a load whose status has not moved for 5 s also shows how long it has been silent and the innermost frames of its thread. Every stage books its wall time: the ready line in the console log lists the stage times, the ready payload carries them under `timings`, and the tab shows the total next to the slowest stage. The full reference cube is read as the row band the cube covers, one contiguous read per z plane, and the last eight loaded cubes stay in memory (`CubeExplorer.CUBE_CACHE_ENTRIES`), so switching back to one is immediate and drops any attached comparison. `kill -USR1 <pid>` dumps every thread's stack of the console to its stderr without stopping it. |
| Slices | `#/slices` | `/api/slices/*` | Multi-run slice collector: tick any number of cubes, hover the primary map to preview the same cuts from every run side by side, and click the map (or a slice's own button) to put cuts in a basket that persists in the browser. Every basket entry keeps its own run, source, cut, point, value space and colormap. The basket renders into one `slice_collections/<name>/` folder with a `collection.json` manifest, and can be mailed as zip volumes through the report mailer (`REPORT_MAIL_PASSWORD`); a basket already saved under the same name is mailed from that folder. A slice that cannot be rendered clears its cell and names the reason in the console log. |
| Fit Lab | `#/fitlab` | `/api/fitlab/*` | Parameter-extraction playground over a dataset: pick pixels on the input or profile-peak map, then run the real Gaussian-fit stack (`PeakInitialiser` plus `SigmaAdamKernel` on CPU) on just those pixels. Each run is a labelled overlay with a per-K mixture, a per-run K override and a penalised-MSE K sweep. |
| Microscope | `#/microscope` | `/api/probe/*` | Model microscope over a trained run: load the checkpoint with its real dataset, click a pixel, and inspect the predicted profile against ground truth, gradient attribution of every output family to every input channel with per-pair spatial maps, feature-map grids along a clickable architecture strip, and live what-if perturbations with per-slot deltas. |
| Triage | `#/triage` | `/api/triage/*` | Error triage over a saved inference: the worst blocks ranked by pixel MSE, tagged with dominant failure mode, label fit, seed disagreement and flip disagreement where those cubes exist. Each case opens the cube explorer at its worst pixel and takes a persisted verdict plus a note under `logs/triage/`. |
| Autopsy | `#/autopsy` | `/api/autopsy/*` | Guided A/B comparison of two inferences over the same region: the largest relative metric gaps with an orientation-aware winner per row, plus the blocks where the runs disagree most, each rendered as both predicted profiles against the ground truth. |
| Feed Tuner | `#/feedtuner` | `/api/scripts/tune_dataloader/config`, `/api/run` | Launch surface for the DataLoader sweep with its own result view. |

## Saving figures

The tabs that show images have a figure saver: Microscope, Survey, Triage, Autopsy, Fit Lab,
Results, Leaderboard, Cube and Slices. It runs in the browser over what is already on
screen, so the server renders nothing extra and no run directory is written.

- **One figure.** Hovering a chart, map or plot floats a small save button in its top-right
  corner, and pressing it downloads that figure as a PNG.
- **Every figure.** A dock at the bottom right counts the figures on the view and packs them
  into one ZIP.
- **A subset.** The dock's second button opens the list of those figures, grouped by the card
  they sit in. Each row has its own PNG button and a checkbox, and the selection saves as one
  ZIP. Hovering a row outlines the figure on the page and clicking it scrolls there.

A file name carries the tab, the view's context and the figure, so a single save lands as
`microscope_<run>_row0413_col0207_predicted_profile.png` and a batch as
`<tab>_<context>_figures.zip`. Each tab supplies its context through a `figureTag()` method.
A figure takes its label from its `data-fig` attribute, or failing that from its `alt` text,
its caption or the heading above it.

The scanner only sees figures that are visible, so on a tab with paged analyses the batch
covers the page on screen. Canvases are written at their backing-store resolution over an
opaque background. Small server-rendered maps are upscaled by whole pixels the way the page
shows them, and vector charts are rasterized at twice their layout size with their computed
styles inlined. `figure_export.js` holds the mechanics, including a store-only ZIP writer,
and `figure_saver.js` the scanner and the dock. The cube explorer keeps its server-side
paper-style figure export beside this.

## How launching works

Every `main/*.py` script builds its configuration as `ConfigCli(<EntryConfig>()).apply()`,
with all defaults living in `configuration/` dataclasses. The console mirrors that
contract and never writes to a config file.

- **Resolution.** `ScriptConfigResolver` finds the entry dataclass by parsing the script's
  AST, then resolves its defaults in a subprocess using the preferred project interpreter,
  exactly as `--help-config` would. Results are cached and invalidated when any file under
  `configuration/`, the script itself, or `tools/runtime/config_cli.py` changes mtime.
- **Layout.** `launch_layout.py` decides which fields the launch form exposes and which
  widget each one gets. It is the single source of truth for the form and refuses to build
  a section containing a field it does not claim, so a new config field surfaces as a loud
  failure rather than a silently missing control. `config_descriptions.json` supplies the
  per-field help text under the same strictness.
- **Overrides.** Edited fields are appended to the command as `--<dotted.path> <value>`,
  which `ConfigCli` coerces to the field's type. The command preview always shows exactly
  what will run.
- **Launch.** `ProcessManager` runs the script from the repository root with the selected
  interpreter and streams its output over SSE. Exit codes are reported as they are.
- **Queue.** Schedule captures the same command but places it in a sequential queue. A
  queued run starts once every running job and every earlier queued run has ended,
  regardless of exit code. Queued runs can be cancelled before they start; an emergency
  stop clears the whole queue.
- **Supervision.** `GpuWatchdog`, `ResourceWatchdog` and `ContentionMonitor` guard GPU
  intrusion, host memory and neighbour impact; `JobNotifier` pushes start and end events
  over ntfy.

Scripts expect the dataset paths and GPUs named in the configuration defaults. On a
machine without them, a launched job streams its real traceback to the console.

## Architecture

One class per file, no comments, per the project coding rules.

**Server.** `serve.py` builds and runs the threaded HTTP server in `web_ui_server.py`,
which owns one instance of every collaborator and composes the routing table. Requests
enter `request_router.py`, which resolves the path's leading section to exactly one
sub-router under `routers/` and delegates: `routers/dispatch.py` carries the shared
`HttpExchange` (query accessors, JSON, PNG, byte and file responses) and the per-router
`RouteTable`; `static_router.py`, `results_routers.py`, `cube_routers.py`,
`analysis_routers.py`, `library_routers.py`, `launch_routers.py`, `system_router.py` and
`tensorboard_router.py` declare the routes of their own domain and hold only the
collaborators they use. A section may be claimed by one sub-router only; a second claim
raises at construction. `project_paths.py` resolves repository paths and candidate
interpreters, `catalog_roots.py` gates filesystem browsing to roots the user has opened,
and `web_logger.py` handles console logging.

**Launching and supervision.** `script_catalog.py`, `script_config_resolver.py`,
`launch_layout.py`, `config_registry.py`, `run_launcher.py`, `process_manager.py`,
`finalize_chain.py`, `saved_run_store.py`, `config_template_store.py`, `job_describer.py`, `notifier.py`, `gpu_schedule.py`,
`gpu_watchdog.py`, `resource_watchdog.py`, `contention_monitor.py`, `system_monitor.py`,
`proc_stats.py`, `tensorboard_manager.py`.

**Curated content.** Hand-written project documentation served as JSON and kept honest by
tests under `tests/webui/`: `equation_library.py`, `flow_library.py`,
`pipeline_library.py`, `repomap_data.json` with `repomap_library.py` and its drift checker
`repomap_derive.py`, `config_descriptions.json`, and the model libraries
`backbone_model_library.py`, `profile_autoencoder_model_library.py`,
`image_autoencoder_model_library.py` and `jepa_model_library.py` on the shared
`model_library_base.py`, which reads the per-model notes from `notes/models/`.

**Analysis surfaces.** `results_browser.py`, `dataset_browser.py`, `run_leaderboard.py`,
`training_curves.py`, `cube_explorer.py` which also carries the slice collector,
`fit_lab.py`, `model_probe.py`, `triage_board.py`, `ab_autopsy.py`.

**Frontend.** `static/index.html`, `static/css/styles.css`, `static/vendor/marked.min.js`
and 48 modules under `static/js/`. `app.js` and `router.js` wire the views; one module per
tab (`home`, `equations`, `flow_view` with `flow_sketches`, `pipelines`, `repomap`,
`models`, `scripts`, `launch` with `launch_widgets` and `launch_pickers`, `ablation`,
`saved_runs`, `templates`, `configs`, `console`, `tensorboard`, `leaderboard`, `results`, `cube_view`,
`slice_collector`, `fit_lab`, `microscope`, `triage`, `autopsy`, `feed_tuner`); the rest
are shared helpers, `run_strip` for the run picker the cube, slices, microscope, triage
and autopsy tabs share, `figure_export` and `figure_saver` for the figure saver, and
`radar`, `diagram`, `gauges`, `process_anim`, `server_anim` and `training_anim` for
drawing and animation.
