# ML-Template control console

A single-page web console for ML-Template. It documents the model zoo,
launches and supervises the `main/` entry points, and browses everything those runs
produce: metrics, reports, figures, prediction maps and TensorBoard curves.

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
the console itself needs the project environment. `web_ui_server.py` imports the
model probe, A/B autopsy and system monitor at boot, and those pull numpy, matplotlib, scipy, torch,
tensorboard and psutil. On a bare interpreter the server fails with an ImportError before
it binds the port, which is why `run.sh` starts from the interpreter the profile records.

The frontend loads MathJax 3.2.2 (tex-svg), highlight.js 11.9.0, GSAP 3.12.5 and xterm.js
5.5.0 with its fit addon from a CDN, so equation typesetting, syntax highlighting,
animation and the terminal view need a network connection. `marked` and the hairline
kernel with its stock figures are vendored under `static/vendor/`.
Everything degrades to plain text offline.

## Tabs

Navigation is hash-based, `#/<route>`. Each tab is one view module under `static/js/`
talking to one group of JSON endpoints.

| Tab | Route | Backed by | What it does |
|---|---|---|---|
| Home | `#/home` | `/api/system`, `/api/jobs`, `/api/gpu-schedule`, `/api/gpu-guard/history`, `/api/gpu-activity`, `/api/host-history`, `/api/server-log`, `/api/notify/*`, `/api/impact/*` | Live host and GPU telemetry as a board of cards: the watchdog, notify and GPU week control cards (a kill switches card stacked under notify), the CUDA devices beside the processor, storage and memory, the day/week/month GPU activity timeline with intrusion markers, the neighbour-impact card, the persistent host history (CPU, RAM and network on a scaled time axis with a hover readout), the jobs card, the GPU guard history, the server's own log, the 30-day GPU heatmap calendar, and the active users and processes at the end. Alerts show for 90 s after they first appear and then hide; the watchdog status reopens them and the dismiss button hides them at once. The watchdog card also holds the launch politeness select and the auto-throttle toggle (`/api/impact/politeness`), and keep-alive, which detaches the server from the terminal and the login session into its own `ml-template-webui-<pid>.scope` systemd scope; the kill switches card pairs the nuke, which kills every process of this user on this machine but the console, with kill front end, which stops every console server on the same project root (`/api/system/shutdown` force-kills the siblings, then stops itself and returns their pids); running and detached jobs survive and are adopted again on the next start. |
| Architectures | `#/architectures` | `/api/backbones` | The 20 backbones across 5 families, with each model's note from `notes/models/`, an architecture diagram and the per-model defaults. |
| Scripts | `#/scripts` | `/api/scripts` | The entry points as cards. `ScriptCatalog` describes all 24 and folds the three comparison entries into one Compare card. |
| Launch | `#/launch/<key>` | `/api/scripts/<key>/config` | Full-screen launch control per script: config sections grouped by dataclass, typed controls, the override manifest, the command preview and the interpreter. Launch now, queue after the current job, or save for later. Entries that write a sweep (the training entries with their trial modes, cross-validation and the benchmark) also carry a Finalize section: the seed comparison, the trial comparison and the report mail are chained onto the finished run one at a time, each starting only once the previous one exited cleanly. The chain is derived before the run starts, so a run tag left to the timestamp default or a missing SMTP password is refused at the button instead of failing hours later. |
| Saved | `#/saved` | `/api/saved-runs` | Configurations stored from launch pages, one JSON file each under `logs/saved_runs/`, finalize switches included. Each card can be launched, queued or deleted. |
| Templates | `#/templates` | `/api/config-templates` | Named sets of config overrides captured from a launch page or a sequence block, one JSON file each under `logs/config_templates/`. A template carries only the fields the form overrode, so it can be imported on any entry point: matching dotted paths apply and the rest are reported as skipped. Each card opens its entry point with the template already applied, renames or deletes it; the `Config templates` rail block on every launch page imports, saves and deletes them without leaving the form. Names are unique per entry point, and saving over one replaces it. |
| Sequences | `#/sequences` | `/api/sequences/*` | Experiment sequences as a drag-and-drop DAG board: launch blocks chained by predecessors with checkpoints, skip toggles, per-block decision axes, an isolated per-execution workspace, resume scans over the persisted workspace state, and per-block finalize chains carrying the same switches as the launch page, chained onto the block's own end without holding back its successors: inside the workspace in an isolated sequence, on the entry's own sweep directory otherwise. A tuning block that declares `model_overrides` (and, when its study samples the batch size, `training.batch_size`) records the best configuration its run wrote as that verdict once its job ends, so the blocks after it run on the tuned values; while it runs, the board prints the study snapshot its workers write (trials done, best value, stall count, rate and time to the target) from `/api/sequences/<id>/study?block=`. The notifier announces a block held at its checkpoint and a recorded verdict. |
| Survey | `#/survey` | `/api/survey/*` | The microscope's diagnostics over every tile of a split: RMSE, MAE and R² per target channel, input attribution and ablation, flip symmetry, noise robustness, the occlusion distance profile, the receptive field and layer vitals, averaged over the region with a progress bar and a cancel button. |
| Configuration | `#/configuration` | `/api/configs` | Every configuration dataclass, field, type and default, parsed live from `configuration/`. |
| Console | `#/console` | `/api/jobs`, `/api/stream` | Real-time stdout of launched jobs with stop control. The page opens one server-sent-events channel (`GET /api/stream?channel=<id>`) and subscribes every job tile and every selected sub-run to it through `POST /api/stream/<channel>/subscribe` and `/unsubscribe` (body `job_id`, optional integer `unit`, and `after`, the last event id held, 0 for none; anything else is a 400), so any number of open tiles costs the browser a single connection out of its six per origin instead of one each, which used to freeze every other request once six tiles or sub-runs were open. Every event carries its subscription key, and viewers of the same job share one subscription: a viewer that joins a job another one already follows (a tile opened after another tile on the same job) re-subscribes it from the first event and is replayed the buffered output, while the earlier viewer drops the event ids it already holds. A channel that cannot keep up is truncated with a marker rather than allowed to stall the publishers. A dropped channel reconnects and each job stream replays from its last event id (a sub-run tail restarts from its top, and a restarted server, recognised by the `boot` token in the channel's `hello` frame, replays everything); while it is down, and after a console restart, a tile takes its status from the polled job list, and a tile whose job the console no longer knows turns `untracked` with its Stop disabled. The terminal refits whenever its tile changes size, and the sub-run menu caps its height at the room left inside the tile. Fan-out experiments add a progress strip from `/api/jobs/<id>/progress`: units done of total, mean unit duration, ETA and projected finish. Once a job has ended (finished, failed, cancelled or untracked) its tile shows Retry, which reopens the launch page with the overrides, interpreter, detach switch, follow-up and finalize switches the job was launched with, ready to adjust and launch again; each job record keeps that request under `request` (overrides before the per-job GPU pool file). |
| TensorBoard | `#/tensorboard` | `/api/tensorboard/*` | Starts and supervises a TensorBoard process against any run directory. The run picker browses the runs root as a folder tree from `/api/tensorboard/tree`: a run is a folder holding `tensorboard/` and is never descended into, a folder of runs only is a set served on click, and a folder holding further run folders is a group that expands, with an open-all button serving the whole group. |
| Results | `#/results` | `/api/results/*`, `/api/fs/*` | Browse any run or dataset directory: rendered markdown reports, figure galleries with compare pinning, configs, and inline log and text files. |
| Microscope | `#/microscope` | `/api/probe/*` | Model microscope over a trained run: load the checkpoint with its real dataset, click a pixel, and inspect the prediction of every target channel against its target, the local prediction, target and error fields, gradient attribution of every target channel to every input channel with per-pair spatial maps, channel ablation, occlusion and flip symmetry, and feature-map grids along a clickable architecture strip. |
| Autopsy | `#/autopsy` | `/api/autopsy/*` | Guided A/B comparison of two inferences over the same region: the largest relative metric gaps with an orientation-aware winner per row, plus the blocks where the standardized errors of the runs differ most, each listing both predictions of every target channel against the target. |

## Saving figures

The tabs that show images have a figure saver: Microscope, Survey, Autopsy and
Results. It runs in the browser over what is already on
screen, so the server renders nothing extra and no run directory is written.

- **One figure.** Hovering a chart, map or plot floats a small save button in its top-right
  corner, and pressing it downloads that figure as a PNG.
- **Every figure.** A dock at the bottom right counts the figures on the view and packs them
  into one ZIP.
- **A subset.** The dock's second button opens the list of those figures, grouped by the card
  they sit in. Each row has its own PNG button and a checkbox, and the selection saves as one
  ZIP. Hovering a row outlines the figure on the page and clicking it scrolls there.

A file name carries the tab, the view's context and the figure, so a single save lands as
`microscope_<run>_row0413_col0207_pointwise_prediction.png` and a batch as
`<tab>_<context>_figures.zip`. Each tab supplies its context through a `figureTag()` method.
A figure takes its label from its `data-fig` attribute, or failing that from its `alt` text,
its caption or the heading above it.

The scanner only sees figures that are visible, so on a tab with paged analyses the batch
covers the page on screen. Canvases are written at their backing-store resolution over an
opaque background. Small server-rendered maps are upscaled by whole pixels the way the page
shows them, and vector charts are rasterized at twice their layout size with their computed
styles inlined. `figure_export.js` holds the mechanics, including a store-only ZIP writer,
and `figure_saver.js` the scanner and the dock.

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
  per-field help text under the same strictness. The dataset field of every training form
  is a picker over the folders that hold a `dataset.json`, badged with their input and
  target channel counts. An `arch_overrides` panel reads its model field from one path
  (`modelFrom`), and the tune form groups its study switches under Search groups and its
  plateau and time-budget rules under Stop rules.
- **Overrides.** Edited fields are appended to the command as `--<dotted.path> <value>`,
  which `ConfigCli` coerces to the field's type. The command preview always shows exactly
  what will run.
- **Launch.** `ProcessManager` runs the script from the repository root with the selected
  interpreter and streams its output over the page's live-output channel. Exit codes are reported as they are.
- **Queue.** Schedule captures the same command but places it in a sequential queue. A
  queued run starts once every running job and every earlier queued run has ended,
  regardless of exit code. Queued runs can be cancelled before they start; an emergency
  stop clears the whole queue.
- **Supervision.** `GpuWatchdog`, `ResourceWatchdog` and `ContentionMonitor` guard GPU
  intrusion, host memory and neighbour impact. The contention monitor reads every login
  user's cgroup slice under `user.slice`: the slice's own pressure files say whether that
  user's tasks are waiting for CPU, memory or disk, and `cpu.stat` and `memory.stat` give
  each user's share of the resource, exited children included. An alarm needs a neighbour
  stalled for two samples and is this user's only when this user is the dominant
  consumer; disk has no per-user byte counter, so its ownership falls back to this
  user's own processes' share of the repo disk. Without `/proc/pressure` or `user.slice`
  the monitor stays off and the panel says so. `JobNotifier` pushes start and end events
  over ntfy.
- **Politeness.** `JobPoliteness` (`politeness.py`) holds the level every local launch
  starts under, persisted in `logs/politeness.json`: `off` adds no prefix, `polite` (the
  default) prefixes the command with `nice -n 10 ionice -c 2 -n 7`, `background` with
  `nice -n 19 ionice -c 3`. `nice -n` adds to the console's own niceness, capped at 19.
  The prefix execs into the interpreter, so the job pid is the interpreter's and the job
  record's command line shows it. `JobThrottle` is the runtime
  lever: with auto-throttle on (the default), the first alarm the contention monitor
  attributes to this user runs `renice -n 19` and `ionice -c 3` over every process this
  user owns except the console's own line, once per alarm until it clears. Priorities can
  only be lowered without privileges, so a throttled job stays throttled until it ends.
  Both are set from the watchdog card on the Home page (`POST /api/impact/politeness`).

Scripts expect the dataset paths and GPUs named in the configuration defaults. On a
machine without them, a launched job streams its real traceback to the console.

## Architecture

One class per file, no comments, per the project coding rules.

**Server.** `serve.py` builds and runs the threaded HTTP server in `web_ui_server.py`,
which owns one instance of every collaborator and composes the routing table. Requests
enter `request_router.py`, which resolves the path's leading section to exactly one
sub-router under `routers/` and delegates: `routers/dispatch.py` carries the shared
`HttpExchange` (query accessors, JSON, PNG and file responses) and the per-router
`RouteTable`; `static_router.py`, `results_routers.py`,
`analysis_routers.py`, `library_routers.py`, `launch_routers.py`, `system_router.py` and
`tensorboard_router.py` declare the routes of their own domain and hold only the
collaborators they use. Each section belongs to one sub-router; when two claim the same
section, the last one listed wins. `project_paths.py` resolves repository paths and candidate
interpreters, `catalog_roots.py` gates filesystem browsing to roots the user has opened,
and `web_logger.py` handles console logging.

**Launching and supervision.** `script_catalog.py`, `script_config_resolver.py`,
`launch_layout.py`, `config_registry.py`, `run_launcher.py`, `process_manager.py`, `stream_hub.py`,
`finalize_chain.py`, `saved_run_store.py`, `config_template_store.py`, `job_describer.py`, `notifier.py`, `gpu_schedule.py`,
`gpu_watchdog.py`, `resource_watchdog.py`, `contention_monitor.py`, `system_monitor.py`,
`proc_stats.py`, `tensorboard_manager.py`.

**Curated content.** Hand-written project documentation served as JSON and kept honest by
tests under `tests/webui/`: `config_descriptions.json` and the model library
`backbone_model_library.py`, which reads the per-model notes from `notes/models/`.

**Analysis surfaces.** `results_browser.py`, `dataset_browser.py`,
`model_probe.py`, `model_survey.py`, `ab_autopsy.py`.

**Frontend.** `static/index.html`, `static/css/styles.css`, `static/vendor/marked.min.js`,
`static/vendor/hairline-kernel.js`,
`static/vendor/hairline-figures.js` and 36 modules under `static/js/`. `app.js` and
`router.js` wire the views; one module per tab (`home`, `models`, `scripts`, `launch` with its widget
modules (`python_literal`, `form_widgets`, `model_panels`, `gpu_pickers`,
`experiment_builder`, `config_form`) and `launch_pickers`, `saved_runs`,
`templates`, `sequences`, `configs`, `console`, `tensorboard`, `results`,
`microscope`, `survey`, `autopsy`);
the rest are shared helpers (`api`, `stream_hub` with the page's single live-output channel, `shared_charts`,
`results_sources`, `run_strip`, `canvas_base`, `figure_export`, `figure_saver`, `diagram`,
`gauges`, `training_anim`, `hairline_figure` with
`hairline_project`).

The console wears one dark monitor theme: the root tokens give a near-black page, dark
translucent glass surfaces with near-white ink and light blue and teal accents, text on
an accent fill uses `--on-accent`, every border, ring and divider is 2 px, and the body
text runs at weight 600. Panels carry alpha so the backdrop shows through them; sticky
bars, table headers, drawers, popovers and the figure saver use a frosted blur instead,
because content scrolls or opens beneath them. The terminals run on a transparent theme
with a bright ANSI palette at a 7:1 minimum contrast. Charts and diagrams draw the light
colour of each series, and an exported figure takes the dark page as its backdrop.

Every wait on the backend shows a short status line (`SharedCharts.waitLine` and
`SharedCharts.waitPanel`): empty containers before their first render, picker and hint
notes, progress cards and the microscope and survey computations.

`hairline_figure` defines `<hairline-figure name>`, an isometric line drawing that
answers the pointer, over the vendored hairline kernel (Lucas Marques, MIT). It marks
itself `data-nofig` so the figure saver skips it, and shows no caption. A figure that
draws live data takes it through the element's `state` property: the element hands the
object to the figure's `setState`, re-applies it after a remount, forgets it when `name`
changes, and throws when the figure has no `setState`. The data sets what stays visible
under any pointer (counts, lights, speeds); the pointer still picks. While a figure is
on screen and the pointer is elsewhere, `HairlineAutopilot` moves a synthetic pointer
across it in a 3.2 s rhythm: it glides for half the cycle and drifts slowly for the other
half, so a picked part holds long enough to read and every figure keeps moving; a real
pointer takes over, the sweep resumes 2.2 s after it leaves, and nothing sweeps under
reduced motion. A figure with `autopilot="off"` gets no sweep of its own: the Home mast
sets it on the two figures flanking the server name, and `MastFigures` drives the pair
from one autopilot on the mast, so both answer the same pointer; any header with the
`mast--flanked` class gets the same treatment.

The stock `drawer`, `terminal` (turned to face right), `plug`, `riffle`, `loupe`, `sieve`,
`elevator`, `slow`, `cabinet`, `router`, `patch` and `phosphor` come from the package;
`hairline_project` draws the project's own on a shared `HairlineFigureBase`:
`folds`, `gpus`, `guard`, `backbone` (the backbone zoo as a
bench of swappable cartridges), `landscape`,
`surface` (two bumps riding a straight track across a ridgeline surface),
`balance`, `podium`, `tiles`, `film`, `tray`, `shelf`, `dial`, `scheduler`, `sampler`, `feed`, `field`,
`watchdog`, `notify`, `nuke`, `cores`, `gpuwire`, `guillotine`, `chips`, `calendar` and
`hourglass`. Thirteen of them take `state`: `gpus` (`{devices: [{holder, util}]}`),
`balance` and `chips` (`{mine, others, alarm}`), `guard` (`{gpus, invaded, critical}`), `scheduler`
(`{jobs: [{id, running, position, of}]}`), `watchdog` (`{armed, gauges: [{level, limit, alert}]}`, four gauges for CPU,
load, RAM and disk), `notify` (`{armed}`), `nuke` (`{firing}`), `cores` (`{cores}`, one
busy percentage per core, the field relaid when the count changes), `gpuwire` (`{util,
power}`, utilisation in percent and power draw as a fraction of the limit, falling back
to the utilisation when no power is given), `guillotine` (`{down}`, which keeps the
halves apart once the front end has been killed), `calendar` (`{days: [{slots: [{mine,
others}]}]}`, six 4-hour slots a day, shares averaged over the GPUs) and `hourglass`
(`{wait_s}`); `gpus`, `balance`, `chips`, `watchdog`, `notify`, `nuke`, `cores`,
`gpuwire`, `guillotine`, `calendar` and `hourglass` draw their static scene when no state
is given.

Every page shows at least one figure, and the element remounts when its `name` changes,
so a page can swap it with what is on screen: Results through `ResultsView.FIGURES` (folder `tray`, all plots `film`,
compare `tiles`) and Launch through `LaunchView.FIGURES`, a pattern table over script
keys (for example `infer_*` `shelf`, `tune` `dial`, `measure_receptive_field`
`field`; a generic `train_*` script and an unmatched key show `surface`). Standard pages
carry the figure in a `page__head--figure` header, Launch puts it at the end of its top
bar, and Configuration (`sampler`) and TensorBoard (`landscape`) at the end of their row
headers. The
console shows `terminal` under the job list and `slow` in the main pane while no job
is open, and `HairlineFigure.empty` shows `plug` on a failed load.

Home flanks the centred server name in the deck mast with `landscape` on the left and
`surface` on the right, and carries figures in the middle of its board, each seated in a
card or in a bordered tile of the grid. The three control cards end in a figure above
their buttons: `watchdog` is a camera used as a searchlight, sweeping its beam around
four targets for CPU, load, RAM and disk (each taller the nearer its reading is to the
alert limit); on a target with an active alert the beam stops for 2.6 s, solid and
bright, then sweeps on, and the camera droops while the watchdog is offline. It runs
with `autopilot="off"`, since the sweep is its own motion; `notify` sends an envelope
from the server into a docked phone while notifications are on and goes quiet when they
are off; the kill switches card stacked under it pairs `nuke`, whose plunger drops while
a nuke request is in flight, sending a pulse along the wire and a mushroom cloud up from
the drum, with `guillotine` over kill front end, a server tower whose blade climbs, drops
and splits it in two on a loop and leaves it split once the front end has been killed;
the GPU week card shows `hourglass`, its `wait_s` set to the time left until the
schedule next switches pools, which `GpuWeekPanel` reads from the 168 hourly phases (from
Monday 00:00) and the current hour that `/api/system` reports under `gpu_schedule.week`
and `gpu_schedule.week_hour`, so the sand runs slower the further off the next switch is.
Each CUDA device card sets its utilisation dial, its utilisation and memory plot and a
`gpuwire` figure side by side, the card's fans spinning with utilisation while energy
packets run down its cable at a rate set by the power draw. The CUDA devices grid fills
the empty cell of an odd GPU count with a `gpus` shelf of the real devices; the processor
card ends in `cores`, one pillar per core as tall as it is busy under a dashed plane at
the 50% dispatch line, the pillars above it rimmed bright; the memory card is a two by
two grid, the disk readings beside `riffle` above the memory rule and the RAM and swap
bars over the memory plot beside the `cabinet` rack below it; the GPU activity card puts
`patch` at its top and lets the lanes fill the rest (30 to 80 px per GPU, the panel
taking any slack); the neighbour impact card weighs your own slice's stall against the
worst neighbour's on `chips`, a poker table whose three rakes, one per user with yours
bright, each take more chips from the pot the more that user stalls the machine; the full-width host history puts `phosphor` (with `autopilot="off"`,
so its own frame loop plays) against its left edge and `router` against its right, the
two plots between them; the jobs card shows `scheduler` fed from the live job list
(running jobs in the rack, queued and scheduled ones on the steps); the GPU guard card
shows `guard`, whose barrier lifts during an intrusion; `terminal` is a tile to the left
of the server internals; and the heatmap card carries `calendar` on its right, a desk
calendar of the same days whose pages carry each day's six 4-hour slots as bars
(autopilot off: it turns one page every 2.6 s and rests 6 s on today). Tile and filler figures sit in an absolutely positioned
`.fig-fill` box with `container-type: size` and take `min(100cqw, 125cqh)` as their
width, so they keep the 5:4 box the kernel's pointer mapping assumes and never add height
to their row.
