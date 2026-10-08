# Visual harnesses

Two headless Playwright tools for auditing the console UI without opening a browser:
`refresh.sh`/`snap_ui.js` screenshot the live UI routes, and `dump_frames.js` renders
the pipeline process animations frame by frame.

## UI screenshot harness

Restarts the webui server and captures a full-page PNG of each route. Run it after
every frontend change to verify text overlap, sizing, and that the change is actually
rendered before considering the change done.

```
tools/visual/refresh.sh                      # restart server, snap the default page set
tools/visual/refresh.sh model pipelines      # snap only these routes
PORT=9000 OUT=/tmp/shots tools/visual/refresh.sh
```

Default routes: `home model pipelines architectures scripts configuration console`.
Parametrised routes work too, e.g. `launch/train_model`. Screenshots land in
`/tmp/ml-template-ui/<route>.png` (slashes become underscores); the directory is cleared
on each run so every PNG reflects the current code. The server is left running on
the port and its log goes to `/tmp/ml-template-webui-<port>.log`.

`snap_ui.js` can also be run directly against an already-running server:

```
node snap_ui.js --port 8772 --out /tmp/ml-template-ui [--settle 1800] [--width 1600] [--height 950] [routes...]
```

It loads the SPA once, then per route sets the hash, waits for the route's page
section to become active, `document.fonts.ready`, and a settle delay (for GSAP
entrances, route API fetches, and KaTeX typesetting), and takes a `fullPage`
screenshot. Network-idle waits are deliberately avoided: the status board keeps polling
`/api/system`, `/api/jobs` and the guard history from app boot, so the network idles only
intermittently and never predictably. `snap_repomap.js` navigates the same way.

## Experiment fan-out harnesses

Two tools for the trial-sweep builder on the launch pages, one with a browser and one
without.

`snap_experiments.js` opens a launch route, switches to its Experiments section, clicks
every mode button in turn, and screenshots each one. It prints the per-mode summary line
and run-name preview as JSON, along with every `pageerror` and console error the page
raised, which is the fastest way to catch a panel that silently fails to build.
Raise `--settle` on a loaded machine when a route reports `experimentsTab: false`; the
launch form has not finished building when the tab is looked for.

```
node snap_experiments.js --port 8772 --out /tmp/ml-template-experiments [--settle 1200] launch/train_jepa launch/train_backbone
```

`trial_names.js` runs the builder script under plain node with no DOM and no server. It
takes the same config-leaf payload the browser receives (`{"key": ..., "leaves": [...]}`,
as `ScriptConfigResolver` produces it) and prints the run names every available mode
previews, keyed by mode. `tests/webui/test_experiment_names.py` drives it to assert those
names byte-match the ones the Python planners plan, which is the contract that keeps the
preview honest about the directories a sweep will create.

```
node trial_names.js /tmp/train_jepa.json
```

## Sequence editor harnesses

Two tools that run the sequence block editor's own modules under plain node, with no
browser and no server.

`literal_split.js` loads `python_literal.js`, takes a JSON array of strings, splits each
one the way the editor splits a decision's candidates field, and prints the candidate
lists that come back. `tests/webui/test_sequence_decision_options.py` drives it over the
values the Python renderer produces, which is the contract that keeps `[32, 32], [64, 32]`
two candidates instead of four.

```
node literal_split.js /tmp/fields.json
```

`sequence_actions.js` loads the editor itself against a stub DOM and a stub router that
answers on the next tick. It clicks start, stop and release three times while the first
request is in flight, then once more after it has landed, and prints the posts each burst
sent with any error the clicks raised. It then opens a block drawer once against a failing
router and three times against a healthy one, and prints how often the script config and
the model list were fetched. `tests/webui/test_sequence_actions.py` drives it to assert
that overlapping clicks post an action once, that the guard lifts afterwards, and that the
drawer keeps a successful reply for the page session but retries a failed one.

```
node sequence_actions.js
```

## Flow sketch harness

`flow_sketch_catalog.js` loads `flow_sketches.js` and `flow_view.js` under plain node with
no DOM and no server, reads the catalogue through `FlowSketches.CATALOG`, the name the
Model walkthrough reads it by, and builds every sketch against a stub SVG. It prints each
sketch's tooltip, the length of the markup it drew and whether it started an animation.
`tests/webui/test_content_libraries.py` drives it, which is the contract that keeps the
walkthrough from failing on a catalogue renamed on one side only.

```
node flow_sketch_catalog.js
```

## Pipeline stage chip harness

`stage_chips.js` loads `process_anim.js` and `pipelines.js` under plain node with a stub
DOM and a stub animator. It takes a JSON map of pipeline key to stage names, binds the
process modal's transport to every animated pipeline, clicks each stage chip in turn and
prints the chapter-to-stage map with the chapter each click sought and the chips it lit.
`tests/webui/test_pipeline_stage_chips.py` drives it over the `PipelineLibrary` stages and
the `ProcessAnimator` chapters, which is the contract that keeps the lit chip moving
forward with the animation and a click landing on the chip that was clicked.

```
node stage_chips.js /tmp/stages.json
```

## Cube view harness

`cube_view_checks.js` loads `cube_view.js` under plain node with no DOM and no server,
runs one named scenario against the real cube page classes with stubbed `fetch` and
ImageBitmaps, and prints what it observed as JSON. The `bitmaps` scenario trims, clears
and releases the slice bitmaps and replaces a cut panel's image, recording which bitmaps
were closed. The `cut` scenario builds a point cloud over a stub canvas and animation
frame queue, then reads the section-cut thresholds, the readout with and without a cut,
the sweep positions over one period, and whether the sweep loop is still running after
each way of leaving it. The `pixel` scenario clicks two parameter-map pixels and answers
the second request before the first, then reads the readout. `tests/webui/test_cube_view_lifecycle.py`
drives all three, which is the contract that keeps evicted and replaced slices from
holding image memory while never closing a bitmap a sweep still draws, keeps the cut sweep
from running on after the cloud is out of sight, and keeps a late answer from overwriting
the pixel picked last.

```
node cube_view_checks.js bitmaps
node cube_view_checks.js cut
node cube_view_checks.js pixel
```

## Repo-map diagram harness

`render_diagram.js` renders a single repo-map schema (one folder/diagram from
`webui/repomap_data.json`, or a bare diagram object) headlessly through the real
`repomap.js` and `styles.css` via `repomap_harness.html`, so a diagram's layout can be
audited or hand-tuned in isolation without the live server. Render at `--width 1520` to
match the real panel width (the point at which cards start clipping off the right edge).

```
node render_diagram.js --json ../../repomap_data.json --folder benchmarking --dkey benchmark-pipeline --out /tmp/rm.png --width 1520 [--labels] [--fit]
```

`--labels` clicks the edge-label toggle before the screenshot. `--fit` drops the panel's
scroll cap so a diagram taller than the viewport is captured whole instead of clipped at
the fold; without it the shot matches what the live panel shows. `--settle <ms>` waits
longer before shooting when a diagram animates in slowly.

`snap_repomap.js` drives the live server instead, selecting the folder tab and diagram
sub-tab before screenshotting the panel:

```
node snap_repomap.js --port 8772 --out /tmp/ml-template-repomap [--labels] backbone:backbone-inference tune:tune-search
```

Targets are `folder:dkey` pairs. `RepoMapLibrary` loads the JSON once at startup, so
restart the server after editing `repomap_data.json` before snapping the live view.

## Animation frame dumper

Renders the pipeline process animations (`webui/static/js/process_anim.js`) headlessly with MathJax loaded and dumps per-timestep PNG frames of the canvas. Used to audit equation placement, text overlap, sizing, and scene pacing without driving the full web UI.

## Setup

```
npm ci
npx playwright-core install chromium
```

`npm ci` installs exactly what `package-lock.json` records, which is the pinned
`playwright-core` from `package.json`. Use it rather than `npm install`, which is free to
resolve a newer release and drift away from the browser build below.

The browser itself is not in the lock file. `npx playwright-core install chromium` fetches
the build that matches the installed `playwright-core` into `~/.cache/ms-playwright`. Note
the binary is `playwright-core`, not `playwright`: this package ships no `playwright` bin,
and calling that name would pull an unrelated release from the registry. The shared
`chromium.js` module picks the highest build number present there, so an older leftover
build never wins; it fails with a message naming the cache directory when there is nothing
to pick. Override the executable with `PW_CHROMIUM=/path/to/chrome`. All five browser scripts
resolve it the same way.

## Usage

```
node dump_frames.js <process_anim.js> <scene-key> <duration-s> <step-s> <out-dir>
```

Example — full sweep of the inference scene at 2 s steps:

```
node dump_frames.js ../../static/js/process_anim.js inference 124 2 /tmp/frames
```

Scene keys and loop durations: `param` 88 · `dataset` 118 · `training` 199 · `inference` 124 · `tuning` 96.

The dumper performs two warm passes first so MathJax equation sprites are rasterized before screenshots, and prints sprite cache stats (`ready/total/failed`). Use a 2 s step for full audits and 0.5 s around transitions.

`mathjax-tex-svg.js` is a vendored copy of MathJax 3.2.2 `tex-svg` so the harness works offline; the live UI loads the same bundle from the jsdelivr CDN in `index.html`.
