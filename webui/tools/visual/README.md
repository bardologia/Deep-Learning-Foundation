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
`/tmp/tomosar-ui/<route>.png` (slashes become underscores); the directory is cleared
on each run so every PNG reflects the current code. The server is left running on
the port and its log goes to `/tmp/tomosar-webui-<port>.log`.

`snap_ui.js` can also be run directly against an already-running server:

```
node snap_ui.js --port 8765 --out /tmp/tomosar-ui [--settle 1800] [--width 1600] [--height 950] [routes...]
```

It loads the SPA once, then per route sets the hash, waits for the route's page
section to become active, `document.fonts.ready`, and a settle delay (for GSAP
entrances, route API fetches, and KaTeX typesetting), and takes a `fullPage`
screenshot. Network-idle waits are deliberately avoided: the status board keeps polling
`/api/system`, `/api/jobs` and the guard history from app boot, so the network idles only
intermittently and never predictably. `snap_repomap.js` navigates the same way.

## Repo-map diagram harness

`render_diagram.js` renders a single repo-map schema (one folder/diagram from
`webui/repomap_data.json`, or a bare diagram object) headlessly through the real
`repomap.js` and `styles.css` via `repomap_harness.html`, so a diagram's layout can be
audited or hand-tuned in isolation without the live server. Render at `--width 1520` to
match the real panel width (the point at which cards start clipping off the right edge).

```
node render_diagram.js --json ../../repomap_data.json --folder benchmarking --dkey benchmark-pipeline --out /tmp/rm.png --width 1520 [--labels]
```

`snap_repomap.js` drives the live server instead, selecting the folder tab and diagram
sub-tab before screenshotting the panel:

```
node snap_repomap.js --port 8765 --out /tmp/tomosar-repomap [--labels] backbone:backbone-inference tune:tune-search
```

Targets are `folder:dkey` pairs. `RepoMapLibrary` loads the JSON once at startup, so
restart the server after editing `repomap_data.json` before snapping the live view.

## Animation frame dumper

Renders the pipeline process animations (`webui/static/js/process_anim.js`) headlessly with MathJax loaded and dumps per-timestep PNG frames of the canvas. Used to audit equation placement, text overlap, sizing, and scene pacing without driving the full web UI.

## Setup

```
npm install
```

Requires a Playwright Chromium under `~/.cache/ms-playwright` (any `chromium*` build). Override the executable with `PW_CHROMIUM=/path/to/chrome`. All four scripts locate it through the shared `chromium.js` module.

## Usage

```
node dump_frames.js <process_anim.js> <scene-key> <duration-s> <step-s> <out-dir>
```

Example — full sweep of the inference scene at 2 s steps:

```
node dump_frames.js ../../static/js/process_anim.js inference 124 2 /tmp/frames
```

Scene keys and loop durations: `param` 88 · `processing` 122 · `dataset` 118 · `training` 210 · `inference` 124 · `tuning` 96.

The dumper performs two warm passes first so MathJax equation sprites are rasterized before screenshots, and prints sprite cache stats (`ready/total/failed`). Use a 2 s step for full audits and 0.5 s around transitions.

`mathjax-tex-svg.js` is a vendored copy of MathJax 3.2.2 `tex-svg` so the harness works offline; the live UI loads the same bundle from the jsdelivr CDN in `index.html`.
