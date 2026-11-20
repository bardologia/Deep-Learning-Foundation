# Visual harnesses

Headless tools for auditing the console UI without opening a browser: `refresh.sh` and
`snap_ui.js` screenshot the live UI routes, and the check harnesses below load single
frontend modules under plain node.

## UI screenshot harness

Restarts the webui server and captures a full-page PNG of each route. Run it after
every frontend change to verify text overlap, sizing, and that the change is actually
rendered before considering the change done.

```
tools/visual/refresh.sh                      # restart server, snap the default page set
tools/visual/refresh.sh home console         # snap only these routes
PORT=9000 OUT=$PWD/shots tools/visual/refresh.sh
```

Default routes: `home architectures scripts configuration console`.
Parametrised routes work too, e.g. `launch/train_backbone`. Screenshots land in
`logs/visual/ui/<route>.png` (slashes become underscores); the directory is cleared
on each run so every PNG reflects the current code. The server is left running on
the port and its log goes to `logs/visual/webui-<port>.log`.

`snap_ui.js` can also be run directly against an already-running server:

```
node snap_ui.js --port 8772 --out ../../../logs/visual/ui [--settle 1800] [--width 1600] [--height 950] [routes...]
```

It loads the SPA once, then per route sets the hash, waits for the route's page
section to become active, `document.fonts.ready`, and a settle delay (for GSAP
entrances, route API fetches, and MathJax typesetting), and takes a `fullPage`
screenshot. Network-idle waits are deliberately avoided: the status board keeps polling
`/api/system`, `/api/jobs` and the guard history from app boot, so the network idles only
intermittently and never predictably.

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
node snap_experiments.js --port 8772 --out ../../../logs/visual/experiments [--settle 1200] launch/train_backbone launch/cross_validate
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
node literal_split.js fields.json
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

## Gauge harness

`gauge_checks.js` loads `canvas_base.js` and `gauges.js` under plain node with a stub
canvas, a stub animation frame queue, stub timers and a `document.fonts.ready` it releases
itself, runs one named scenario and prints what it observed as JSON. The `fonts` scenario
counts the draws of a dial and a linear meter before and after the fonts settle; `sweep`
feeds a dial readings during and after its ignition sweep and past its trail cap;
`dispose` disposes a dial mid-sweep; `alarm` runs the frame loop with the needle below,
inside and back out of the last alarm zone. `tests/webui/test_gauges.py` drives all four,
which is the contract that keeps the dial text in the web font, the first reading from
being lost to the sweep, and the frame loop from running on once the needle has landed
outside an alarm.

```
node gauge_checks.js fonts
node gauge_checks.js sweep
```

## Neighbour impact harness

`impact_checks.js` loads `home.js` under plain node with a stub DOM, runs one named
scenario against the real `UsersTable` and `StatusBoard` and prints what it observed as
JSON. The `users` scenario renders the active users table over a set of users and the
per-user slice rows of the contention monitor, and reads the header, the column count of
each row and each user's stall cell. The `impact` scenario renders the neighbour impact
card from four monitor payloads (this user blamed, the kernel sources missing, the console
outside the user slice, and someone else's contention) and reads the meters, the share
lines, the stalled count, the verdict, the lede, the light and the state handed to the
figure. The `politeness` scenario renders the card with the monitor's politeness
settings and a throttled alarm, then changes the launch level and flips the
auto-throttle toggle against a stub `Api.post`, and reads the controls, the posted
bodies, the toasts and the lede. `tests/webui/test_home_impact.py` drives all three,
which is the contract that keeps the card reading the monitor's field names and saying
why it cannot measure.

```
node impact_checks.js users
node impact_checks.js impact
node impact_checks.js politeness
```

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
to pick. Override the executable with `PW_CHROMIUM=/path/to/chrome`. Both browser scripts
resolve it the same way.
