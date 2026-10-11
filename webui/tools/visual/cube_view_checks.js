const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class CubeViewChecks {
  constructor(staticDir) {
    this.sandbox = { window: {}, console, setTimeout, clearTimeout };
    this.sandbox.globalThis = this.sandbox;
    vm.createContext(this.sandbox);

    const source = fs.readFileSync(path.join(staticDir, "cube_view.js"), "utf8");
    vm.runInContext(`${source}\nglobalThis.classes = { CubeSweep, CubeParams, CubeMetrics, CubeTransect, CubeCloud, CubeVolume, CubeTimeline, CubeFigures, CubeView };`, this.sandbox, { filename: "cube_view.js" });
    this.classes = this.sandbox.classes;
  }

  _bitmap() {
    const bitmap = { closed: false };
    bitmap.close = () => { bitmap.closed = true; };
    return bitmap;
  }

  _settle() {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  async bitmaps() {
    const { CubeSweep, CubeView } = this.classes;
    const cap = CubeView.SWEEP_CACHE_BYTES;

    const shown   = this._bitmap();
    const evicted = this._bitmap();
    const kept    = this._bitmap();

    const sweep  = Object.create(CubeSweep.prototype);
    sweep.panels = [{ bitmap: shown }];

    const view       = Object.create(CubeView.prototype);
    view.sweeps      = [sweep];
    view.bitmapBytes = 2 * cap + 1;
    view.bitmapCache = new Map([
      ["shown", { promise: Promise.resolve(shown), bytes: cap }],
      ["evicted", { promise: Promise.resolve(evicted), bytes: cap }],
      ["kept", { promise: Promise.resolve(kept), bytes: 1 }],
    ]);

    view._trimBitmapCache();
    await this._settle();
    const trim = { shown_closed: shown.closed, evicted_closed: evicted.closed, kept_closed: kept.closed, bytes: view.bitmapBytes, keys: [...view.bitmapCache.keys()] };

    view._clearBitmapCache();
    await this._settle();
    const clear = { kept_closed: kept.closed, size: view.bitmapCache.size, bytes: view.bitmapBytes };

    const released = this._bitmap();
    const panel    = { bitmap: released, key: "3:4", drawnSpace: "physical" };
    view._releasePanel(panel);
    const release = { closed: released.closed, bitmap: panel.bitmap, key: panel.key, space: panel.drawnSpace };

    const stale = this._bitmap();
    const fresh = this._bitmap();

    this.sandbox.fetch             = async () => ({ ok: true, blob: async () => ({}) });
    this.sandbox.createImageBitmap = async () => fresh;

    view.space       = "physical";
    view.selectedId  = "cube";
    view.cmap        = "jet";
    view._paintSlice = () => {};

    const slicePanel = { bitmap: stale, source: "pred", axis: "row", root: { classList: { add: () => {}, remove: () => {} } } };
    await view._fetchSlice(slicePanel, { row: 1, col: 2, key: "1:2" });
    const replace = { stale_closed: stale.closed, fresh_closed: fresh.closed, holds_fresh: slicePanel.bitmap === fresh, key: slicePanel.key };

    return { trim, clear, release, replace };
  }

  _element(extra = {}) {
    const element = { value: "", checked: false, disabled: false, hidden: false, title: "", textContent: "", style: {}, listeners: {}, ...extra };
    element.addEventListener = (type, handler) => { element.listeners[type] = handler; };
    element.querySelectorAll = () => [];
    element.classList        = { toggle: () => {}, add: () => {}, remove: () => {} };
    return element;
  }

  _frames() {
    const frames = { pending: new Map(), cancelled: [], next: 0 };

    this.sandbox.requestAnimationFrame = (handler) => {
      frames.next += 1;
      frames.pending.set(frames.next, handler);
      return frames.next;
    };
    this.sandbox.cancelAnimationFrame = (id) => {
      frames.cancelled.push(id);
      frames.pending.delete(id);
    };
    frames.fire = (now) => {
      const [id, handler] = [...frames.pending.entries()][0];
      frames.pending.delete(id);
      handler(now);
    };
    return frames;
  }

  _cloud() {
    const { CubeCloud } = this.classes;
    const fills = { count: 0 };

    const ctx = {
      createImageData : (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData    : () => {},
      beginPath       : () => {},
      moveTo          : () => {},
      lineTo          : () => {},
      closePath       : () => {},
      fill            : () => { fills.count += 1; },
      stroke          : () => {},
    };

    const host = {
      view           : "cloud",
      selectedId     : "cube",
      meta           : { n_rows: 10, n_cols: 20, sources: ["pred"], aux: false, metric_maps: [], params: { sources: ["pred"], threshold: 0.1, ranges: { amp: [0, 1] } } },
      _fmt           : (value) => String(value),
      hasMetricLayer : (key) => host.meta.metric_maps.some((layer) => layer.key === key),
    };

    const refs = {
      source   : this._element(),
      color    : this._element(),
      thr      : this._element({ value: "0" }),
      thrLabel : this._element(),
      thrVal   : this._element(),
      max      : this._element({ value: "60000" }),
      cutAxis  : this._element({ value: "off" }),
      cutPos   : this._element({ value: "50" }),
      cutSweep : this._element(),
      auxWrap  : this._element(),
      aux      : this._element(),
      at       : this._element(),
      canvas   : this._element({ width: 0, height: 0, parentElement: { clientWidth: 400 }, getContext: () => ctx }),
    };

    const cloud    = new CubeCloud(refs, host);
    cloud.points   = new Float32Array([0, 0, -5, 1, NaN, 5, 5, 5, 1, NaN, 9, 19, 15, 1, NaN]);
    cloud.width    = 5;
    cloud.total    = 3;
    cloud.muRange  = [-5, 15];
    cloud.muBounds = CubeCloud.valueBounds(cloud.points, 2, 5);
    return { cloud, refs, host, fills, ctx };
  }

  async cut() {
    const { CubeCloud, CubeView } = this.classes;
    Object.assign(this.sandbox.window, { innerHeight: 600, devicePixelRatio: 1, addEventListener: () => {} });

    const frames = this._frames();
    const bounds = { mixed: CubeCloud.valueBounds(new Float32Array([1, NaN, 3, -2]), 0, 1), empty: CubeCloud.valueBounds(new Float32Array([NaN]), 0, 1) };

    const { cloud, refs, host, fills } = this._cloud();
    const specs = {};
    [["off", "50"], ["row", "50"], ["col", "25"], ["z", "50"], ["z", "0"]].forEach(([axis, pos]) => {
      refs.cutAxis.value = axis;
      refs.cutPos.value  = pos;
      specs[`${axis}@${pos}`] = cloud._cutSpec();
    });

    refs.cutAxis.value = "row";
    refs.cutPos.value  = "50";
    cloud.muBounds     = null;
    const unbounded    = cloud._cutSpec();
    cloud.muBounds     = CubeCloud.valueBounds(cloud.points, 2, 5);

    cloud._draw();
    const cutDraw = { text: refs.at.textContent, fills: fills.count };

    refs.cutAxis.value = "off";
    cloud._draw();
    const openDraw = { text: refs.at.textContent, fills: fills.count };

    refs.cutSweep.checked = true;
    refs.cutSweep.listeners.change();
    const started = { axis: refs.cutAxis.value, running: cloud.sweepFrame !== null };

    const positions = [0, 2250, 6750].map((now) => {
      frames.fire(now);
      return refs.cutPos.value;
    });

    cloud.leave();
    const left = { running: cloud.sweepFrame !== null, pending: frames.pending.size };

    const view        = Object.create(CubeView.prototype);
    view.sweeps       = [];
    view.cloud        = cloud;
    view.timeline     = this._timelineStub();
    view.storyTimer   = null;
    view.storyPlayBtn = this._element();
    cloud._startSweep();
    view.leave();
    const viewLeft = { running: cloud.sweepFrame !== null, pending: frames.pending.size };

    cloud._startSweep();
    host.view = "explorer";
    frames.fire(0);
    const offView = { running: cloud.sweepFrame !== null, pending: frames.pending.size };

    host.view = "cloud";
    cloud._startSweep();
    cloud.configure(host.meta);
    const reconfigured = { running: cloud.sweepFrame !== null, muBounds: cloud.muBounds };

    refs.cutSweep.checked = true;
    cloud._startSweep();
    refs.cutAxis.value = "off";
    refs.cutAxis.listeners.change();
    const axisOff = { running: cloud.sweepFrame !== null, sweep: refs.cutSweep.checked };

    return { bounds, specs, unbounded, cutDraw, openDraw, started, positions, left, viewLeft, offView, reconfigured, axisOff };
  }

  async uncertainty() {
    const { CubeCloud } = this.classes;
    Object.assign(this.sandbox.window, { innerHeight: 600, devicePixelRatio: 1, addEventListener: () => {} });

    const ranges = {
      mixed : CubeCloud.finiteRange(new Float32Array([0, NaN, 0, 0.5, 0, 0.25]), 1, 2),
      empty : CubeCloud.finiteRange(new Float32Array([0, NaN, 0, NaN]), 1, 2),
    };

    const { cloud, refs, host, ctx } = this._cloud();
    const buttons = ["mu", "amp", "std"].map((color) => ({ dataset: { color }, disabled: false, classList: { toggle: () => {} } }));
    refs.color.querySelectorAll = () => buttons;

    const painted = [];
    ctx.putImageData = (image) => painted.push(image.data);
    const greys = () => {
      const data = painted[painted.length - 1];
      let count  = 0;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] === 128 && data[i + 1] === 128 && data[i + 2] === 128 && data[i + 3] === 255) count += 1;
      }
      return count;
    };

    cloud.colorBy = "std";
    cloud._syncBtns();
    const absent = { colorBy: cloud.colorBy, disabled: buttons[2].disabled };

    host.meta.metric_maps = [{ key: "seed_std_profile" }];
    cloud.colorBy = "std";
    cloud._syncBtns();
    const present = { colorBy: cloud.colorBy, disabled: buttons[2].disabled };

    cloud.points   = new Float32Array([0, 0, -5, 1, 0.5, 5, 5, 5, 1, NaN, 9, 19, 15, 1, 2.5]);
    cloud.stdRange = CubeCloud.finiteRange(cloud.points, 4, 5);

    const ts      = [];
    const palette = cloud._palette.bind(cloud);
    cloud._palette = (t) => {
      ts.push(t);
      return palette(t);
    };

    cloud._draw();
    const byStd = { ts: ts.splice(0), greys: greys(), text: refs.at.textContent };

    cloud.colorBy = "mu";
    cloud._draw();
    const byZ = { ts: ts.splice(0), greys: greys() };

    const blob = new Float32Array([2, 7, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    this.sandbox.fetch = async () => ({ ok: true, arrayBuffer: async () => blob.buffer });
    cloud._draw     = () => {};
    cloud.available = true;
    await cloud._fetch();
    const fetched = { width: cloud.width, total: cloud.total, muBounds: cloud.muBounds, stdRange: cloud.stdRange };

    return { ranges, absent, present, byStd, byZ, fetched };
  }

  async cloudRace() {
    Object.assign(this.sandbox.window, { innerHeight: 600, devicePixelRatio: 1, addEventListener: () => {} });

    const { cloud, host } = this._cloud();
    const answers = [];
    const blob    = (total) => ({ ok: true, arrayBuffer: async () => new Float32Array([1, total, 5, 0, 0, 0, total, 1, 0]).buffer });

    this.sandbox.fetch = () => new Promise((resolve) => answers.push(resolve));
    cloud._draw        = () => {};
    cloud.available    = true;

    const earlier = cloud._fetch();
    const later   = cloud._fetch();
    answers[1](blob(2));
    await later;
    answers[0](blob(1));
    await earlier;
    const raced = { total: cloud.total, mu: cloud.points[2] };

    const stale = cloud._fetch();
    cloud.configure(host.meta);
    answers[2](blob(3));
    await stale;
    const reconfigured = { points: cloud.points, total: cloud.total };

    return { raced, reconfigured };
  }

  cutColors() {
    const { CubeView } = this.classes;
    const strokes = [];
    const ctx     = { drawImage: () => {}, setLineDash: () => {}, beginPath: () => {}, moveTo: () => {}, lineTo: () => {}, stroke: () => strokes.push(ctx.strokeStyle) };

    this.sandbox.getComputedStyle = () => ({ getPropertyValue: (name) => ` ${name} ` });

    const view  = Object.create(CubeView.prototype);
    view.stage  = {};
    view.colors = view._readColors();

    const markers = {};
    ["col", "row"].forEach((axis) => {
      const panel = { axis, marker: 0.5, bitmap: { width: 4, height: 4 }, canvas: { width: 4, height: 4, getContext: () => ctx } };
      view._paintSlice(panel);
      markers[axis] = strokes.splice(0);
    });

    return { markers };
  }

  async coverage() {
    const timers = { next: 0, live: new Map(), cleared: [] };
    this.sandbox.setInterval   = (handler) => { timers.next += 1; timers.live.set(timers.next, handler); return timers.next; };
    this.sandbox.clearInterval = (id) => { timers.cleared.push(id); timers.live.delete(id); };

    const answers = [];
    const { metrics, refs, host } = this._metrics((query) => new Promise((resolve) => answers.push({ query, resolve })));
    const fetched = [];
    metrics.layer       = { key: "pixel_mse", label: "MSE", vmin: 0, vmax: 1, classes: null };
    metrics._onCoverage = () => fetched.push(refs.selCov.value);
    metrics._refresh    = () => {};
    host.view           = "metrics";
    refs.selCov.value   = "15";

    refs.selPlay.listeners.click();
    const started = { mask: refs.selMask.checked, label: refs.selPlay.textContent, running: timers.live.size };

    const tick = () => [...timers.live.values()][0]();
    [1, 2, 3, 4].forEach(tick);
    const swept = { values: fetched.splice(0), dir: metrics.selPlayDir };

    const low = metrics._fetchSelective(0.4);
    answers.shift().resolve({ ok: true, direction: "low", threshold: 0.2, coverage: 0.4, rows: [], n_kept: 4, n_total: 10 });
    await low;
    const lowWindow = metrics._keepWindow();

    const high = metrics._fetchSelective(0.4);
    answers.shift().resolve({ ok: true, direction: "high", threshold: 0.7, coverage: 0.4, rows: [], n_kept: 4, n_total: 10 });
    await high;
    const highWindow = metrics._keepWindow();

    host.view = "explorer";
    tick();
    const leftView = { running: timers.live.size, label: refs.selPlay.textContent };

    host.view = "metrics";
    refs.selPlay.listeners.click();
    metrics.leave();
    const left = { running: timers.live.size, label: refs.selPlay.textContent };

    refs.selPlay.listeners.click();
    refs.selMask.checked = false;
    refs.selMask.listeners.change();
    const unmasked = { running: timers.live.size, window: metrics.maskWindow, keep: metrics._keepWindow() };

    const ignored = metrics._fetchSelective(0.4);
    answers.shift().resolve({ ok: true, direction: "low", threshold: 0.2, coverage: 0.4, rows: [], n_kept: 4, n_total: 10 });
    await ignored;
    const unmaskedAnswer = metrics.maskWindow;

    refs.selMask.checked = true;
    metrics.maskWindow   = { keep_min: "-inf", keep_max: 0.2 };
    refs.selPlay.listeners.click();
    metrics.configure({ metric_maps: [] });
    const reconfigured = { running: timers.live.size, mask: refs.selMask.checked, window: metrics.maskWindow };

    return { started, swept, lowWindow, highWindow, leftView, left, unmasked, unmaskedAnswer, reconfigured };
  }

  async profileStd() {
    const { CubeView } = this.classes;
    const pending      = [];
    const calls        = [];

    this.sandbox.Api = {
      get: (url) => {
        calls.push(url);
        return new Promise((resolve) => pending.push(resolve));
      },
    };

    const view = Object.create(CubeView.prototype);
    Object.assign(view, {
      selectedId : "cube",
      meta       : { n_rows: 4, n_cols: 6, metric_maps: [{ key: "seed_std_profile" }] },
      profData   : { row: 1, col: 2 },
      profAt     : this._element({ textContent: "profiles at row = 1 · col = 2" }),
    });

    const shown = view._fetchProfStd(1, 2);
    pending.shift()({ ok: true, row: 1, col: 2, key: "seed_std_profile", value: 0.125 });
    await shown;
    const value = view.profAt.textContent;

    const stale = view._fetchProfStd(1, 2);
    view.profData = { row: 3, col: 3 };
    view.profAt.textContent = "profiles at row = 3 · col = 3";
    pending.shift()({ ok: true, row: 1, col: 2, key: "seed_std_profile", value: 0.5 });
    await stale;
    const moved = view.profAt.textContent;

    view.profData = { row: 3, col: 3 };
    const empty = view._fetchProfStd(3, 3);
    pending.shift()({ ok: true, row: 3, col: 3, key: "seed_std_profile", value: null });
    await empty;
    const missing = view.profAt.textContent;

    const layers = { present: view.hasMetricLayer("seed_std_profile"), absent: view.hasMetricLayer("pixel_mse") };
    view.meta    = null;
    layers.unloaded = view.hasMetricLayer("seed_std_profile");

    return { calls, value, moved, missing, layers };
  }

  async sweepUrls() {
    const { CubeSweep } = this.classes;
    const host          = { selectedId: "stamp/a", space: "physical", cmap: "jet" };

    const urls = {};
    ["z", "row", "col"].forEach((axis) => {
      const sweep = new CubeSweep({ axis }, host);
      sweep.steps = 5;
      urls[axis]  = sweep._url("pred", 3);
    });
    return urls;
  }

  _timeline() {
    const { CubeTimeline } = this.classes;
    const fetched          = [];
    const drawn            = [];
    const answers          = [];

    this.sandbox.Api = { get: (url) => new Promise((resolve) => answers.push({ url, resolve })) };

    const ctx = { drawImage: (bitmap) => drawn.push(bitmap.url), clearRect: () => drawn.push("cleared") };
    const bitmaps = { pending: [], missing: new Set() };

    const host = {
      selectedId  : "stamp/b",
      meta        : { n_rows: 8, n_cols: 6, sources: ["pred", "gt"] },
      view        : "timeline",
      space       : "physical",
      cmap        : "jet",
      focus       : null,
      focusPoint  : () => host.focus,
      _fmt        : (value) => String(value),
      cacheBitmap : (url) => {
        fetched.push(url);
        return Promise.resolve(bitmaps.missing.has(url) ? null : { url, width: 6, height: 5 });
      },
    };

    const buttons = (attr, values) => values.map((value) => ({ dataset: { [attr]: value }, disabled: false, classList: { toggle: () => {} }, addEventListener: () => {} }));
    const refs    = {
      source : this._element(),
      axis   : this._element(),
      play   : this._element({ textContent: "Play" }),
      pos    : this._element({ value: "0", max: "0" }),
      at     : this._element(),
      canvas : this._element({ width: 0, height: 0, getContext: () => ctx }),
      label  : this._element(),
    };
    refs.play.classList = { add: () => {}, remove: () => {}, toggle: () => {} };

    const sourceBtns = buttons("source", ["pred", "gt", "full"]);
    const axisBtns   = buttons("axis", ["col", "row"]);
    refs.source.querySelectorAll = () => sourceBtns;
    refs.axis.querySelectorAll   = () => axisBtns;

    const timeline = new CubeTimeline(refs, host);
    const stamps   = [{ id: "/runs/a/inference/stamp_1", stamp: "stamp_1", mse: null }, { id: "/runs/a/inference/stamp_2", stamp: "stamp_2", mse: 0.125 }, { id: "/runs/a/inference/stamp_3", stamp: "stamp_3", mse: 0.5 }];
    return { timeline, host, refs, sourceBtns, fetched, drawn, answers, bitmaps, stamps };
  }

  async timeline() {
    const { CubeTimeline } = this.classes;
    const { timeline, host, refs, sourceBtns, fetched, drawn, answers, bitmaps, stamps } = this._timeline();
    const query = (url) => Object.fromEntries(new URLSearchParams(url.split("?")[1]));

    timeline.configure(host.meta);
    const rendering = timeline.render();
    answers.shift().resolve({ ok: true, current: 1, stamps });
    await rendering;
    const first = {
      asked    : answers.length,
      idx      : timeline.idx,
      max      : refs.pos.max,
      pos      : refs.pos.value,
      at       : refs.at.textContent,
      label    : refs.label.textContent,
      urls     : fetched.splice(0).map(query),
      drawn    : drawn.splice(0),
      disabled : sourceBtns.map((btn) => btn.disabled),
      canvas   : [refs.canvas.width, refs.canvas.height],
    };

    await timeline.render();
    const cached = { asked: answers.length, urls: fetched.splice(0).length, drawn: drawn.splice(0).length };

    timeline._setSource("full");
    const refused = { source: timeline.source, urls: fetched.splice(0).length };

    host.focus = { row: 2, col: 3 };
    timeline._setAxis("row");
    await this._settle();
    const row = { at: refs.at.textContent, url: query(fetched.splice(0).pop()), drawn: drawn.splice(0).length };

    refs.pos.value = "0";
    timeline.playing = true;
    refs.pos.listeners.input();
    await this._settle();
    const scrubbed = { idx: timeline.idx, playing: timeline.playing, label: refs.label.textContent, prefetched: fetched.splice(0).map((url) => query(url).id), drawn: drawn.splice(0).length };

    timeline._setSource("gt");
    bitmaps.missing.add(timeline._url(0));
    await timeline._show();
    const missing = { at: refs.at.textContent, drawn: drawn.splice(0) };
    bitmaps.missing.clear();

    const slow   = { resolve: null };
    host.cacheBitmap = (url) => new Promise((resolve) => { if (!slow.resolve) slow.resolve = () => resolve({ url: "old", width: 6, height: 5 }); else resolve({ url: "new", width: 6, height: 5 }); });
    const older  = timeline._show();
    const newer  = timeline._show();
    await newer;
    slow.resolve();
    await older;
    const stale = drawn.splice(0);

    host.cacheBitmap = (url) => Promise.resolve({ url, width: 6, height: 5 });
    CubeTimeline.FRAME_MS = 0;
    timeline.idx = 0;
    timeline._togglePlay();
    const started = { playing: timeline.playing, label: refs.play.textContent };
    for (let i = 0; i < 6; i += 1) await new Promise((resolve) => setTimeout(resolve, 1));
    const advanced = timeline.idx > 0 || drawn.length > 1;
    host.view = "explorer";
    for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setTimeout(resolve, 1));
    const stopped = { playing: timeline.playing, label: refs.play.textContent };
    host.view = "timeline";

    timeline.configure({ sources: ["pred"] });
    const late = timeline.render();
    host.selectedId = "stamp/c";
    answers.shift().resolve({ ok: true, current: 0, stamps });
    await late;
    const dropped = { stamps: timeline.stamps.length, loadedFor: timeline.loadedFor };

    const failing = timeline.render();
    answers.shift().resolve({ ok: false, error: "unknown cube id: stamp/c" });
    await failing;
    const failed = refs.at.textContent;

    const single = timeline.render();
    answers.shift().resolve({ ok: true, current: 0, stamps: stamps.slice(0, 1) });
    await single;
    timeline._togglePlay();
    const alone = { at: refs.at.textContent, playing: timeline.playing };

    return { first, cached, refused, row, scrubbed, missing, stale, started, advanced, stopped, dropped, failed, alone };
  }

  timelineViews() {
    const calls = [];
    const stub  = this._displayStub("timeline", { timeline: this._timelineStub(calls) });
    stub._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4 });
    const loaded = calls.splice(0);

    stub.meta    = { sources: ["pred"] };
    stub.viewEls = [];
    stub._setView("explorer");
    const left = calls.splice(0);

    stub._setView("timeline");
    const opened = calls.splice(0);

    stub.cmap        = "jet";
    stub.panels      = [];
    stub.point       = null;
    stub.transect    = null;
    this.sandbox.localStorage = { setItem: () => {} };
    stub._setCmap("viridis");
    const recoloured = calls.splice(0);

    stub.leave();
    const tab = calls.splice(0);

    return { loaded, left, opened, recoloured, tab };
  }

  _linkedView() {
    const { CubeView } = this.classes;
    const toasts       = [];
    const calls        = [];

    this.sandbox.Toast           = { show: (text, kind) => toasts.push({ text, kind }) };
    this.sandbox.URLSearchParams = URLSearchParams;

    const modeBtn = (view, label, hidden) => ({ dataset: { view }, textContent: ` ${label} `, hidden, classList: { toggle: () => {} } });
    const view    = Object.create(CubeView.prototype);
    Object.assign(view, {
      cubes      : [{ id: "/runs/a/inference/stamp_1" }, { id: "/runs/a/inference/stamp_2" }],
      selectedId : "/runs/a/inference/stamp_1",
      meta       : { n_rows: 8, n_cols: 6, sources: ["pred"], metric_maps: [] },
      view       : "explorer",
      space      : "physical",
      cmap       : "jet",
      locked     : null,
      focus      : null,
      polling    : false,
      metrics    : { layer: { key: "pixel_mse" }, layers: [{ key: "pixel_mse" }, { key: "pixel_r2" }], _setLayer: (key) => calls.push(`layer ${key}`) },
      modeBtns   : [modeBtn("explorer", "Cuts", false), modeBtn("metrics", "Metric maps", false), modeBtn("params", "Parameters", true)],
      cmapSel    : { value: "jet", options: [{ value: "jet" }, { value: "viridis" }] },
    });
    view.select       = (id) => calls.push(`select ${id}`);
    view._setCmap     = (cmap) => { view.cmap = cmap; calls.push(`cmap ${cmap}`); };
    view._setSpace    = (space) => { view.space = space; calls.push(`space ${space}`); };
    view._setView     = (name) => { view.view = name; calls.push(`view ${name}`); };
    view._follow      = (point) => calls.push(`follow ${point.row},${point.col},${point.fx},${point.fy}`);
    view._enterSlices = (point) => { view.locked = { row: point.row, col: point.col }; calls.push(`lock ${point.row},${point.col}`); };
    return { view, toasts, calls };
  }

  async deepLink() {
    const { view, toasts, calls } = this._linkedView();
    const take = () => ({ calls: calls.splice(0), toasts: toasts.splice(0).map((toast) => toast.text) });

    view.focus  = { row: 2, col: 3 };
    const plain = view.stateParam();
    view.locked = { row: 5, col: 1 };
    view.view   = "metrics";
    const param = view.stateParam();
    const query = Object.fromEntries(new URLSearchParams(decodeURIComponent(param)));
    const parsed = view._parseState(param);

    const refused = { noId: view._parseState(encodeURIComponent("view=metrics")), empty: view._parseState(null), broken: view._parseState("%E0%A4%A"), toasts: take().toasts };

    view.view   = "explorer";
    view.locked = null;
    view.pendingState = view._parseState(encodeURIComponent("id=/runs/a/inference/stamp_1&view=metrics&row=99&col=-4&layer=pixel_r2&space=normalized&cmap=viridis"));
    view._applyPendingState();
    const applied = { ...take(), pending: view.pendingState };

    view.pendingState = view._parseState(encodeURIComponent("id=/runs/a/inference/stamp_1&view=params&cmap=plasma&layer=banana"));
    view._applyPendingState();
    const hidden = take();

    view.pendingState = view._parseState(encodeURIComponent("id=/runs/a/inference/stamp_1&view=sideways"));
    view._applyPendingState();
    const unknown = take();

    view.pendingState = view._parseState(encodeURIComponent("id=/runs/a/inference/stamp_2&view=metrics"));
    view.polling      = true;
    view._applyPendingState();
    const waiting = { ...take(), pending: !!view.pendingState };

    view.polling = false;
    view._applyPendingState();
    const selected = { ...take(), pending: !!view.pendingState };

    view.selectedId = "/runs/a/inference/stamp_2";
    view.meta       = null;
    view._applyPendingState();
    const failed = { ...take(), pending: view.pendingState };

    view.meta         = { n_rows: 8, n_cols: 6, sources: ["pred"], metric_maps: [] };
    view.pendingState = view._parseState(encodeURIComponent("id=/runs/b/inference/stamp_9"));
    view._applyPendingState();
    const foreign = { ...take(), pending: view.pendingState };

    view.entered       = true;
    view._refreshStrip = async () => calls.push("strip");
    await view.enter(encodeURIComponent("id=/runs/a/inference/stamp_2&row=1&col=1"));
    const reentered = take();

    this.sandbox.Api  = { get: async () => ({ id: "/runs/a/inference/stamp_2", state: "ready", cube: { n_rows: 8, n_cols: 6 } }) };
    view._setProgress = () => {};
    view._display     = () => calls.push("display");
    view.pendingState = view._parseState(encodeURIComponent("id=/runs/a/inference/stamp_2&row=3&col=4"));
    await view._poll();
    const polled = { ...take(), polling: view.polling, pending: view.pendingState };

    return { plain, query, parsed, refused, applied, hidden, unknown, waiting, selected, failed, foreign, reentered, polled };
  }

  _storyView() {
    const { CubeView } = this.classes;
    const { view, toasts, calls } = this._linkedView();
    const saved   = [];
    const copied  = [];
    const timers  = { next: 0, live: new Map() };

    const node = (tag) => {
      const el = { tag, children: [], className: "", textContent: "", listeners: {} };
      el.appendChild      = (child) => el.children.push(child);
      el.addEventListener = (type, handler) => { el.listeners[type] = handler; };
      return el;
    };

    this.sandbox.localStorage  = this._storage();
    this.sandbox.document      = { createElement: (tag) => node(tag) };
    this.sandbox.location      = { origin: "http://host:8772", pathname: "/" };
    this.sandbox.navigator     = { clipboard: { writeText: (url) => { copied.push(url); return Promise.resolve(); } } };
    this.sandbox.FigureExport  = { download: (name, source) => saved.push({ name, source }), slug: (text) => String(text).replace(/[^A-Za-z0-9._-]+/g, "_") };
    this.sandbox.setTimeout    = (handler) => { timers.next += 1; timers.live.set(timers.next, handler); return timers.next; };
    this.sandbox.clearTimeout  = (id) => timers.live.delete(id);

    const list = node("ol");
    Object.defineProperty(list, "innerHTML", { set: () => { list.children = []; } });

    const snap = { canvas: null };
    const active = { hidden: false, querySelectorAll: (selector) => (selector === "canvas" && snap.canvas ? [snap.canvas] : []) };

    Object.assign(view, {
      storySnapshots : {},
      storyTimer     : null,
      storyPanel     : this._element({ hidden: true }),
      storyName      : this._element({ value: "" }),
      storyPlayBtn   : this._element({ textContent: "Play" }),
      storyHint      : this._element(),
      storyList      : list,
      viewEls        : [{ hidden: true, querySelectorAll: () => [] }, active],
      sweeps         : [],
      timeline       : this._timelineStub(),
    });
    return { view, toasts, calls, saved, copied, timers, list, snap, CubeView };
  }

  async story() {
    const { view, toasts, calls, saved, copied, timers, list, snap, CubeView } = this._storyView();
    const steps = () => JSON.parse(this.sandbox.localStorage.getItem(CubeView.STORY_KEY) || "[]");
    const rows  = () => list.children.map((item) => [item.children[0].textContent, ...item.children.slice(1).map((btn) => btn.textContent)]);

    view._storyTogglePlay();
    const emptyPlay = { toasts: toasts.splice(0).map((toast) => toast.text), timers: timers.live.size };

    view._toggleStoryPanel();
    const opened = { hidden: view.storyPanel.hidden, hint: view.storyHint.textContent };

    snap.canvas = { width: 40, offsetParent: {}, toDataURL: () => "data:image/png;base64,AAA" };
    view.focus  = { row: 2, col: 3 };
    view.storyName.value = "  overview  ";
    view._storyAdd();
    snap.canvas = null;
    view.locked = { row: 5, col: 1 };
    view._storyAdd();
    const added = { steps: steps().map((step) => ({ name: step.name, param: decodeURIComponent(step.param) })), snapshots: Object.keys(view.storySnapshots).length, name: view.storyName.value, rows: rows(), hint: view.storyHint.textContent };
    toasts.splice(0);

    view._storyExport();
    const exported = saved.splice(0);

    list.children[1].children[2].listeners.click();
    await Promise.resolve();
    const linked = copied.splice(0);

    calls.splice(0);
    list.children[0].children[1].listeners.click();
    const went = calls.splice(0);

    view._storyTogglePlay();
    const playing = { label: view.storyPlayBtn.textContent, timers: timers.live.size };
    [...timers.live.entries()].forEach(([id, handler]) => { timers.live.delete(id); handler(); });
    const first = calls.splice(0);
    view.leave = view.leave.bind(view);
    view.metrics.leave = () => {};
    view.leave();
    const stopped = { label: view.storyPlayBtn.textContent, timer: view.storyTimer, timers: timers.live.size };

    view._storyTogglePlay();
    while (timers.live.size) {
      const [id, handler] = [...timers.live.entries()][0];
      timers.live.delete(id);
      handler();
    }
    const replayed = { calls: calls.splice(0).filter((call) => call.startsWith("lock")), label: view.storyPlayBtn.textContent };

    list.children[0].children[3].listeners.click();
    const removed = { names: steps().map((step) => step.name), snapshots: Object.keys(view.storySnapshots).length, rows: rows().length };

    toasts.splice(0);
    view._storyExport();
    const none = { saved: saved.splice(0).length, toast: toasts.splice(0).map((toast) => toast.kind) };

    view._storyClear();
    const cleared = { steps: steps().length, rows: rows().length, hint: view.storyHint.textContent };

    view.meta = null;
    view._storyAdd();
    view._copyLink();
    const unloaded = { toasts: toasts.splice(0).map((toast) => toast.text), steps: steps().length };

    view.meta              = { n_rows: 8, n_cols: 6, sources: ["pred"], metric_maps: [] };
    this.sandbox.navigator = {};
    view._copyLink();
    const insecure = toasts.splice(0).map((toast) => toast.text);

    const drawnImages = [];
    this.sandbox.document = {
      createElement: () => ({ width: 0, height: 0, getContext: () => ({ drawImage: (img) => drawnImages.push(img.id) }), toDataURL () { return `copy ${this.width}x${this.height}`; } }),
    };
    const pick = (canvases, imgs) => {
      view.viewEls = [{ hidden: false, querySelectorAll: (selector) => (selector === "canvas" ? canvases : imgs) }];
      return view._snapshotDataUrl();
    };
    const shownCanvas  = { width: 10, offsetParent: {}, toDataURL: () => "canvas" };
    const hiddenCanvas = { width: 10, offsetParent: null, toDataURL: () => "hidden" };
    const blankCanvas  = { width: 0, offsetParent: {}, toDataURL: () => "blank" };
    const shownImg     = { id: "map", naturalWidth: 6, naturalHeight: 8, offsetParent: {} };
    const hiddenImg    = { id: "off", naturalWidth: 6, naturalHeight: 8, offsetParent: null };
    const snapshots    = {
      canvas : pick([hiddenCanvas, blankCanvas, shownCanvas], [shownImg]),
      img    : pick([hiddenCanvas, blankCanvas], [hiddenImg, shownImg]),
      none   : pick([hiddenCanvas], [hiddenImg]),
      drawn  : drawnImages,
    };

    return { emptyPlay, opened, added, exported, linked, went, playing, first, stopped, replayed, removed, none, cleared, unloaded, insecure, snapshots };
  }

  async pixel() {
    const { CubeParams } = this.classes;
    const pending = new Map();

    this.sandbox.Api = {
      get: (url) => new Promise((resolve) => pending.set(new URLSearchParams(url.split("?")[1]).get("row"), resolve)),
    };

    const params        = Object.create(CubeParams.prototype);
    params.host         = { selectedId: "cube" };
    params.pixelSeq     = 0;
    params.coordsEl     = this._element();
    params.openBtn      = this._element({ hidden: true });
    params.rendered     = [];
    params._renderTable = (data) => params.rendered.push(data.row);

    const first  = params._fetchPixel(1, 1);
    const second = params._fetchPixel(2, 2);

    pending.get("2")({ ok: true, row: 2, col: 2, threshold: 0.1 });
    await second;
    pending.get("1")({ ok: true, row: 1, col: 1, threshold: 0.1 });
    await first;

    return { coords: params.coordsEl.textContent, rendered: params.rendered, open: !params.openBtn.hidden };
  }

  _metrics(answer) {
    const { CubeMetrics } = this.classes;

    this.sandbox.SharedCharts           = { esc: (text) => String(text) };
    this.sandbox.Api                    = { get: async (url) => answer(new URLSearchParams(url.split("?")[1])) };
    this.sandbox.window.addEventListener = () => {};

    const refs = {};
    ["layer", "rangeWrap", "classes", "vmin", "vmax", "reset", "mode", "thr", "thrVal", "alpha", "img", "cross", "cbar", "min", "max", "coords", "readout", "open", "selCov", "selVal", "selTab", "selMask", "selPlay", "histNote", "download", "rect", "regionHint", "regionTab", "regionWrap", "regionProf"].forEach((name) => {
      refs[name] = this._element({ innerHTML: "" });
    });

    const drawn = { fills: [], strokes: 0, cleared: 0 };
    const ctx   = {
      setTransform : () => {},
      clearRect    : () => { drawn.cleared += 1; },
      fillRect     : (x, y, w, h) => drawn.fills.push({ color: ctx.fillStyle, x, w, h }),
      beginPath    : () => {},
      moveTo       : () => {},
      lineTo       : () => {},
      stroke       : () => { drawn.strokes += 1; },
    };
    refs.hist = this._element({ width: 0, height: 0, clientWidth: 240, clientHeight: 92, getContext: () => ctx });

    const host = { selectedId: "cube", meta: { n_rows: 4, n_cols: 6 }, colors: { accent: "#a3c0ff", warn: "#fcd34d" }, _fmt: (value) => String(value) };
    return { metrics: new CubeMetrics(refs, host), refs, host, drawn };
  }

  async legend() {
    const legend  = [{ index: 0, name: "clean", color: "#4c78a8" }, { index: 1, name: "suspect", color: "#f58518" }];
    const answers = { label_suspect: { value: 1, name: "suspect" }, pixel_r2: { value: 0.25 } };
    const { metrics, refs } = this._metrics((query) => ({ ok: true, row: Number(query.get("row")), col: Number(query.get("col")), key: query.get("key"), ...answers[query.get("key")] }));

    const classLayer      = { key: "label_suspect", label: "label suspect", vmin: 0, vmax: 1, classes: legend };
    const continuousLayer = { key: "pixel_r2", label: "R2", vmin: 0, vmax: 1, classes: null };
    metrics.layers        = [classLayer, continuousLayer];

    const shown = () => ({ range: !refs.rangeWrap.hidden, cbar: !refs.cbar.hidden, bounds: !refs.min.hidden && !refs.max.hidden, legend: !refs.classes.hidden, html: refs.classes.innerHTML });

    metrics.layer = classLayer;
    metrics._draw();
    const discrete = shown();

    metrics.hoverQueued = { row: 1, col: 2 };
    await metrics._hoverPump();
    const hover = refs.coords.textContent;

    await metrics._fetchReadout(1, 2);
    const readout = refs.readout.innerHTML;

    metrics.layer = continuousLayer;
    metrics._draw();
    const continuous = shown();

    return { discrete, continuous, hover, readout };
  }

  async histogram() {
    const legend   = [{ index: 0, name: "clean", color: "#4c78a8" }, { index: 1, name: "suspect", color: "#f58518" }];
    const answers  = [];
    const { metrics, refs, host, drawn } = this._metrics((query) => new Promise((resolve) => answers.push({ query, resolve })));

    const classLayer      = { key: "label_suspect", label: "label suspect", vmin: 0, vmax: 1, classes: legend };
    const continuousLayer = { key: "pixel_r2", label: "R2", vmin: 0, vmax: 1, classes: null };
    const reset           = () => { drawn.fills = []; drawn.strokes = 0; };

    metrics.layer = classLayer;
    const classFetch = metrics._fetchHist();
    answers.shift().resolve({ ok: true, kind: "classes", names: ["clean", "suspect"], counts: [30, 10] });
    await classFetch;
    const classes = { fills: drawn.fills.map((fill) => fill.color), heights: drawn.fills.map((fill) => fill.h), note: !refs.histNote.hidden };

    reset();
    metrics.layer = continuousLayer;
    metrics._drawHist();
    const stale = { fills: drawn.fills.length, cleared: drawn.cleared > 0 };

    reset();
    refs.vmin.value = "0.25";
    refs.vmax.value = "0.75";
    metrics.mode    = "above";
    refs.thr.value  = "500";
    const continuousFetch = metrics._fetchHist();
    answers.shift().resolve({ ok: true, kind: "continuous", edges: [0, 0.25, 0.5, 0.75, 1], counts: [1, 4, 2, 0], n_low: 3, n_high: 1200 });
    await continuousFetch;
    const continuous = { fills: drawn.fills.length, band: drawn.fills[0].color, bars: drawn.fills[1].color, strokes: drawn.strokes, note: refs.histNote.hidden ? null : refs.histNote.textContent };

    metrics.hist    = null;
    metrics.histKey = null;
    const lateFetch = metrics._fetchHist();
    host.selectedId = "other";
    answers.shift().resolve({ ok: true, kind: "continuous", edges: [0, 1], counts: [5], n_low: 0, n_high: 0 });
    await lateFetch;
    const late = { hist: metrics.hist, histKey: metrics.histKey };

    return { classes, stale, continuous, late };
  }

  _exports() {
    const saved = [];
    const toasts = [];

    this.sandbox.Blob         = Blob;
    this.sandbox.Toast        = { show: (text, kind) => toasts.push({ text, kind }) };
    this.sandbox.FigureExport = {
      download   : (name, source) => saved.push({ via: "download", name, source }),
      saveCanvas : (canvas, name) => saved.push({ via: "canvas", name, source: canvas }),
      saveBitmap : (bitmap, name) => saved.push({ via: "bitmap", name, source: bitmap }),
    };
    return { saved, toasts };
  }

  async downloads() {
    const { CubeView, CubeTransect } = this.classes;
    const { saved, toasts } = this._exports();

    const view      = Object.create(CubeView.prototype);
    view.cubes      = [{ id: "cube", run: "group/run a", stamp: "stamp_1" }];
    view.selectedId = "cube";
    view.space      = "physical";
    view.meta       = { n_rows: 8, n_cols: 6 };

    view.point    = null;
    view.profData = null;
    view.panels   = [];
    view._downloadCuts();
    view._downloadProfilesCsv();
    const refused = toasts.map((toast) => toast.kind);

    const shown  = { root: { hidden: false }, bitmap: { id: "shown" }, axis: "col", source: "pred" };
    const hidden = { root: { hidden: true }, bitmap: { id: "hidden" }, axis: "row", source: "gt" };
    const blank  = { root: { hidden: false }, bitmap: null, axis: "row", source: "pred" };
    view.point   = { row: 3, col: 12 };
    view.panels  = [shown, hidden, blank];
    view._downloadCuts();
    const cuts = saved.splice(0).map((entry) => ({ via: entry.via, name: entry.name, bitmap: entry.source.id }));

    view.profData = { row: 3, col: 4, sources: { pred: { z_values: [-1, 0.5], values: [0.25, 2] }, gt: { z_values: [-1, 0.5], values: [0, 1] } } };
    view._downloadProfilesCsv();
    const csvEntry = saved.splice(0)[0];
    const csv      = { name: csvEntry.name, type: csvEntry.source.type, text: await csvEntry.source.text() };

    const transect  = Object.create(CubeTransect.prototype);
    transect.host   = view;
    transect.points = [{ row: 1, col: 2 }, { row: 5, col: 0 }];
    transect.panels = [{ root: { hidden: false }, canvas: { width: 40 }, source: "pred" }, { root: { hidden: false }, canvas: { width: 0 }, source: "gt" }];
    transect._download();
    const line = saved.splice(0).map((entry) => ({ via: entry.via, name: entry.name }));

    transect.points = [{ row: 1, col: 2 }, { row: 1, col: 5 }, { row: 5, col: 0 }];
    transect._download();
    const bent = saved.splice(0).map((entry) => entry.name);

    const { metrics, refs } = this._metrics(() => ({}));
    const exported          = this._exports();
    metrics.host            = view;
    metrics.layer           = { key: "failure_mode" };
    refs.img.src            = "/api/cubes/metric_map?id=cube&key=failure_mode";

    this.sandbox.fetch = async () => ({ ok: true, blob: async () => "png-bytes" });
    await metrics._downloadOverlay();
    this.sandbox.fetch = async () => ({ ok: false });
    await metrics._downloadOverlay();
    const overlay = { saved: exported.saved.map((entry) => ({ via: entry.via, name: entry.name, source: entry.source })), toasts: exported.toasts.map((toast) => toast.kind) };

    return { refused, cuts, csv, line, bent, overlay, tag: view.fileTag() };
  }

  _select() {
    const select = this._element();
    let html     = "";

    Object.defineProperty(select, "innerHTML", {
      get : () => html,
      set : (value) => {
        html                = value;
        select.optionValues = [...value.matchAll(/<option value="([^"]*)"/g)].map((match) => match[1]);
        select.value        = select.optionValues[0] || "";
      },
    });
    Object.defineProperty(select, "options", { get: () => select.optionValues || [] });
    return select;
  }

  _storage() {
    const items = new Map();
    return {
      items,
      getItem    : (key) => (items.has(key) ? items.get(key) : null),
      setItem    : (key, value) => items.set(key, String(value)),
      removeItem : (key) => items.delete(key),
    };
  }

  _transect(meta) {
    const { CubeTransect } = this.classes;
    const asked  = [];
    const toasts = [];

    this.sandbox.SharedCharts      = { esc: (text) => String(text) };
    this.sandbox.Api               = { get: (url) => new Promise((resolve) => asked.push({ query: new URLSearchParams(url.split("?")[1]), resolve })) };
    this.sandbox.Toast             = { show: (text, kind) => toasts.push({ text, kind }) };
    this.sandbox.fetch             = async () => ({ ok: false });
    this.sandbox.localStorage      = this.sandbox.localStorage || this._storage();

    const refs = {};
    ["at", "undo", "clear", "print", "download", "name", "save", "remove", "stripWrap", "stripVal", "map", "grid"].forEach((name) => {
      refs[name] = this._element({ innerHTML: "" });
    });
    refs.overlay    = this._element({ innerHTML: "" });
    refs.saved      = this._select();
    refs.stripLayer = this._select();

    const { ctx, ops } = this._recordingCtx();
    refs.strip         = this._element({ clientWidth: 200, clientHeight: 70, width: 0, height: 0, getContext: () => ctx });
    refs.map.getBoundingClientRect = () => ({ left: 0, top: 0, width: 60, height: 80 });
    refs.panels = [];

    const host = { selectedId: "cube", meta, view: "transect", space: "physical", cmap: "jet", visible: new Set(meta.sources), colors: { accent: "#a3c0ff" }, _fmt: (value) => String(value) };
    const transect = new CubeTransect(refs, host);
    transect.configure();

    return { transect, refs, host, asked, toasts, ops };
  }

  async transect() {
    const layers = [{ key: "pixel_mse", label: "MSE" }, { key: "pixel_r2", label: "R2" }];
    const meta   = { n_rows: 8, n_cols: 6, sources: ["pred"], metric_maps: layers };
    const { transect, refs, host, asked, toasts, ops } = this._transect(meta);
    const click  = (x, y) => refs.map.listeners.click({ clientX: x, clientY: y });
    const strip  = (entry, values) => entry.resolve({ ok: true, n_samples: values.length, rows: [{ key: "pixel_mse", label: "MSE", values }, { key: "pixel_r2", label: "R2", values: values.map((v) => (v === null ? null : -v)) }] });

    const configured = { layers: refs.stripLayer.options, saved: refs.saved.options, undo: refs.undo.hidden, strip: refs.stripWrap.hidden };

    click(25, 35);
    click(29, 39);
    const single = { points: transect.points.map((p) => [p.row, p.col]), asked: asked.length, undo: refs.undo.hidden, print: refs.print.hidden, at: refs.at.textContent };

    click(55, 35);
    click(55, 75);
    const traced = { pts: asked.map((entry) => entry.query.get("pts")), print: refs.print.hidden, at: refs.at.textContent, segments: (refs.overlay.innerHTML.match(/<line /g) || []).length };

    strip(asked.shift(), [1, 2]);
    await this._settle();
    const overtaken = { shown: !refs.stripWrap.hidden, data: transect.stripData };

    strip(asked.shift(), [1, null, 3]);
    await this._settle();
    const drawn = { shown: !refs.stripWrap.hidden, value: refs.stripVal.textContent, strokes: ops.strokes.length, segments: ops.strokes[0].points.length };

    transect._undo();
    const shortened = asked.splice(0).map((entry) => entry.query.get("pts"));
    transect._undo();
    const afterUndo = { shortened, points: transect.points.length, strip: refs.stripWrap.hidden, data: transect.stripData, asked: asked.length };

    click(55, 35);
    const pendingUndo = asked.shift();
    transect._undo();
    strip(pendingUndo, [4, 5]);
    await this._settle();
    const lateAfterUndo = { strip: refs.stripWrap.hidden, data: transect.stripData };

    click(55, 35);
    const pendingClear = asked.shift();
    refs.clear.listeners.click();
    strip(pendingClear, [4, 5]);
    await this._settle();
    const lateAfterClear = { strip: refs.stripWrap.hidden, data: transect.stripData, points: transect.points.length };

    click(25, 35);
    click(55, 35);
    click(55, 75);
    const superseded = asked.shift();
    strip(asked.shift(), [1, 2, 3]);
    strip(superseded, [7, 8]);
    await this._settle();
    refs.stripLayer.value = "pixel_r2";
    refs.stripLayer.listeners.change();
    const picked = refs.stripVal.textContent;

    host.meta = { ...meta, metric_maps: [...layers, { key: "delta_pixel_mse", label: "Δ MSE" }] };
    transect.renderStripLayers();
    const attached = { layers: refs.stripLayer.options, kept: refs.stripLayer.value, refetched: asked.map((entry) => entry.query.get("pts")) };
    asked.splice(0);

    refs.save.listeners.click();
    const unnamed = JSON.parse(this.sandbox.localStorage.getItem("cube-transects:8x6"));

    refs.name.value = "ridge";
    refs.save.listeners.click();
    const named = { stored: JSON.parse(this.sandbox.localStorage.getItem("cube-transects:8x6")), options: refs.saved.options, selected: refs.saved.value, remove: refs.remove.hidden, name: refs.name.value };

    refs.clear.listeners.click();
    refs.saved.value = "ridge";
    refs.saved.listeners.change();
    const replayed = { points: transect.points.map((p) => [p.row, p.col, p.fx, p.fy]), asked: asked.map((entry) => entry.query.get("pts")), remove: refs.remove.hidden };
    asked.splice(0);

    refs.remove.listeners.click();
    const deleted = { stored: JSON.parse(this.sandbox.localStorage.getItem("cube-transects:8x6")).map((entry) => entry.name), options: refs.saved.options };

    const other = this._transect({ n_rows: 4, n_cols: 6, sources: ["pred"], metric_maps: [] });
    const otherFootprint = { options: other.refs.saved.options, layers: other.refs.stripLayer.options };
    other.refs.map.listeners.click({ clientX: 5, clientY: 5 });
    other.refs.map.listeners.click({ clientX: 55, clientY: 75 });
    const noLayers = { asked: other.asked.length, strip: other.refs.stripWrap.hidden };

    other.refs.clear.listeners.click();
    other.refs.save.listeners.click();
    const refusedSave = toasts.concat(other.toasts).filter((toast) => toast.kind === "warn").map((toast) => toast.text);

    return { configured, single, traced, overtaken, drawn, afterUndo, lateAfterUndo, lateAfterClear, picked, attached, unnamed, named, replayed, deleted, otherFootprint, noLayers, refusedSave };
  }

  _gl(calls) {
    return new Proxy({}, {
      get: (target, name) => {
        if (typeof name === "string" && /^[A-Z0-9_]+$/.test(name)) return name;
        return (...args) => {
          calls.push({ name, args });
          return name === "getShaderParameter" || name === "getProgramParameter" ? true : {};
        };
      },
    });
  }

  _volumeBlob(dims, diverging, vmin, vmax) {
    const count  = dims[0] * dims[1] * dims[2];
    const buffer = new ArrayBuffer(32 + count);
    new Float32Array(buffer, 0, 8).set([...dims, diverging ? 1 : 0, vmin, vmax, 0, 0]);
    new Uint8Array(buffer, 32).set(Array.from({ length: count }, (_, i) => i % 256));
    return buffer;
  }

  _volume(context) {
    const { CubeVolume } = this.classes;
    const asked = [];
    const calls = [];

    this.sandbox.ThinkingOrb             = { busy: (state, text) => `<orb>${text}</orb>` };
    this.sandbox.window.addEventListener = () => {};
    this.sandbox.window.innerHeight      = 800;
    this.sandbox.fetch                   = (url) => new Promise((resolve) => asked.push({ query: new URLSearchParams(url.split("?")[1]), resolve }));

    const buttons = ["pred", "predb", "diff", "gt", "full"].map((source) => this._element({ dataset: { source } }));
    buttons.forEach((btn) => {
      btn.active          = false;
      btn.classList.toggle = (name, on) => { btn.active = on; };
    });

    const refs = {};
    ["lo", "hi", "opacity", "steps", "clipAxis", "clipPos", "at"].forEach((name) => {
      refs[name] = this._element({ innerHTML: "" });
    });
    refs.lo.value       = "35";
    refs.hi.value       = "90";
    refs.opacity.value  = "45";
    refs.steps.value    = "256";
    refs.clipAxis.value = "off";
    refs.clipPos.value  = "50";

    refs.source                  = this._element();
    refs.source.querySelectorAll = () => buttons;

    const options = [];
    refs.canvas = this._element({ width: 0, height: 0, parentElement: { clientWidth: 640 } });
    refs.canvas.getContext = (kind, opts) => {
      options.push({ kind, opts });
      return context ? this._gl(calls) : null;
    };

    const host   = { selectedId: "cube/a", meta: { sources: ["pred", "gt"] }, view: "volume", _fmt: (value) => String(value) };
    const volume = new CubeVolume(refs, host);
    const answer = async (entry, payload) => {
      entry.resolve(payload);
      await this._settle();
    };

    return { volume, refs, host, asked, calls, buttons, options, answer };
  }

  async volume() {
    const { volume, refs, host, asked, calls, buttons, options, answer } = this._volume(true);
    const blob = (vmin, vmax) => ({ ok: true, arrayBuffer: async () => this._volumeBlob([2, 3, 4], false, vmin, vmax) });

    volume.source = "full";
    volume.configure(host.meta);
    const configured = { source: volume.source, disabled: buttons.map((btn) => btn.disabled), active: buttons.map((btn) => btn.active) };

    volume.render();
    const context = options[0];
    const request = { id: asked[0].query.get("id"), source: asked[0].query.get("source"), max: asked[0].query.get("max"), at: refs.at.innerHTML };
    await answer(asked.shift(), blob(0.5, 2));
    const drawn = {
      dims       : volume.volume.dims,
      key        : volume.volumeKey,
      voxels     : volume.volume.voxels.length,
      upload     : calls.filter((call) => call.name === "texImage3D").map((call) => call.args.slice(3, 6)),
      draws      : calls.filter((call) => call.name === "drawArrays").length,
      at         : refs.at.textContent,
    };

    calls.splice(0);
    volume.render();
    const cached = { asked: asked.length, draws: calls.filter((call) => call.name === "drawArrays").length };

    volume._setSource("gt");
    const pending = asked.shift();
    host.selectedId = "cube/b";
    volume.configure({ sources: ["pred", "gt"] });
    await answer(pending, blob(9, 10));
    const late = { volume: volume.volume, key: volume.volumeKey };

    volume.render();
    await answer(asked.shift(), { ok: false });
    const failed = { volume: volume.volume, at: refs.at.textContent };

    const clips = {};
    ["off", "row", "col", "z"].forEach((axis) => {
      refs.clipAxis.value = axis;
      refs.clipPos.value  = "25";
      clips[axis]         = volume._clip().max;
    });

    volume.render();
    await answer(asked.shift(), blob(0, 1));
    volume.yaw = 2.0;
    volume.refreshSources({ sources: ["pred", "predb", "diff", "gt"] });
    const refreshed = { volume: volume.volume, yaw: volume.yaw, source: volume.source, disabled: buttons.map((btn) => btn.disabled) };

    host.view = "explorer";
    calls.splice(0);
    volume.render();
    await answer(asked.shift(), blob(0, 1));
    const hiddenDraws = calls.filter((call) => call.name === "drawArrays").length;

    const blind = this._volume(false);
    blind.volume.render();
    const unavailable = { available: blind.volume.available, at: blind.refs.at.textContent, asked: blind.asked.length };

    return { configured, context, request, drawn, cached, late, failed, clips, refreshed, hiddenDraws, unavailable, views: this._volumeViews() };
  }

  _volumeViews() {
    const { CubeView } = this.classes;
    const record       = (available) => {
      const calls = [];
      return { calls, available, configure: () => calls.push("configure"), refreshSources: () => calls.push("refresh"), render: () => calls.push("render") };
    };
    const button = () => ({ dataset: { view: "volume" }, hidden: false, classList: { toggle: () => {} } });

    const shown      = record(true);
    const shownBtn   = button();
    const shownStub  = this._displayStub("volume", { volume: shown, modeBtns: [shownBtn] });
    shownStub._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4 });

    const blind     = record(false);
    const blindBtn  = button();
    const blindStub = this._displayStub("volume", { volume: blind, modeBtns: [blindBtn], viewEls: [] });
    blindStub._stopSweeps = () => {};
    blindStub._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4 });

    const attached = record(true);
    const view     = Object.create(CubeView.prototype);
    Object.assign(view, { view: "volume", visible: new Set(["pred"]), panels: [], sweeps: [], metrics: null, transect: null, volume: attached, point: null, locked: null });
    ["_clearBitmapCache", "_renderSourceToggles", "_applyVisibility"].forEach((name) => { view[name] = () => {}; });
    view._refreshSources({ sources: ["pred", "predb", "diff", "gt"] });

    const picked = record(true);
    const picker = Object.create(CubeView.prototype);
    Object.assign(picker, { view: "explorer", meta: { sources: ["pred"] }, modeBtns: [], viewEls: [], sweeps: [], volume: picked, timeline: this._timelineStub() });
    picker._setView("volume");

    return {
      loaded   : { calls: shown.calls, hidden: shownBtn.hidden },
      blind    : { calls: blind.calls, hidden: blindBtn.hidden, view: blindStub.view },
      attached : attached.calls,
      picked   : { calls: picked.calls, view: picker.view },
    };
  }

  _timelineStub(calls = []) {
    return { calls, configure: () => calls.push("configure"), render: () => calls.push("render"), leave: () => calls.push("leave") };
  }

  _displayStub(view, extra) {
    const { CubeView } = this.classes;

    this.sandbox.getComputedStyle = () => ({ getPropertyValue: () => "#a3c0ff" });

    const stub = Object.create(CubeView.prototype);
    Object.assign(stub, {
      view         : view,
      selectedId   : "stamp/b",
      visible      : new Set(),
      sweeps       : [],
      modeBtns     : [],
      progress     : this._element(),
      coords       : this._element(),
      hint         : this._element(),
      stage        : this._element(),
      page         : this._element(),
      topdown      : this._element(),
      mapWrap      : this._element(),
      figures      : { configure: () => {}, render: () => {} },
      timeline     : this._timelineStub(),
      storyTimer   : null,
      storyPlayBtn : this._element(),
      ...extra,
    });
    ["_clearBitmapCache", "_syncSpaceBtns", "_syncProfModeBtns", "_renderSourceToggles", "_applyVisibility", "_initCutBounds", "_follow", "_consumeFocus"].forEach((name) => { stub[name] = () => {}; });
    stub._slowestStage = () => "read";
    stub._sweepFor     = () => null;
    return stub;
  }

  _focusView() {
    const { CubeView } = this.classes;
    const view         = Object.create(CubeView.prototype);

    Object.assign(view, {
      meta        : { n_rows: 4, n_cols: 6 },
      selectedId  : "cube",
      view        : "params",
      focus       : null,
      focusSeq    : 0,
      cutsAdopted : 0,
      mode        : "map",
      locked      : null,
      sweeps      : [],
      modeBtns    : [],
      viewEls     : [],
      deck        : { dataset: {} },
      back        : this._element(),
      slicesAt    : this._element(),
      followed    : [],
      profiled    : [],
      timeline    : this._timelineStub(),
    });
    view._follow        = (point) => view.followed.push([point.row, point.col]);
    view._queueProfiles = (row, col) => view.profiled.push([row, col]);
    return view;
  }

  async focus() {
    const { CubeParams } = this.classes;
    const view           = this._focusView();

    view.setFocus(9, -2);
    const clamped = view.focusPoint();

    const paramRefs = {};
    ["source", "field", "slot", "at", "cross", "cbar", "min", "max", "coords", "table", "open"].forEach((name) => { paramRefs[name] = this._element({ innerHTML: "" }); });
    paramRefs.canvas = this._element({ getBoundingClientRect: () => ({ left: 0, top: 0, width: 60, height: 40 }) });

    const params   = new CubeParams(paramRefs, view);
    params.meta    = { n_slots: 2 };
    params.fetched = [];
    ["_syncBtns", "_fetchMap", "_updateLegend"].forEach((name) => { params[name] = () => {}; });
    params._fetchPixel = (row, col) => params.fetched.push([row, col]);

    params._onClick({ clientX: 35, clientY: 15 });
    const paramPick = { focus: view.focus, seq: view.focusSeq, adopted: params.adopted, fetched: params.fetched.splice(0) };

    const readouts = [];
    const { metrics, refs } = this._metrics((query) => {
      readouts.push([Number(query.get("row")), Number(query.get("col"))]);
      return { ok: true, row: Number(query.get("row")), col: Number(query.get("col")), value: 0.5 };
    });
    metrics.host   = view;
    metrics.layers = [{ key: "pixel_r2", label: "R2", vmin: 0, vmax: 1, classes: null }];
    metrics.layer  = metrics.layers[0];
    ["_refresh", "_fetchHist", "_onCoverage"].forEach((name) => { metrics[name] = () => {}; });

    metrics.render();
    await this._settle();
    metrics.render();
    await this._settle();
    const adopted = { picked: [metrics.picked.row, metrics.picked.col], coords: refs.coords.textContent, cross: [refs.cross.style.left, refs.cross.style.top], open: !refs.open.hidden, readouts: readouts.splice(0) };

    refs.img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 60, height: 40 });
    metrics.picked = null;
    metrics._onClick({ clientX: 55, clientY: 35, shiftKey: false });
    await this._settle();
    const metricPick = { focus: view.focus, adopted: metrics.adopted === view.focusSeq, readouts: readouts.splice(0) };

    params.render();
    const paramAdopt = { picked: [params.picked.row, params.picked.col], fetched: params.fetched.splice(0), cross: !paramRefs.cross.hidden };

    view.params  = params;
    view.metrics = metrics;
    view.view    = "metrics";
    view._setView("explorer");
    const entered = { followed: view.followed.splice(0), mode: view.mode, locked: view.locked, profiled: view.profiled.splice(0), adopted: view.cutsAdopted === view.focusSeq };

    view._setView("params");
    view._setView("explorer");
    const kept = { followed: view.followed.splice(0), fetched: params.fetched.splice(0) };

    view.mode         = "map";
    view.locked       = null;
    view.view         = "metrics";
    view.pendingFocus = { row: 99, col: 2 };
    view._consumeFocus();
    const opened = { focus: view.focus, mode: view.mode, followed: view.followed.splice(0), view: view.view };

    metrics.picked = null;
    view._setView("metrics");
    await this._settle();
    const reopened = { picked: [metrics.picked.row, metrics.picked.col], readouts: readouts.splice(0) };

    const jumpParams = this._element({ hidden: true });
    const display    = this._displayStub("explorer", { jumpParams, profMixWrap: this._element(), params: { configure: () => {} } });
    display._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4, params: { sources: ["pred"] } });
    const withParams = !jumpParams.hidden;
    display._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4, params: null });
    const button = { withParams, withoutParams: !jumpParams.hidden };

    return { clamped, paramPick, adopted, metricPick, paramAdopt, entered, kept, opened, reopened, button };
  }

  _recordingCtx() {
    const ops = { strokes: [], dots: [], path: [], dash: [] };
    const ctx = {
      setTransform : () => {},
      clearRect    : () => {},
      closePath    : () => {},
      fill         : () => {},
      fillText     : () => {},
      setLineDash  : (dash) => { ops.dash = dash; },
      beginPath    : () => { ops.path = []; },
      moveTo       : (x, y) => ops.path.push([x, y]),
      lineTo       : (x, y) => ops.path.push([x, y]),
      arc          : (x, y) => ops.dots.push({ x, y, color: ctx.fillStyle }),
      stroke       : () => ops.strokes.push({ color: ctx.strokeStyle, dashed: ops.dash.length > 0, points: ops.path.slice() }),
    };
    ops.reset = () => { ops.strokes = []; ops.dots = []; };
    return { ctx, ops };
  }

  async region() {
    const asked  = [];
    const toasts = [];
    const { metrics, refs, host } = this._metrics((query) => new Promise((resolve) => asked.push({ query, resolve })));
    const { ctx, ops }            = this._recordingCtx();

    this.sandbox.Toast = { show: (text, kind) => toasts.push({ text, kind }) };

    host.colors.pred = "#a3c0ff";
    host.colors.gt   = "#34d399";
    host.setFocus    = () => {};
    host.focusSeq    = 0;

    metrics.layer  = { key: "pixel_r2", label: "R2", vmin: 0, vmax: 1, classes: null };
    metrics.layers = [metrics.layer];
    Object.assign(refs.regionProf, { clientWidth: 240, clientHeight: 92, width: 0, height: 0, getContext: () => ctx });
    refs.img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 60, height: 40 });

    const bounds  = (entry) => ["row0", "row1", "col0", "col1"].map((key) => Number(entry.query.get(key)));
    const press   = (x, y, shiftKey) => ({ clientX: x, clientY: y, shiftKey, button: 0, preventDefault: () => { press.prevented += 1; } });
    press.prevented = 0;
    const answer  = (row0, row1, col0, col1) => ({
      ok       : true,
      row0, row1, col0, col1,
      n_pixels : (row1 - row0 + 1) * (col1 - col0 + 1),
      rows     : [{ key: "pixel_r2", label: "R2", region: 0.5, scene: 0.25 }, { key: "label_r2", label: "label R2", region: null, scene: 0.75 }],
      profiles : { pred: { z_values: [0, 1, 2], values: [0, 2, 1] }, gt: { z_values: [0, 1, 2], values: [1, 1, 0] } },
    });

    metrics._onRectStart(press(10, 5, true));
    metrics._onRectMove({ clientX: 45, clientY: 35 });
    const painted = { hidden: refs.rect.hidden, style: { ...refs.rect.style }, prevented: press.prevented };
    metrics._onRectEnd();
    const request = { bounds: bounds(asked[0]), id: asked[0].query.get("id") };
    asked.shift().resolve(answer(0, 3, 1, 4));
    await this._settle();
    const shown = { hint: refs.regionHint.textContent, table: refs.regionTab.innerHTML, wrap: !refs.regionWrap.hidden, strokes: ops.strokes.map((stroke) => stroke.color) };

    metrics._onClick(press(30, 20, false));
    const swallowed = { picked: metrics.picked, suppress: metrics.suppressClick, asked: asked.length };

    metrics._onRectStart(press(10, 5, true));
    metrics._onRectEnd();
    metrics._onClick(press(10, 5, true));
    const tiny = { rect: refs.rect.hidden, asked: asked.length, picked: metrics.picked };

    metrics._onRectStart(press(0, 0, true));
    metrics._onRectMove({ clientX: 80, clientY: 50 });
    metrics._onRectEnd();
    const clipped = bounds(asked[0]);
    asked.shift().resolve(answer(0, 3, 0, 5));
    await this._settle();

    metrics._onRectStart(press(30, 20, false));
    metrics._onClick(press(30, 20, false));
    const offMap = { picked: metrics.picked ? [metrics.picked.row, metrics.picked.col] : null };
    asked.splice(0).forEach((entry) => entry.resolve({ ok: true, value: 0.5 }));
    await this._settle();

    metrics._fetchRegion(0, 1, 0, 1);
    metrics._fetchRegion(2, 3, 2, 3);
    const [first, second] = asked.splice(0);
    second.resolve(answer(2, 3, 2, 3));
    await this._settle();
    first.resolve(answer(0, 1, 0, 1));
    await this._settle();
    const late = refs.regionHint.textContent;

    metrics._fetchRegion(0, 1, 0, 1);
    host.selectedId = "other";
    asked.shift().resolve(answer(0, 1, 0, 1));
    await this._settle();
    const replaced = refs.regionHint.textContent;

    metrics._fetchRegion(0, 1, 0, 1);
    asked.shift().resolve({ ok: false, error: "cube not loaded" });
    await this._settle();

    metrics._renderLayerBtns = () => {};
    metrics.configure({ metric_maps: [metrics.layer] });
    const reset = { rect: refs.rect.hidden, table: refs.regionTab.innerHTML, wrap: refs.regionWrap.hidden, hint: refs.regionHint.textContent };

    return { painted, request, shown, swallowed, tiny, clipped, offMap, late, replaced, toasts, reset };
  }

  async delta() {
    const { CubeView }      = this.classes;
    const { metrics, refs } = this._metrics(() => ({}));

    metrics.layer = { key: "delta_pixel_mse", label: "Δ MSE", vmin: -1, vmax: 1, classes: null, diverging: true };
    metrics._draw();
    const diverging = refs.cbar.src;

    metrics.layer = { key: "pixel_mse", label: "MSE", vmin: 0, vmax: 1, classes: null, diverging: false };
    metrics._draw();
    const plain = refs.cbar.src;

    const calls      = [];
    const metricsBtn = { dataset: { view: "metrics" }, hidden: true };
    const view       = Object.create(CubeView.prototype);
    Object.assign(view, {
      view     : "metrics",
      visible  : new Set(["pred"]),
      panels   : [],
      sweeps   : [],
      modeBtns : [metricsBtn],
      point    : null,
      locked   : null,
      metrics  : { configure: (meta) => calls.push(["configure", ...meta.metric_maps.map((layer) => layer.key)]), render: () => calls.push(["render"]) },
    });
    ["_clearBitmapCache", "_renderSourceToggles", "_applyVisibility"].forEach((name) => { view[name] = () => {}; });
    view._sweepFor = () => null;
    view._setView  = (name) => calls.push(["view", name]);

    view._refreshSources({ sources: ["pred", "predb", "diff", "gt"], metric_maps: [{ key: "pixel_mse" }, { key: "delta_pixel_mse" }] });
    const attached = { calls: calls.splice(0), button: !metricsBtn.hidden };

    view._refreshSources({ sources: ["pred", "gt"], metric_maps: [] });
    const emptied = { calls: calls.splice(0), button: !metricsBtn.hidden };

    return { diverging, plain, attached, emptied };
  }

  async mixture() {
    const { CubeView } = this.classes;
    const calls        = [];
    const pending      = [];

    this.sandbox.Api = {
      get: (url) => {
        const query = new URLSearchParams(url.split("?")[1]);
        calls.push(`${url.split("?")[0]} ${query.get("row")},${query.get("col")}`);
        return new Promise((resolve) => pending.push(resolve));
      },
    };

    const slots = {
      pred : [{ amp: 2, mu: 1, sigma: 0.5, active: true }, { amp: 1, mu: 3, sigma: 0.25, active: true }, { amp: 0.5, mu: 2, sigma: 1, active: false }],
      gt   : [{ amp: 1, mu: 2, sigma: 0.5, active: true }, { amp: 0, mu: null, sigma: null, active: false }],
    };
    const series  = { z_values: [0, 1, 2, 3, 4], values: [0, 2, 0.5, 1, 0] };
    const params  = (row, col) => ({ ok: true, row, col, n_slots: 3, threshold: 0.1, sources: slots });
    const profile = (row, col) => ({ ok: true, row, col, sources: { pred: series, gt: series } });

    const { ctx, ops } = this._recordingCtx();
    const slotColors   = new Set(CubeView.SLOT_COLORS);
    const summary      = () => ({ dashed: ops.strokes.filter((stroke) => stroke.dashed).length, stems: ops.dots.filter((dot) => slotColors.has(dot.color)).map((dot) => dot.color) });

    const view = Object.create(CubeView.prototype);
    Object.assign(view, {
      selectedId    : "cube",
      meta          : { n_rows: 4, n_cols: 6, metric_maps: [], params: { sources: ["pred", "gt"] } },
      profMode      : "raw",
      profMix       : false,
      profData      : profile(1, 2),
      profParams    : null,
      profMetricsEl : null,
      colors        : {},
      profAt        : this._element(),
      profPanels    : ["pred", "gt"].map((source) => ({
        source,
        root   : this._element(),
        canvas : { clientWidth: 240, clientHeight: 190, width: 0, height: 0, getContext: () => ctx },
      })),
    });

    view._setProfMix(true);
    const requested = { calls: calls.splice(0), drawn: summary() };
    pending.shift()(params(1, 2));
    await this._settle();
    const drawn = summary();

    const identity = (value) => value;
    const curve    = () => ops.strokes.find((stroke) => stroke.dashed).points;
    const stems    = () => ops.strokes.filter((stroke) => slotColors.has(stroke.color)).map((stroke) => ({ color: stroke.color, points: stroke.points }));

    ops.reset();
    view._drawMixture(view.profPanels[0], series, ctx, identity, identity);
    const raw = { curve: curve(), stems: stems() };

    ops.reset();
    view.profMode = "unit";
    view._drawMixture(view.profPanels[0], series, ctx, identity, identity);
    const unit = { curve: curve(), stems: stems() };

    ops.reset();
    view.profMode = "raw";
    view._setProfMix(false);
    const off = { calls: calls.splice(0), drawn: summary() };

    view._setProfMix(true);
    const cached = { calls: calls.splice(0), drawn: summary() };

    const moved = view._fetchProfiles(3, 4);
    pending.shift()(profile(3, 4));
    await moved;
    const followed = calls.splice(0);
    pending.shift()(params(3, 4));
    await this._settle();

    const early = view._fetchProfParams(3, 4);
    view.profData = profile(2, 2);
    const later = view._fetchProfParams(2, 2);
    pending.pop()(params(2, 2));
    await later;
    pending.pop()(params(3, 4));
    await early;
    const late = { params: [view.profParams.row, view.profParams.col], matches: view._mixtureMatches() };

    const swapped = view._fetchProfParams(2, 2);
    view.selectedId = "other";
    view.profParams = null;
    pending.pop()(params(2, 2));
    await swapped;
    const replaced = { params: view.profParams };

    const profMixWrap = this._element({ hidden: true });
    const display     = this._displayStub("explorer", { jumpParams: this._element(), profMixWrap, params: { configure: () => {} } });
    display._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4, params: { sources: ["pred"] } });
    const withParams = !profMixWrap.hidden;
    display._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4, params: null });
    const toggle = { withParams, withoutParams: !profMixWrap.hidden };

    return { requested, drawn, raw, unit, off, cached, followed, late, replaced, toggle };
  }

  _reloaded(view) {
    const shown = [];
    const stub  = this._displayStub(view, { figures: { configure: () => shown.push("configure"), render: () => shown.push("render") } });

    stub._display({ timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4 });
    return shown;
  }

  async figures() {
    const { CubeFigures } = this.classes;
    const calls   = [];
    const answers = [];

    this.sandbox.SharedCharts = { esc: (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;") };
    this.sandbox.ThinkingOrb  = { panel: (state, text) => `<orb>${text}</orb>` };
    this.sandbox.Api          = { get: (url) => { calls.push(url); return new Promise((resolve) => answers.push(resolve)); } };

    const host    = { selectedId: "stamp/a" };
    const figures = new CubeFigures({ root: this._element({ innerHTML: "" }) }, host);

    const answer = async (payload) => {
      const pending = figures.render();
      answers.shift()(payload);
      await pending;
      return figures.root.innerHTML;
    };

    figures.configure();
    const loading = figures.root.innerHTML;
    const listed  = await answer({ ok: true, groups: [{ dir: "animations", files: ["walk z.gif"] }, { dir: "figures/pixel_maps", files: ["mse.png", "r2.png"] }] });

    await figures.render();
    const cached = calls.length;

    figures.configure();
    const late = figures.render();
    host.selectedId = "stamp/b";
    answers.shift()({ ok: true, groups: [] });
    await late;
    const dropped = { loadedFor: figures.loadedFor, html: figures.root.innerHTML };

    figures.configure();
    const none = await answer({ ok: true, groups: [] });

    figures.configure();
    const error = await answer({ ok: false, error: "cube not loaded" });

    return { loading, listed, cached, dropped, none, error, calls, reloaded: { figures: this._reloaded("figures"), explorer: this._reloaded("explorer") } };
  }

  async stage() {
    const staged   = new Set();
    const page     = this._element();
    page.classList = { toggle: (name, on) => (on ? staged.add(name) : staged.delete(name)) };

    const meta = { timings: { total: 1 }, sources: ["pred"], n_rows: 4, n_cols: 4 };
    const view = this._displayStub("explorer", { page });
    const read = () => ({ hidden: view.stage.hidden, staged: staged.has("is-staged") });

    view.stage.hidden = true;
    const before      = read();

    view._display(meta);
    const shown = read();

    Object.assign(view, { polling: false, cubes: [], panels: [], runStrip: null, focusSeq: 0, atLabels: {}, deck: { dataset: {} }, back: this._element(), cross: this._element(), slicesAt: this._element(), profAt: this._element(), slicesEl: this._element() });
    ["_stopSweeps", "_clearProfMetrics", "_hideRefs", "_renderStrip"].forEach((name) => { view[name] = () => {}; });
    this.sandbox.Api            = { post: async () => ({ ok: false, error: "cube gone" }), get: async () => ({ cubes: [] }) };
    this.sandbox.ThinkingOrb    = { busy: () => "" };
    this.sandbox.ResultsSources = { runs: () => "runs" };

    await view.select("stamp/c");
    const loading = read();

    view._display(meta);
    const reshown = read();

    await view.refresh();
    const empty = read();

    return { before, shown, loading, reshown, empty };
  }

  async run(name) {
    const result = await this[name]();
    process.stdout.write(JSON.stringify(result));
  }
}

new CubeViewChecks(path.resolve(__dirname, "..", "..", "static", "js")).run(process.argv[2]);
