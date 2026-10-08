const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class CubeViewChecks {
  constructor(staticDir) {
    this.sandbox = { window: {}, console, setTimeout, clearTimeout };
    this.sandbox.globalThis = this.sandbox;
    vm.createContext(this.sandbox);

    const source = fs.readFileSync(path.join(staticDir, "cube_view.js"), "utf8");
    vm.runInContext(`${source}\nglobalThis.classes = { CubeSweep, CubeParams, CubeMetrics, CubeCloud, CubeView };`, this.sandbox, { filename: "cube_view.js" });
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
      view       : "cloud",
      selectedId : "cube",
      meta       : { n_rows: 10, n_cols: 20, sources: ["pred"], aux: false, params: { sources: ["pred"], threshold: 0.1, ranges: { amp: [0, 1] } } },
      _fmt       : (value) => String(value),
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
    cloud.points   = new Float32Array([0, 0, -5, 1, 5, 5, 5, 1, 9, 19, 15, 1]);
    cloud.total    = 3;
    cloud.muRange  = [-5, 15];
    cloud.muBounds = CubeCloud.valueBounds(cloud.points, 2, 4);
    return { cloud, refs, host, fills };
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
    cloud.muBounds     = CubeCloud.valueBounds(cloud.points, 2, 4);

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

    const view  = Object.create(CubeView.prototype);
    view.sweeps = [];
    view.cloud  = cloud;
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

    this.sandbox.SharedCharts = { esc: (text) => String(text) };
    this.sandbox.Api          = { get: async (url) => answer(new URLSearchParams(url.split("?")[1])) };

    const refs = {};
    ["layer", "rangeWrap", "classes", "vmin", "vmax", "reset", "mode", "thr", "thrVal", "alpha", "img", "cross", "cbar", "min", "max", "coords", "readout", "open"].forEach((name) => {
      refs[name] = this._element({ innerHTML: "" });
    });

    const host = { selectedId: "cube", meta: { n_rows: 4, n_cols: 6 }, _fmt: (value) => String(value) };
    return { metrics: new CubeMetrics(refs, host), refs };
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

  async run(name) {
    const result = await this[name]();
    process.stdout.write(JSON.stringify(result));
  }
}

new CubeViewChecks(path.resolve(__dirname, "..", "..", "static", "js")).run(process.argv[2]);
