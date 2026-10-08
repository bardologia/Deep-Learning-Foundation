const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class CubeViewChecks {
  constructor(staticDir) {
    this.sandbox = { window: {}, console, setTimeout, clearTimeout };
    this.sandbox.globalThis = this.sandbox;
    vm.createContext(this.sandbox);

    const source = fs.readFileSync(path.join(staticDir, "cube_view.js"), "utf8");
    vm.runInContext(`${source}\nglobalThis.classes = { CubeSweep, CubeParams, CubeCloud, CubeView };`, this.sandbox, { filename: "cube_view.js" });
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

  async run(name) {
    const result = await this[name]();
    process.stdout.write(JSON.stringify(result));
  }
}

new CubeViewChecks(path.resolve(__dirname, "..", "..", "static", "js")).run(process.argv[2]);
