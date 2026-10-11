"use strict";
const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class GaugeChecks {
  constructor(staticDir) {
    this.staticDir = staticDir;
    this.clock     = { now: 0 };
    this.frames    = [];
    this.timers    = new Map();
    this.cleared   = [];
    this.nextTimer = 0;
    this.fontGate  = null;
  }

  _context(reduced) {
    let release   = null;
    const ready   = new Promise((resolve) => { release = resolve; });
    this.fontGate = { ready, release };

    const sandbox = {
      console,
      window               : { matchMedia: () => ({ matches: reduced }), addEventListener: () => {}, removeEventListener: () => {}, devicePixelRatio: 1 },
      document             : { fonts: { ready } },
      performance          : { now: () => this.clock.now },
      requestAnimationFrame: (handler) => { this.frames.push(handler); return this.frames.length; },
      cancelAnimationFrame : () => {},
      setTimeout           : (handler) => { this.nextTimer += 1; this.timers.set(this.nextTimer, handler); return this.nextTimer; },
      clearTimeout         : (id) => { this.cleared.push(id); this.timers.delete(id); },
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    ["canvas_base.js", "gauges.js"].forEach((name) => {
      vm.runInContext(fs.readFileSync(path.join(this.staticDir, name), "utf8"), sandbox, { filename: name });
    });
    return sandbox;
  }

  _canvas() {
    const draws = { count: 0 };
    const ctx   = new Proxy({}, {
      get: (target, key) => {
        if (key === "clearRect") return () => { draws.count += 1; };
        if (key === "createLinearGradient") return () => ({ addColorStop: () => {} });
        if (key in target) return target[key];
        return () => {};
      },
      set: (target, key, value) => { target[key] = value; return true; },
    });
    const canvas = { width: 0, height: 0, getContext: () => ctx, getBoundingClientRect: () => ({ width: 172, height: 140 }) };
    return { canvas, draws };
  }

  _run(limit) {
    let ran = 0;
    while (this.frames.length && ran < limit) {
      this.clock.now += 16;
      this.frames.shift()(this.clock.now);
      ran += 1;
    }
    return ran;
  }

  _fire(id) {
    const handler = this.timers.get(id);
    this.timers.delete(id);
    handler();
  }

  async fonts() {
    const sandbox = this._context(true);
    const dial    = this._canvas();
    const meter   = this._canvas();

    new sandbox.window.DialGauge(dial.canvas, { big: true, label: "UTIL %" });
    new sandbox.window.LinearMeter(meter.canvas, { label: "TEMP" });
    const before = { dial: dial.draws.count, meter: meter.draws.count };

    this.fontGate.release();
    await new Promise((resolve) => setImmediate(resolve));

    return { before, after: { dial: dial.draws.count, meter: meter.draws.count } };
  }

  sweep() {
    const sandbox = this._context(false);
    const Dial    = sandbox.window.DialGauge;
    const dial    = new Dial(this._canvas().canvas, { big: true, label: "UTIL %" });

    dial.set(42);
    const first = { target: dial.t, held: dial.held, trail: dial.trail.slice(), timer: dial._sweep !== null, frames: this.frames.length };

    dial.set(55);
    const during = { target: dial.t, held: dial.held, trail: dial.trail.slice() };

    this._fire(dial._sweep);
    const landed = { target: dial.t, held: dial.held, trail: dial.trail.slice(), timer: dial._sweep };

    dial.set(60);
    for (let i = 0; i < Dial.TRAIL + 6; i++) dial.set(i);
    const capped = { length: dial.trail.length, last: dial.trail[dial.trail.length - 1], cap: Dial.TRAIL };

    return { first, during, landed, capped };
  }

  dispose() {
    const sandbox = this._context(false);
    const dial    = new sandbox.window.DialGauge(this._canvas().canvas, { big: true, label: "UTIL %" });

    dial.set(30);
    const timer = dial._sweep;
    dial.dispose();

    return { timer, cleared: this.cleared.includes(timer), pending: this.timers.has(timer) };
  }

  alarm() {
    const sandbox = this._context(false);
    const zones   = [{ from: 75, to: 90, color: "251, 191, 36" }, { from: 90, to: 100, color: "248, 113, 113" }];
    const dial    = new sandbox.window.DialGauge(this._canvas().canvas, { big: true, label: "BUSY %", zones });

    dial.swept = true;
    dial.set(40);
    const calmFrames = this._run(2000);
    const calm       = { value: dial.v, raf: dial._raf, frames: calmFrames };

    dial.set(96);
    const hotFrames = this._run(600);
    const hot       = { value: dial.v, raf: dial._raf !== null, frames: hotFrames, alarm: dial._alarm() };

    dial.set(20);
    const coolFrames = this._run(2000);
    const cool       = { value: dial.v, raf: dial._raf, frames: coolFrames };

    return { calm, hot, cool };
  }

  async run(name) {
    const result = await this[name]();
    process.stdout.write(JSON.stringify(result));
  }
}

new GaugeChecks(path.resolve(__dirname, "..", "..", "static", "js")).run(process.argv[2]);
