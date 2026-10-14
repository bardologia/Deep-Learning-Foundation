import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

class FakeTerm {
  constructor() {
    this.writes = [];
    this.resets = 0;
  }

  write(text) {
    this.writes.push(text);
  }

  writeln(text) {
    this.writes.push(`${text}\n`);
  }

  reset() {
    this.resets += 1;
    this.writes = [];
  }

  get text() {
    return this.writes.join("");
  }
}


class FakeXterm {
  constructor(options) {
    const log    = [];
    this.log     = log;
    this.cols    = options.cols;
    this.rows    = options.rows;
    this.element = null;
    this.options = new Proxy({ ...options }, {
      set(target, name, value) {
        if (name === "fontSize") log.push(["font", value]);
        target[name] = value;
        return true;
      },
    });
  }

  loadAddon(addon) {
    addon.term = this;
  }

  resize(cols, rows) {
    this.cols = cols;
    this.rows = rows;
    this.log.push(["resize", cols, rows]);
  }

  open(host) {
    this.element = host;
    this.log.push(["open"]);
  }

  scrollToBottom() {
    return;
  }
}


class FakeFitAddon {
  static CELL_WIDTH  = 0.6;
  static CELL_HEIGHT = 1.25;

  proposeDimensions() {
    const host = this.term.element;
    if (!host) return undefined;

    const width  = host.clientWidth  ?? parseFloat(host.style.width);
    const height = host.clientHeight ?? parseFloat(host.style.height);
    const font   = this.term.options.fontSize;
    return { cols: Math.floor((width - 24) / (font * FakeFitAddon.CELL_WIDTH)), rows: Math.floor((height - 20) / (font * FakeFitAddon.CELL_HEIGHT)) };
  }
}


class FakeDocument {
  constructor() {
    this.body = { children: [], appendChild(el) { this.children.push(el); } };
  }

  createElement(tag) {
    return { tag, className: "", style: {}, setAttribute() { return; } };
  }
}


class FakeStreamHub {
  constructor() {
    this.subs      = [];
    this.connected = true;
  }

  subscribe(jobId, unit, handler) {
    const sub = { jobId, unit, handler, closed: false, close() { this.closed = true; } };
    this.subs.push(sub);
    return sub;
  }
}


class ConsoleClientSuite {
  constructor(source) {
    this.hub      = new FakeStreamHub();
    this.document = new FakeDocument();
    this.timers   = new Map();
    this.frames   = new Map();
    this.handle   = 0;

    const sandbox = {
      window                : {},
      console,
      document              : this.document,
      Terminal              : FakeXterm,
      FitAddon              : { FitAddon: FakeFitAddon },
      StreamHub             : { instance: () => this.hub },
      setTimeout            : (fn) => this._schedule(this.timers, fn),
      clearTimeout          : (handle) => this.timers.delete(handle),
      requestAnimationFrame : (fn) => this._schedule(this.frames, fn),
      cancelAnimationFrame  : (handle) => this.frames.delete(handle),
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(`${source}\nglobalThis.classes = { ConsoleTile, RunConsole };`, sandbox, { filename: "console.js" });

    this.classes = sandbox.classes;
  }

  _schedule(queue, fn) {
    this.handle += 1;
    queue.set(this.handle, fn);
    return this.handle;
  }

  flush(queue) {
    const due = [...queue.values()];
    queue.clear();
    due.forEach((fn) => fn());
  }

  tile() {
    this.hub.subs.length = 0;
    this.hub.connected   = true;

    const tile      = Object.create(this.classes.ConsoleTile.prototype);
    tile.job        = { job_id: "job-1", status: "running" };
    tile.term       = new FakeTerm();
    tile.unitTerm   = new FakeTerm();
    tile.sub        = null;
    tile.unitSub    = null;
    tile.streamLost = false;
    return tile;
  }

  testATileFollowsItsJobThroughThePageHub() {
    const tile = this.tile();
    tile._connect();

    assert.equal(this.hub.subs.length, 1);
    assert.equal(this.hub.subs[0].jobId, "job-1");
    assert.equal(this.hub.subs[0].unit, null);

    this.hub.subs[0].handler({ key: "job-1", type: "chunk", data: "epoch 1", id: 1 });
    assert.equal(tile.term.text, "epoch 1");
  }

  testALostLinkIsNotedOnceUntilTheStreamDeliversAgain() {
    const tile = this.tile();
    tile._connect();
    const deliver = this.hub.subs[0].handler;

    deliver({ type: "link", state: "lost" });
    deliver({ type: "link", state: "lost" });
    assert.equal(tile.term.writes.filter((w) => w.includes("live stream interrupted")).length, 1);

    deliver({ key: "job-1", type: "chunk", data: "x", id: 2 });
    deliver({ type: "link", state: "lost" });
    assert.equal(tile.term.writes.filter((w) => w.includes("live stream interrupted")).length, 2);
  }

  testATruncationMarkerIsNotedOnTheTerminalItReached() {
    const tile = this.tile();

    tile._onEvent({ key: null, type: "truncated", dropped: 12 });
    tile._onUnitEvent({ key: null, type: "truncated", dropped: 3 });

    assert.match(tile.term.text, /stream truncated: 12 event\(s\) dropped/);
    assert.match(tile.unitTerm.text, /stream truncated: 3 event\(s\) dropped/);
  }

  testAnUnavailableStreamIsNotedAndReleased() {
    const tile = this.tile();
    tile._connect();
    const sub = this.hub.subs[0];

    sub.handler({ type: "unavailable", error: "unknown job" });

    assert.match(tile.term.text, /live stream unavailable: unknown job/);
    assert.equal(sub.closed, true);
    assert.equal(tile.sub, null);
    assert.equal(tile.streamOpen, false);
  }

  testTheStreamCountsAsOpenOnlyWhileTheHubLinkIsUp() {
    const tile = this.tile();
    tile._connect();
    assert.equal(tile.streamOpen, true);

    this.hub.connected = false;
    assert.equal(tile.streamOpen, false);
  }

  testAReplayedSubRunTailResetsItsTerminalInsteadOfRepeatingIt() {
    const tile = this.tile();
    tile.unitSub = this.hub.subscribe("job-1", 2, (data) => tile._onUnitEvent(data));
    const deliver = tile.unitSub.handler;

    deliver({ key: "job-1/2", type: "unit", name: "seed0", log: "/x/seed0.log", id: 1 });
    deliver({ key: "job-1/2", type: "chunk", data: "line 1\n", id: 2 });
    deliver({ key: "job-1/2", type: "unit", name: "seed0", log: "/x/seed0.log", id: 1 });
    deliver({ key: "job-1/2", type: "chunk", data: "line 1\n", id: 2 });

    assert.equal(tile.unitTerm.resets, 2);
    assert.equal(tile.unitTerm.text.split("line 1").length - 1, 1, "the replayed tail is shown once");
    assert.match(tile.unitTerm.text, /seed0 — following \/x\/seed0\.log/);
  }

  testAnUnavailableSubRunIsNotedOnItsTerminalAndReleased() {
    const tile = this.tile();
    tile.unitSub = this.hub.subscribe("job-1", 4, (data) => tile._onUnitEvent(data));
    const sub    = tile.unitSub;

    sub.handler({ type: "unavailable", error: "unit index 4 is out of range" });

    assert.match(tile.unitTerm.text, /sub-run stream unavailable: unit index 4 is out of range/);
    assert.equal(sub.closed, true);
    assert.equal(tile.unitSub, null);
  }

  console() {
    const runConsole     = Object.create(this.classes.RunConsole.prototype);
    runConsole.tiles     = new Map();
    runConsole._gauge    = null;
    runConsole._fitTimer = null;
    runConsole._fitFrame = 0;
    return runConsole;
  }

  sizingTile(width, height) {
    const { ConsoleTile } = this.classes;

    const tile   = Object.create(ConsoleTile.prototype);
    tile.manager = this.console();
    tile.opened  = false;

    const term = new FakeXterm(ConsoleTile.TERMINAL_OPTIONS);
    const fit  = new FakeFitAddon();
    term.loadAddon(fit);

    return { tile, term, fit, host: { clientWidth: width, clientHeight: height } };
  }

  testFontSizesAreQuantisedToQuarterPixels() {
    const { ConsoleTile } = this.classes;

    assert.equal(ConsoleTile.quantize(12.13), 12.25);
    assert.equal(ConsoleTile.quantize(12.1), 12);
    assert.equal(ConsoleTile.quantize(7.874), 7.75);
  }

  testATileSizesItsTerminalBeforeOpeningIt() {
    const { tile, term, fit, host } = this.sizingTile(900, 400);

    tile._fitInto(term, fit, host, "opened");

    const openAt = term.log.findIndex((op) => op[0] === "open");
    const before = term.log.slice(0, openAt);
    const after  = term.log.slice(openAt + 1);
    const sized  = before.filter((op) => op[0] === "resize").pop();

    assert.ok(openAt > 0, "the terminal is opened after it was sized");
    assert.ok(sized, "the dimensions are set before open()");
    assert.equal(tile.opened, true);
    assert.equal(term.options.fontSize % 0.25, 0, "the font size sits on a quarter pixel");
    assert.ok(term.cols <= 120 && term.rows >= 2);
    assert.deepEqual(after.filter((op) => op[0] === "font"), [], "no font change on the open terminal");
    after.forEach((op) => assert.deepEqual(op, sized, "the post-open proposal check is a no-op resize"));
  }

  testRefittingAnOpenTerminalAtTheSameSizeLeavesItsFontAlone() {
    const { tile, term, fit, host } = this.sizingTile(900, 400);
    tile._fitInto(term, fit, host, "opened");
    term.log.length = 0;

    tile._fitInto(term, fit, host, "opened");

    assert.deepEqual(term.log.filter((op) => op[0] === "font" || op[0] === "open"), []);
  }

  testANarrowTileShrinksTheFontToKeepItsColumns() {
    const { tile, term, fit, host } = this.sizingTile(500, 300);

    tile._fitInto(term, fit, host, "opened");

    assert.ok(term.options.fontSize < 8, `font ${term.options.fontSize} shrank below the 8 px floor of the width rule`);
    assert.ok(term.options.fontSize >= 7);
    assert.equal(term.options.fontSize % 0.25, 0);
  }

  testAnUnmeasuredTileIsLeftClosed() {
    const { tile, term, fit } = this.sizingTile(0, 0);

    tile._fitInto(term, fit, { clientWidth: 0, clientHeight: 400 }, "opened");

    assert.deepEqual(term.log, []);
    assert.equal(tile.opened, false);
  }

  testThePageBuildsOneHiddenGaugeAndReusesIt() {
    const before     = this.document.body.children.length;
    const runConsole = this.console();

    const gauge = runConsole.gauge;

    assert.equal(runConsole.gauge, gauge);
    assert.equal(this.document.body.children.length, before + 1);
    assert.match(gauge.host.style.cssText, /visibility:hidden/);
  }

  testTilesAreFittedOnePerAnimationFrame() {
    const runConsole = this.console();
    const order      = [];
    ["a", "b", "c"].forEach((id) => runConsole.tiles.set(id, { fit: () => order.push(id) }));

    runConsole.queueFit();
    assert.deepEqual(order, [], "nothing is fitted before the debounce timer fires");

    this.flush(this.timers);
    assert.deepEqual(order, ["a"]);

    this.flush(this.frames);
    assert.deepEqual(order, ["a", "b"]);

    this.flush(this.frames);
    assert.deepEqual(order, ["a", "b", "c"]);
    assert.equal(this.frames.size, 0);
  }

  testANewFitRequestCancelsThePendingFrames() {
    const runConsole = this.console();
    const order      = [];
    ["a", "b", "c"].forEach((id) => runConsole.tiles.set(id, { fit: () => order.push(id) }));

    runConsole.queueFit();
    this.flush(this.timers);
    runConsole.queueFit();

    assert.equal(this.frames.size, 0, "the frame chain of the earlier request is cancelled");
    this.flush(this.timers);
    this.flush(this.frames);
    this.flush(this.frames);
    assert.deepEqual(order, ["a", "a", "b", "c"]);
  }

  async run() {
    const cases = Object.getOwnPropertyNames(ConsoleClientSuite.prototype).filter((name) => name.startsWith("test"));
    let failed  = 0;

    for (const name of cases) {
      try {
        await this[name]();
        process.stdout.write(`ok   ${name}\n`);
      } catch (error) {
        failed += 1;
        process.stdout.write(`FAIL ${name}\n     ${error.message}\n`);
      }
    }

    process.stdout.write(failed ? `${failed} failing\n` : `all ${cases.length} passing\n`);
    process.exit(failed ? 1 : 0);
  }
}

const here = path.dirname(fileURLToPath(import.meta.url));
await new ConsoleClientSuite(fs.readFileSync(path.join(here, "..", "..", "webui", "static", "js", "console.js"), "utf-8")).run();
