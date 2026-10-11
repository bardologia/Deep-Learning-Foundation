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
    this.hub = new FakeStreamHub();

    const sandbox = { window: {}, console, StreamHub: { instance: () => this.hub } };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(`${source}\nglobalThis.classes = { ConsoleTile, RunConsole };`, sandbox, { filename: "console.js" });

    this.classes = sandbox.classes;
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
